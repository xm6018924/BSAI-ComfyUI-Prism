# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— 模型加载 / 卸载 / 缓存管理

Prism 权重体积巨大（MOVA-360p 基底 + preview_alpha/beta 各 65GB），
在 ComfyUI 进程内用模块级缓存按 (base, ckpt, bsa_hash) 复用已加载模型，
避免重复加载导致几十 GB 的 RAM/VRAM 反复占用。
"""
import ctypes
import hashlib
import json
import os
import struct
import sys
import time

import torch

# 基底映射之外还要留给 torch / CUDA / 其它节点的提交量余量（GB）。
_COMMIT_MARGIN_GB = 8.0


class _LoadedModel:
    """一次加载的模型 + 附加组件 + 生成时配置闭包。"""

    __slots__ = ("bridge", "extras", "key", "loaded_at", "device", "offload")

    def __init__(self, bridge, extras, key, device, offload):
        self.bridge = bridge
        self.extras = extras
        self.key = key
        self.loaded_at = time.time()
        self.device = device
        self.offload = offload


# 进程内模型缓存：{key: _LoadedModel}
_MODEL_CACHE = {}


def model_cache_key(base_model_path, resume_ckpt, device, offload, bsa_hash):
    return hashlib.sha1(
        f"{base_model_path}|{resume_ckpt}|{device}|{offload}|{bsa_hash}".encode("utf-8")
    ).hexdigest()


def get_cached(key):
    return _MODEL_CACHE.get(key)


def cache_put(key, loaded: _LoadedModel):
    # 同一 key 重复加载：先释放旧的，防止显存/内存泄漏
    old = _MODEL_CACHE.pop(key, None)
    if old is not None:
        _free_loaded(old)
    _MODEL_CACHE[key] = loaded


def cache_keys():
    return list(_MODEL_CACHE.keys())


def _free_loaded(loaded: _LoadedModel):
    try:
        import gc

        bridge = loaded.bridge
        extras = loaded.extras
        bridge.cpu()
        for name in ("video_vae", "audio_vae", "text_encoder"):
            comp = extras.get(name)
            if comp is not None:
                try:
                    comp.cpu()
                except Exception:
                    pass
        del bridge
        for name in list(extras.keys()):
            extras[name] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _materialize_model(model):
    """将 mmap 只读映射的权重真正加载到 RAM 中。
    遍历所有参数和缓冲区，用 clone 替换 data，强制分配物理内存。
    """
    import gc
    count = 0
    total_bytes = 0
    for p in model.parameters():
        if p.data.device.type == "meta":
            continue
        p.data = p.data.clone()
        count += 1
        total_bytes += p.data.numel() * p.data.element_size()
    for buf in model.buffers():
        if buf.data.device.type == "meta":
            continue
        buf.data = buf.data.clone()
        count += 1
        total_bytes += buf.data.numel() * buf.data.element_size()
    gc.collect()
    return count, total_bytes / (1024**3)


def unload_all() -> int:
    """释放全部缓存模型，返回释放的数量。"""
    keys = list(_MODEL_CACHE.keys())
    for key in keys:
        loaded = _MODEL_CACHE.pop(key, None)
        if loaded is not None:
            _free_loaded(loaded)
    return len(keys)


def _safetensors_bytes(root):
    """root 下所有 safetensors 的字节数：映射它们就要占这么多提交量。"""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(dirpath, name))
    return total


def _commit_headroom_gb():
    """Windows 上返回 (进程提交上限, 剩余可提交量) GB，其它平台返回 None。

    基底是按 PAGE_WRITECOPY 映射的，整份映射在建立时就计入提交量；本机只有
    63.5GB 物理内存 + 128GB 页面文件，若系统里另有进程占着大量提交量，加载中途
    会因为无法满足缺页而直接 access violation。
    """

    class _MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    if sys.platform != "win32":
        return None
    status = _MemStatus()
    status.dwLength = ctypes.sizeof(_MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullTotalPageFile / 1024**3, status.ullAvailPageFile / 1024**3


def _warn_if_commit_is_tight(need_gb, logger):
    """need_gb = 这次加载要一次性占住的提交量（映射多少权重就要多少）。"""
    headroom = _commit_headroom_gb()
    if headroom is None:
        return
    limit_gb, avail_gb = headroom
    if avail_gb < need_gb + _COMMIT_MARGIN_GB:
        logger(
            f"内存余量不足：本次加载需要约 {need_gb:.0f}GB 提交量，当前只剩 "
            f"{avail_gb:.1f}GB / 上限 {limit_gb:.0f}GB。请先关闭其它 ComfyUI 实例、"
            f"浏览器或重启机器，否则加载中途可能直接 access violation 崩掉。"
        )


def _stream_load_state_dict(model, ckpt_path, logger):
    """纯文件 seek/read 逐 tensor 流式覆盖模型权重。

    与 load_state_dict(strict=False) 语义一致（返回 missing / unexpected）。
    不复用 safetensors 的 mmap（Windows 上大文件 mmap 与已加载模型的虚拟内存
    共存时触发 access violation），也绝不整包 materialize（65GB 超过本机 RAM）：
    每个张量按声明的 dtype 先分配好，再把文件字节直接读进这块内存，最后 copy_ 覆盖
    参数。峰值额外内存 = 单个张量大小，全程只有一次内存拷贝，不产生等大的 bytes
    中转缓冲。
    """
    DTYPE_MAP = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "I64": torch.int64,
        "I32": torch.int32,
        "U8": torch.uint8,
    }
    sd = model.state_dict()
    missing = []
    unexpected = []
    t0 = time.time()
    n_covered = 0
    with open(ckpt_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            param = sd.get(name)
            if param is None:
                unexpected.append(name)
                continue
            s, e = meta["data_offsets"]
            dt = DTYPE_MAP.get(meta["dtype"])
            if dt is None:
                raise ValueError(f"未知 dtype {meta['dtype']} @ {name}")
            tensor = torch.empty(tuple(meta["shape"]), dtype=dt)
            f.seek(data_start + s)
            if f.readinto(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())) != e - s:
                raise OSError(f"权重文件不完整：{name} 需要 {e - s} 字节")
            param.copy_(tensor)
            del tensor
            n_covered += 1
        for k in sd:
            if k not in header:
                missing.append(k)
    if logger is not None:
        logger(f"流式覆盖完成：{n_covered} 个参数，耗时 {time.time() - t0:.1f}s")
    return missing, unexpected


def load_prism_model(base_model_path, resume_ckpt, device, offload, bsa_params):
    """加载 MOVABridge + 附加组件，并按 BSA 参数完成运行时配置。

    流程与官方 sample_mova_single.main() 完全一致：
      1. MOVABridge.from_mova_pretrained(base, dtype=bf16)（全组件落 CPU）；
      2. 若提供 resume_ckpt，safetensors state_dict strict=False 覆盖权重；
      3. set_sparse_high_noise_only -> configure_bsa -> 音频/方差引导 ->
         动态块 -> 层自适应配置（含互斥校验，冲突即抛错）。
    返回 (bridge, extras)。
    """
    t0 = time.time()

    class _Logger:
        """兼容 from_mova_pretrained 的 logger.info(...) 接口，输出到 ComfyUI 控制台。"""

        @staticmethod
        def info(msg):
            print(f"[BSAI Prism] {msg}")

        def __call__(self, msg):
            print(f"[BSAI Prism] {msg}")

    _log = _Logger()

    _log.info(f"加载 MOVA 基底：{base_model_path}")
    from prism_runtime.models.modules.mova import MOVABridge, alias_bridge_from_checkpoint

    prism_ckpt = resume_ckpt if (resume_ckpt or "").endswith(".safetensors") else None
    missing = []
    unexpected = []

    # CPU offload 模式下，权重需要频繁在 CPU↔GPU 之间搬运。
    # 只读 mmap 会导致每次搬运都可能触发磁盘 I/O（缺页中断），速度极其缓慢。
    # 因此 offload=cpu 时强制流式加载到实际内存，牺牲加载速度换推理速度。
    force_materialize = (offload == "cpu")

    if prism_ckpt:
        if force_materialize:
            _log(f"[offload=cpu] 流式加载权重到内存（避免 mmap 磁盘瓶颈）：{prism_ckpt}")
            bridge, extras = MOVABridge.from_mova_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, logger=_log,
            )
            missing, unexpected = _stream_load_state_dict(bridge, prism_ckpt, _log)
            # 冻结组件（VAE / text_encoder / audio_vae）也 materialize，
            # 避免 offload 阶段每次搬运触发磁盘 I/O
            for name in ("video_vae", "audio_vae", "text_encoder"):
                comp = extras.get(name)
                if comp is not None:
                    cnt, gb = _materialize_model(comp)
                    _log(f"[offload=cpu] Materialized {name}: {cnt} tensors, {gb:.1f} GB")
        else:
            # 零拷贝：只建结构，参数直接指向 checkpoint 的只读映射。冻结组件（VAE / 文本
            # 编码器）同样只读映射，所以整条路径一次性占住的提交量只有十几 GB，而不是把
            # 72GB 基底按写时复制映射进来（本机 63.5GB 内存 + 128GB 页面文件顶不住）。
            _warn_if_commit_is_tight(
                sum(_safetensors_bytes(os.path.join(base_model_path, sub))
                    for sub in ("video_vae", "audio_vae", "text_encoder")) / 1024**3,
                _log,
            )
            _log(f"加载 Prism 微调权重（只读映射，零拷贝）：{prism_ckpt}")
            bridge, extras = MOVABridge.from_mova_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, logger=_log, empty_dits=True
            )
            missing = alias_bridge_from_checkpoint(bridge, prism_ckpt, _log)
            if missing:
                _log(f"checkpoint 覆盖不全（缺 {len(missing)} 个张量，例如 {missing[0]}），"
                     f"回退为基底权重 + 流式覆盖")
                bridge, extras = MOVABridge.from_mova_pretrained(
                    base_model_path, torch_dtype=torch.bfloat16, logger=_log
                )
                missing, unexpected = _stream_load_state_dict(bridge, prism_ckpt, _log)
    else:
        _warn_if_commit_is_tight(_safetensors_bytes(base_model_path) / 1024**3, _log)
        bridge, extras = MOVABridge.from_mova_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, logger=_log
        )
        if resume_ckpt:
            _log(f"加载 Prism 微调权重：{resume_ckpt}")
            if resume_ckpt.endswith(".pt"):
                state_dict = torch.load(resume_ckpt, map_location="cpu")
                if "module" in state_dict:
                    state_dict = state_dict["module"]
                missing, unexpected = bridge.load_state_dict(state_dict, strict=False)
            else:
                raise ValueError(f"不支持的权重格式（仅 .safetensors / .pt）：{resume_ckpt}")

    if resume_ckpt:
        _log(f"权重覆盖完成。missing={len(missing)} unexpected={len(unexpected)}")

    # ---- 稀疏注意力配置（顺序与官方 main() 一致）----
    bsa_params.validate()

    bridge.set_sparse_high_noise_only(bsa_params.sparse_high_noise_only)

    enable_bsa = bsa_params.enable_bsa
    enable_bsa_v2a = bsa_params.enable_bsa_v2a
    if enable_bsa or enable_bsa_v2a:
        bsa_kwargs = (
            {
                "sparsity": bsa_params.bsa_sparsity,
                "cdf_threshold": bsa_params.bsa_cdf_threshold,
                "chunk_3d_shape_q": bsa_params.chunk_q(),
                "chunk_3d_shape_k": bsa_params.chunk_k(),
            }
            if enable_bsa
            else None
        )
        bsa_v2a_kwargs = (
            {
                "sparsity": bsa_params.bsa_v2a_sparsity,
                "cdf_threshold": None,
                "chunk_size_q": bsa_params.bsa_v2a_audio_chunk_size,
                "chunk_3d_shape_k": bsa_params.chunk_k_v2a(),
            }
            if enable_bsa_v2a
            else None
        )
        bridge.configure_bsa(
            enable_bsa=enable_bsa,
            bsa_params=bsa_kwargs,
            enable_bsa_v2a=enable_bsa_v2a,
            bsa_params_v2a=bsa_v2a_kwargs,
        )

    # 音频引导（Path A / Path B）
    if bsa_params.enable_audio_guidance or bsa_params.enable_audio_weighted_pooling:
        for fused_block in bridge.fusion_blocks:
            sa = fused_block.video_block.self_attn
            sa.bsa_params["audio_boost_gamma"] = bsa_params.audio_boost_gamma
            sa.bsa_params["audio_weighted_lambda"] = bsa_params.audio_weighted_lambda
        for block in bridge.remaining_video_blocks:
            block.self_attn.bsa_params["audio_boost_gamma"] = bsa_params.audio_boost_gamma
            block.self_attn.bsa_params["audio_weighted_lambda"] = bsa_params.audio_weighted_lambda
        bridge.configure_audio_guidance(
            enable=bsa_params.enable_audio_guidance,
            enable_concentration_gate=bsa_params.enable_audio_concentration_gate,
            enable_timestep_reliability_gate=bsa_params.enable_timestep_reliability_gate,
            enable_audio_weighted_pooling=bsa_params.enable_audio_weighted_pooling,
        )

    # 通道方差引导
    if bsa_params.enable_variance_guidance:
        for fused_block in bridge.fusion_blocks:
            fused_block.video_block.self_attn.bsa_params[
                "variance_boost_gamma"
            ] = bsa_params.variance_boost_gamma
        for block in bridge.remaining_video_blocks:
            block.self_attn.bsa_params["variance_boost_gamma"] = bsa_params.variance_boost_gamma
        bridge.configure_variance_guidance(enable=True)

    # 偏差校正（Taylor / Rectified 互斥，validate 已保证）
    if bsa_params.enable_taylor_sparse_attn:
        for fb in bridge.fusion_blocks:
            fb.video_block.self_attn.bsa_params["taylor_alpha_f"] = bsa_params.taylor_alpha_f
        for blk in bridge.remaining_video_blocks:
            blk.self_attn.bsa_params["taylor_alpha_f"] = bsa_params.taylor_alpha_f
        bridge.configure_taylor_sparse_attn(enable=True)
    if bsa_params.enable_rectified_sparse_attn:
        bridge.configure_rectified_sparse_attn(enable=True)

    # 动态块形状（IVPQ，Prism 核心）—— 须在其它引导之后
    if bsa_params.enable_ivpq_dynamic_block:
        for sa in bridge._video_self_attns():
            sa.bsa_params["dynamic_block_lambda_a"] = 0.5
            sa.bsa_params["dynamic_block_tau_128"] = 0.25
            sa.bsa_params["dynamic_block_lambda_128"] = 1.0
        bridge.configure_ivpq_dynamic_block(enable=True)
        if bsa_params.enable_layer_adaptive_dynamic_block:
            split, cache_src = bridge.configure_layer_adaptive_dynamic_block(enable=True)
            _log(f"[BSAI Prism] 层自适应动态块：shallow(<{split}) 固定 / "
                 f"deep(>={split}) 动态，音频缓存源层={cache_src}")

    bridge.eval()
    _log(f"模型加载与配置完成，耗时 {time.time() - t0:.1f}s")
    return bridge, extras


def build_pipeline(bridge, extras, device, offload):
    """构造 MOVAPipeline（与官方 sample_mova_single 相同的两种内存模式）。"""
    from prism_runtime.diffusion.pipelines.mova_pipeline import MOVAPipeline

    if offload == "cpu":
        pipeline = MOVAPipeline(
            transformer=bridge,
            video_vae=extras["video_vae"],
            audio_vae=extras["audio_vae"],
            text_encoder=extras["text_encoder"],
            tokenizer=extras["tokenizer"],
            scheduler=extras["scheduler"],
            boundary_ratio=extras["boundary_ratio"],
            device=device,
        )
        pipeline.enable_cpu_offload(gpu_device=device)
    else:
        bridge = bridge.to(device)
        pipeline = MOVAPipeline(
            transformer=bridge,
            video_vae=extras["video_vae"].to(device),
            audio_vae=extras["audio_vae"].to(device),
            text_encoder=extras["text_encoder"].to(device),
            tokenizer=extras["tokenizer"],
            scheduler=extras["scheduler"],
            boundary_ratio=extras["boundary_ratio"],
            device=device,
        )
    return pipeline
