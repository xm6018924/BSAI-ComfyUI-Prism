# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— ComfyUI 节点定义

节点一览（菜单：BSAI/Prism）：
  BSAIPrismLoader         加载 MOVA 基底 + preview_alpha/beta 权重并配置
                          Prism Block Sparse Attention（含互斥校验）
  BSAIPrismSampler        图文 -> 视频 + 音频（完整 denoising 流程）
  BSAIPrismVideoEncode    IMAGE + AUDIO -> mp4（ffmpeg 合成，可独立使用）
  BSAIPrismWeightsDownload  选择性下载 208GB 权重（支持 hf-mirror）
  BSAIPrismUnload         释放进程内缓存的模型，归还显存/内存

输出格式约定：
  IMAGE   : [B, H, W, 3] float32（0..1）
  AUDIO   : {"waveform": [B, C, 1, T] float32（-1..1）, "sample_rate": int}
  STRING  : 视频文件绝对路径（mp4）
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import folder_paths
import comfy.model_management as mm

from .bsa_config import (
    BSAParams,
    DEFAULT_NEGATIVE_PROMPT,
    RESOLUTION_ORDER,
    RESOLUTION_PROFILES,
    _LEGACY_RESOLUTION_MAP,
)
from . import model_manager
from .ffmpeg_helper import save_video_with_audio
from .download_weights import (
    COMPONENT_PATTERNS,
    PRISM_MODELS_SUBDIR,
    download_component,
)

PRISM_MODEL_TYPE = "BSAI_PRISM_MODEL"

# Fixed Prism weight root (under ComfyUI models/diffusers/BSAI-Prism)
_PRISM_ROOT = os.path.join(folder_paths.models_dir, PRISM_MODELS_SUBDIR)


def _scan_base_models():
    """扫描 Prism 根目录下所有 pretrained_models/<name> 基底目录。"""
    bases = []
    pretrained_dir = os.path.join(_PRISM_ROOT, "pretrained_models")
    if os.path.isdir(pretrained_dir):
        for name in sorted(os.listdir(pretrained_dir)):
            sub = os.path.join(pretrained_dir, name)
            if os.path.isfile(os.path.join(sub, "model_index.json")):
                bases.append(name)
    return bases or ["MOVA-360p"]


def _scan_resume_ckpts():
    """扫描 Prism 根目录下所有 preview_*/diffusion_pytorch_model.safetensors。"""
    ckpts = []
    if os.path.isdir(_PRISM_ROOT):
        for name in sorted(os.listdir(_PRISM_ROOT)):
            sub = os.path.join(_PRISM_ROOT, name)
            ckpt = os.path.join(sub, "diffusion_pytorch_model.safetensors")
            if os.path.isfile(ckpt):
                ckpts.append(name)
    return ckpts or ["preview_alpha"]


def _resolve_base_path(base_name):
    """Resolve the selected base dropdown name to a folder under the Prism root.

    Only a bare folder name is accepted; any path separators are stripped so the
    resolved directory always stays inside <models>/BSAI-Prism/pretrained_models.
    """
    name = os.path.basename(str(base_name).replace("\\", "/"))
    return os.path.join(_PRISM_ROOT, "pretrained_models", name)


def _resolve_resume_ckpt(ckpt_name):
    """Resolve the selected checkpoint dropdown name to a file under the Prism root."""
    if not ckpt_name or ckpt_name in ("(None)", "（无）", "none"):
        return ""
    name = os.path.basename(str(ckpt_name).replace("\\", "/"))
    return os.path.join(_PRISM_ROOT, name, "diffusion_pytorch_model.safetensors")


def _log(msg):
    print(f"[BSAI Prism] {msg}")


# ---------------------------------------------------------------------------
# 类型桥：加载器返回的模型句柄
# ---------------------------------------------------------------------------

@dataclass
class PrismModelHandle:
    key: str
    base_model_path: str
    resume_ckpt: str
    device: str
    offload: str
    bsa: BSAParams


# ---------------------------------------------------------------------------
# 1. 加载节点
# ---------------------------------------------------------------------------

class BSAIPrismLoader:
    @classmethod
    def INPUT_TYPES(cls):
        base_models = _scan_base_models()
        # 末项 "（无）" 为旧版哨兵标签：兼容老工作流 JSON 保存值
        resume_ckpts = _scan_resume_ckpts() + ["(None)", "（无）"]
        devices = ["cuda:0"] if torch.cuda.is_available() else []
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return {
            "required": {
                "base_model": (base_models, {
                    "default": base_models[0],
                    "tooltip": "MOVA base model (architecture skeleton, no fine-tuned weights). "
                               "Defines the overall structure, VAE, text_encoder and other base components.",
                }),
                "resume_ckpt": (resume_ckpts, {
                    "default": resume_ckpts[0],
                    "tooltip": "Prism fine-tuned weights (preview_alpha / preview_beta). "
                               "Loaded on top of the base; determines generation quality and style. "
                               "Choose (None) to use the base only (not recommended, poor quality).",
                }),
                "device": (devices or ["N/A"], {
                    "default": "cuda:0" if devices else "N/A",
                    "tooltip": "Runtime device. Prism depends on flash-attn; NVIDIA CUDA only.",
                }),
                "offload": (["cpu", "none"], {
                    "default": "cpu",
                    "tooltip": "cpu = encoder/VAE moved in and out of GPU per stage, saves VRAM on a single GPU (official default); "
                               "none = fully resident on GPU, higher peak usage.",
                }),
                "enable_bsa": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Prism Block Sparse Attention (video self-attention top-k/top-p sparsity).",
                }),
                "bsa_sparsity": ("FLOAT", {
                    "default": 0.75, "min": 0.1, "max": 0.95, "step": 0.01,
                    "tooltip": "top-k sparsity (official recommended 0.93 / 0.85 / 0.75; smaller = sparser/faster).",
                }),
                "bsa_chunk_3d_shape_q": ("STRING", {
                    "default": "4 4 4", "multiline": False,
                    "tooltip": "3D macro-block shape (T H W) for video self-attention Q; official optimum 4 4 4.",
                }),
                "bsa_chunk_3d_shape_k": ("STRING", {
                    "default": "4 4 4", "multiline": False,
                    "tooltip": "3D macro-block shape (T H W) for video self-attention K; official optimum 4 4 4.",
                }),
                "bsa_cdf_threshold": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "CDF threshold for mixed Top-k + Top-p (official 0.16~0.4, optimum 0.2). "
                               "0 = pure Top-k.",
                }),
                "enable_ivpq_dynamic_block": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prism core: anisotropic block shape adapted to local content variance and audio-video coupling. "
                               "When enabled, it is mutually exclusive with audio/variance guidance and Taylor/Rectified (conflicts raise an error).",
                }),
                "enable_layer_adaptive_dynamic_block": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Shallow fixed blocks / deep dynamic blocks + shallow->deep audio caching (only effective when dynamic blocks are on).",
                }),
                "sparse_high_noise_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "true = sparse attention only on the high-noise expert video_dit; "
                               "the low-noise expert video_dit_2 keeps full attention.",
                }),
                "enable_bsa_v2a": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Bridge v2a cross-attention BSA (Q=audio, K=video 3D blocking).",
                }),
                "bsa_v2a_sparsity": ("FLOAT", {
                    "default": 0.875, "min": 0.1, "max": 0.95, "step": 0.01,
                }),
                "bsa_v2a_audio_chunk_size": ("INT", {
                    "default": 64, "min": 32, "max": 512, "step": 32,
                    "tooltip": "audio Q block size (multiple of 64).",
                }),
                "bsa_v2a_chunk_3d_shape_k": ("STRING", {
                    "default": "4 4 4", "multiline": False,
                }),
            },
            "optional": {
                "enable_audio_guidance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Audio guidance BSA (Path A; mutually exclusive with dynamic blocks).",
                }),
                "audio_boost_gamma": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 4.0, "step": 0.1,
                }),
                "enable_variance_guidance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Channel variance guidance BSA (mutually exclusive with dynamic blocks).",
                }),
                "variance_boost_gamma": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 4.0, "step": 0.1,
                }),
                "enable_taylor_sparse_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Taylor sparse-attention bias correction (mutually exclusive with Rectified and with dynamic blocks).",
                }),
                "enable_rectified_sparse_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Rectified sparse-attention bias correction (mutually exclusive with Taylor and with dynamic blocks).",
                }),
            },
        }

    RETURN_TYPES = (PRISM_MODEL_TYPE, "STRING")
    RETURN_NAMES = ("prism_model", "status")
    FUNCTION = "load"
    CATEGORY = "BSAI/Prism"

    def load(
        self,
        base_model,
        resume_ckpt,
        device,
        offload,
        enable_bsa,
        bsa_sparsity,
        bsa_chunk_3d_shape_q,
        bsa_chunk_3d_shape_k,
        bsa_cdf_threshold,
        enable_ivpq_dynamic_block,
        enable_layer_adaptive_dynamic_block,
        sparse_high_noise_only,
        enable_bsa_v2a,
        bsa_v2a_sparsity,
        bsa_v2a_audio_chunk_size,
        bsa_v2a_chunk_3d_shape_k,
        enable_audio_guidance=False,
        audio_boost_gamma=2.0,
        enable_variance_guidance=False,
        variance_boost_gamma=2.0,
        enable_taylor_sparse_attn=False,
        enable_rectified_sparse_attn=False,
    ):
        # ---- Runtime checks (no silent fallback) ----
        if not torch.cuda.is_available():
            raise RuntimeError(
                "BSAI Prism requires NVIDIA CUDA (flash-attn dependency). "
                "Run ComfyUI in the python_embeded_cuda environment (torch+cu130)."
            )

        # Ported from richservo/comfyui-mova: release other ComfyUI-managed models before loading
        mm.unload_all_models()
        if device in ("N/A", "") or not device.startswith("cuda"):
            raise ValueError(f"Invalid device: {device}")
        dev_idx = int(device.split(":")[-1])
        if dev_idx >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {dev_idx} out of range ({torch.cuda.device_count()} device(s) available)."
            )

        # Resolve base and checkpoint paths (dropdown name -> absolute path)
        base_model_path = _resolve_base_path(base_model)
        resume_ckpt_path = _resolve_resume_ckpt(resume_ckpt)

        base_model_path = os.path.abspath(base_model_path)
        if not os.path.isfile(os.path.join(base_model_path, "model_index.json")):
            raise FileNotFoundError(
                f"MOVA base not found: {base_model_path} (must contain model_index.json). "
                "Use the BSAIPrismWeightsDownload node, or check the path."
            )
        if resume_ckpt_path and not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(f"resume_ckpt not found: {resume_ckpt_path}")

        bsa = BSAParams(
            enable_bsa=enable_bsa,
            bsa_sparsity=bsa_sparsity,
            bsa_chunk_3d_shape_q=bsa_chunk_3d_shape_q,
            bsa_chunk_3d_shape_k=bsa_chunk_3d_shape_k,
            bsa_cdf_threshold=bsa_cdf_threshold,
            enable_bsa_v2a=enable_bsa_v2a,
            bsa_v2a_sparsity=bsa_v2a_sparsity,
            bsa_v2a_audio_chunk_size=bsa_v2a_audio_chunk_size,
            bsa_v2a_chunk_3d_shape_k=bsa_v2a_chunk_3d_shape_k,
            enable_ivpq_dynamic_block=enable_ivpq_dynamic_block,
            enable_layer_adaptive_dynamic_block=enable_layer_adaptive_dynamic_block,
            sparse_high_noise_only=sparse_high_noise_only,
            enable_audio_guidance=enable_audio_guidance,
            audio_boost_gamma=audio_boost_gamma,
            enable_variance_guidance=enable_variance_guidance,
            variance_boost_gamma=variance_boost_gamma,
            enable_taylor_sparse_attn=enable_taylor_sparse_attn,
            enable_rectified_sparse_attn=enable_rectified_sparse_attn,
        )
        bsa.validate()  # 互斥/取值校验，冲突直接抛错

        import dataclasses
        bsa_hash = str(
            hash(tuple(sorted(
                (f.name, str(getattr(bsa, f.name)))
                for f in dataclasses.fields(BSAParams)
            )))
        )
        key = model_manager.model_cache_key(
            base_model_path, resume_ckpt_path, device, offload, bsa_hash
        )

        cached = model_manager.get_cached(key)
        if cached is not None:
            _log(f"cache hit ({base_model_path} / {resume_ckpt_path or 'base-only'})")
            handle = PrismModelHandle(
                key=key, base_model_path=base_model_path, resume_ckpt=resume_ckpt_path,
                device=device, offload=offload, bsa=bsa,
            )
            return (handle, f"cache-hit | device={device} offload={offload}")

        bridge, extras = model_manager.load_prism_model(
            base_model_path, resume_ckpt_path, device, offload, bsa
        )
        model_manager.cache_put(
            key,
            model_manager._LoadedModel(
                bridge=bridge, extras=extras, key=key,
                device=torch.device(device), offload=offload,
            ),
        )
        handle = PrismModelHandle(
            key=key, base_model_path=base_model_path, resume_ckpt=resume_ckpt_path,
            device=device, offload=offload, bsa=bsa,
        )
        return (
            handle,
            f"loaded | device={device} offload={offload} "
            f"bsa_sparsity={bsa_sparsity} ivpq={enable_ivpq_dynamic_block}",
        )


# ---------------------------------------------------------------------------
# 2. 采样节点
# ---------------------------------------------------------------------------

class BSAIPrismSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prism_model": (PRISM_MODEL_TYPE, {}),
                "prompt": ("STRING", {
                    "default": "A man with short gray hair plays a red electric guitar.",
                    "multiline": True,
                    "tooltip": "Video prompt; supports structured tags <music> <sfx> <speech> <text> <lyrics>.",
                }),
                "audio_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Audio prompt; leave empty to reuse the video prompt.",
                }),
                "negative_prompt": ("STRING", {
                    "default": DEFAULT_NEGATIVE_PROMPT,
                    "multiline": True,
                }),
                "resolution": (RESOLUTION_ORDER, {
                    "default": "Landscape 720p (1280x720)",
                    "tooltip": "Resolution tier; choose Custom to manually adjust width/height below.",
                }),
                "width": ("INT", {"default": 1280, "min": 448, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 720, "min": 256, "max": 2048, "step": 16}),
                "num_frames": ("INT", {
                    "default": 205, "min": 17, "max": 1024, "step": 4,
                    "tooltip": "Number of frames (pixel space). Auto-aligned to (n-1) % 4 == 0.",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "num_inference_steps": ("INT", {
                    "default": 50, "min": 1, "max": 200, "step": 1,
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance strength; 1.0 = no guidance.",
                }),
                "visual_shift": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Video noise-schedule shift. 720p~7.0, 1080p~9~13, 2K~13~17.",
                }),
                "audio_shift": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF, "step": 1}),
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Tile VAE decoding (2K/1080p OOM prevention). Tier presets override this automatically.",
                }),
                "tile_sample_min_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
                "tile_sample_stride": ("INT", {"default": 192, "min": 32, "max": 1024, "step": 32}),
                "remove_video_dit_after_switch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Release the high-noise expert video_dit after the boundary switch to save VRAM (irreversible; internal release only).",
                }),
                "audio_strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Audio injection strength (lip-sync). 1.0 = fully preserve input audio, 0.0 = ignore audio and generate from scratch. "
                               "Requires a connected audio input.",
                }),
                "auto_duration": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-compute frame count from the input audio length (requires a connected audio input).",
                }),
                "save_video": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "BSAI_Prism"}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Reference image (i2v first frame). Leave empty = text-to-video (denoise from a black frame). Center-cropped and resized to the target resolution.",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Pre-generated audio (speech/music) for lip-sync. Once connected, use audio_strength to control how much is preserved.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "BSAI/Prism"

    def _resolve_geometry(self, resolution, width, height, visual_shift, audio_shift,
                          enable_vae_tiling, num_frames):
        """Tier presets override geometry and shift; Custom uses the widget values."""
        if resolution == "Custom":
            return int(height), int(width), float(visual_shift), float(
                audio_shift), bool(enable_vae_tiling)
        prof = RESOLUTION_PROFILES[resolution]
        return prof.height, prof.width, prof.visual_shift, prof.audio_shift, prof.vae_tiling

    @staticmethod
    def _snap(value, base, label):
        snapped = int(value) - (int(value) % base)
        if snapped != int(value):
            _log(f"{label} {value} -> {snapped} (must be a multiple of {base})")
        return snapped

    @staticmethod
    def _crop_and_resize(img, height, width):
        from PIL import Image

        w, h = img.size
        target_ratio = width / height
        img_ratio = w / h
        if img_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        elif img_ratio < target_ratio:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        return img.resize((width, height), Image.LANCZOS)

    def sample(
        self,
        prism_model: PrismModelHandle,
        prompt,
        audio_prompt="",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        resolution="Landscape 720p (1280x720)",
        width=1280,
        height=720,
        num_frames=205,
        fps=24.0,
        num_inference_steps=50,
        cfg_scale=5.0,
        visual_shift=7.0,
        audio_shift=7.0,
        seed=42,
        enable_vae_tiling=False,
        tile_sample_min_size=256,
        tile_sample_stride=192,
        remove_video_dit_after_switch=False,
        audio_strength=0.8,
        auto_duration=False,
        save_video=True,
        filename_prefix="BSAI_Prism",
        image=None,
        audio=None,
    ):
        from PIL import Image

        # 旧版中文档位标签归一化到英文标签（语义/参数完全一致）
        resolution = _LEGACY_RESOLUTION_MAP.get(resolution, resolution)

        loaded = model_manager.get_cached(prism_model.key)
        if loaded is None:
            raise RuntimeError(
                "Model cache lost (it may have been released by BSAIPrismUnload or a process restart); "
                "re-run the loader node."
            )

        h, w, v_shift, a_shift, tiling = self._resolve_geometry(
            resolution, width, height, visual_shift, audio_shift, enable_vae_tiling, num_frames
        )
        # 高度/宽度对齐 16（Wan VAE 空间压缩 8 倍，check_inputs 要求 16 的倍数）
        h = self._snap(h, 16, "height")
        w = self._snap(w, 16, "width")

        # auto_duration: 根据输入音频长度自动计算帧数
        if auto_duration and audio is not None:
            audio_samples = audio["waveform"].shape[-1]
            audio_sr = audio["sample_rate"]
            duration = audio_samples / audio_sr
            raw_frames = int(duration * fps)
            num_frames = max(17, ((raw_frames - 1) // 4) * 4 + 1)
            _log(f"auto_duration: audio {duration:.2f}s -> {num_frames} frames")

        _log(
            f"sampling: {h}x{w} @ {fps}fps, {num_frames} frames, {num_inference_steps} steps, "
            f"cfg={cfg_scale}, visual_shift={v_shift}, audio_shift={a_shift}, seed={seed}, "
            f"tiling={tiling}, bsa_sparsity={prism_model.bsa.bsa_sparsity}"
        )

        pipeline = model_manager.build_pipeline(
            loaded.bridge, loaded.extras, torch.device(prism_model.device), prism_model.offload
        )

        # 参考图：ComfyUI IMAGE [B,H,W,3] -> PIL
        if image is None:
            # t2v: 纯噪声初始化（pipeline 内部 condition 全零）
            _log("[t2v] image not connected; pure-noise text-to-video mode")
            ref_img = None
        else:
            ref_np = image[0].cpu().numpy()
            ref_np = (np.clip(ref_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            ref_img = Image.fromarray(ref_np).convert("RGB")
            ref_img = self._crop_and_resize(ref_img, height=h, width=w)

        # 帧数对齐 (n-1) % 4 == 0
        raw_num_frames = int(num_frames)
        num_frames = raw_num_frames - (raw_num_frames - 1) % pipeline.vae_scale_factor_temporal
        if num_frames != raw_num_frames:
            _log(f"num_frames {raw_num_frames} -> {num_frames} (must satisfy (n-1) % 4 == 0)")

        # 需要对齐到 check_inputs 的另一个条件：(n-1) % temporal == 0 已满足；
        # 若用户给 < 5 帧则兜底
        if num_frames < pipeline.vae_scale_factor_temporal + 1:
            num_frames = pipeline.vae_scale_factor_temporal + 1

        torch.manual_seed(seed)

        # ---- 音频注入（lip-sync，移植自 richservo/comfyui-mova）----
        _orig_prepare_audio = None
        _orig_set_pp = None
        _clean_audio_latents = None
        _audio_device = torch.device(prism_model.device)

        if audio is not None:
            import torchaudio

            # ComfyUI AUDIO [B, C, T] → 单声道 waveform [1, 1, T]
            waveform = audio["waveform"].to(_audio_device).float()
            sr_in = audio["sample_rate"]
            if waveform.shape[1] > 1:
                waveform = waveform.mean(dim=1, keepdim=True)

            target_sr = pipeline.audio_sample_rate
            target_samples = int(target_sr * num_frames / fps)

            if sr_in != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr_in, target_sr)
            if waveform.shape[-1] < target_samples:
                pad = torch.zeros(1, 1, target_samples - waveform.shape[-1], device=_audio_device)
                waveform = torch.cat([waveform, pad], dim=-1)
            elif waveform.shape[-1] > target_samples:
                waveform = waveform[:, :, :target_samples]

            # 通过 audio_vae 编码得到干净音频 latents
            pipeline.audio_vae.to(_audio_device)
            with torch.no_grad():
                x_pad = pipeline.audio_vae.preprocess(waveform, sample_rate=target_sr)
                x_pad = x_pad.to(dtype=pipeline.audio_vae.dtype)
                enc_out = pipeline.audio_vae.encode(x_pad)
                z = enc_out[0] if isinstance(enc_out, tuple) else enc_out
                _clean_audio_latents = z.mode().float()
            pipeline.audio_vae.to("cpu")
            _log(f"audio injection: strength={audio_strength}, latents {_clean_audio_latents.shape}")

            denoise_strength = max(1.0 - audio_strength, 0.001)

            # Monkey-patch scheduler：注入 audio_denoising_strength
            _orig_set_pp = pipeline.scheduler.set_pair_postprocess_by_name
            def _patched_set_pp(name, **kwargs):
                if name == "dual_sigma_shift":
                    kwargs["audio_denoising_strength"] = denoise_strength
                return _orig_set_pp(name, **kwargs)
            pipeline.scheduler.set_pair_postprocess_by_name = _patched_set_pp

            # Monkey-patch prepare_audio_latents：用编码后的 latents + 噪声替代纯随机噪声
            _orig_prepare_audio = pipeline.prepare_audio_latents
            def _patched_prepare_audio(a, bs, nc, ns, dt, dev, gen=None, latents=None):
                pairs = pipeline.scheduler.get_pairs(source="sigmas")
                first_sigma = pairs[0, 1].item()
                noise = torch.randn_like(_clean_audio_latents)
                injected = (1.0 - first_sigma) * _clean_audio_latents + first_sigma * noise
                return injected.to(device=dev, dtype=dt)
            pipeline.prepare_audio_latents = _patched_prepare_audio

        try:
            frames, audio_out = pipeline(
                prompt=prompt,
                image=ref_img,
                audio_prompt=audio_prompt if audio_prompt.strip() else None,
                negative_prompt=negative_prompt,
                seed=seed,
                height=h,
                width=w,
                num_frames=num_frames,
                video_fps=fps,
                num_inference_steps=num_inference_steps,
                visual_shift=v_shift,
                audio_shift=a_shift,
                cfg_scale=cfg_scale,
                remove_video_dit=remove_video_dit_after_switch,
                enable_vae_tiling=tiling,
                vae_tile_sample_min_size=tile_sample_min_size,
                vae_tile_sample_stride=tile_sample_stride,
            )
        finally:
            # 恢复 monkey-patch
            if _orig_prepare_audio is not None:
                pipeline.prepare_audio_latents = _orig_prepare_audio
            if _orig_set_pp is not None:
                pipeline.scheduler.set_pair_postprocess_by_name = _orig_set_pp

        # ---- 转 ComfyUI 格式 ----
        images = torch.from_numpy(
            np.stack([np.array(f) for f in frames[0]], axis=0).astype(np.float32) / 255.0
        )
        audio_wav = audio_out[0].detach().cpu().float()  # [1, T]（DAC 单声道）
        # ComfyUI 标准 AUDIO 格式：[batch, channels, samples]
        if audio_wav.dim() == 1:
            audio_wav = audio_wav.unsqueeze(0).unsqueeze(0)  # [T] -> [1, 1, T]
        elif audio_wav.dim() == 2:
            audio_wav = audio_wav.unsqueeze(0)  # [C, T] -> [1, C, T]
        audio_dict = {
            "waveform": audio_wav,
            "sample_rate": int(pipeline.audio_sample_rate),
        }

        video_path = ""
        videos_ui = []
        mm.soft_empty_cache()
        if save_video:
            audio_save = audio_out[0].cpu().squeeze()
            full_folder, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(), images.shape[1], images.shape[2]
            )
            if subfolder != "BSAI_Prism":
                subfolder = os.path.join("BSAI_Prism", filename_prefix)
            out_dir = os.path.join(folder_paths.get_output_directory(), "BSAI_Prism")
            os.makedirs(out_dir, exist_ok=True)
            video_path = os.path.join(out_dir, f"{filename_prefix}_{counter:05d}.mp4")
            save_video_with_audio(
                frames[0], audio_save, video_path, fps=fps, sample_rate=pipeline.audio_sample_rate
            )
            videos_ui = [{
                "filename": os.path.basename(video_path),
                "subfolder": "BSAI_Prism",
                "type": "output",
            }]
            _log(f"saved: {video_path}")

        return {"ui": {"videos": videos_ui}, "result": (images, audio_dict, video_path)}


# ---------------------------------------------------------------------------
# 3. 音视频合成节点
# ---------------------------------------------------------------------------

class BSAIPrismVideoEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "filename_prefix": ("STRING", {"default": "BSAI_Prism"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = "BSAI/Prism"

    def encode(self, images, audio, fps=24.0, filename_prefix="BSAI_Prism"):
        from PIL import Image

        frames = [
            Image.fromarray((np.clip(img.cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8))
            for img in images
        ]
        waveform = audio["waveform"]  # [B, C, 1, T]
        sr = int(audio["sample_rate"])
        audio_t = waveform.squeeze(2).squeeze(0)  # [C, T] -> 取 batch 首条
        if audio_t.ndim == 2:
            audio_t = audio_t[0] if audio_t.shape[0] == 1 else audio_t.mean(0)
        audio_t = audio_t.detach().cpu().float()

        out_dir = os.path.join(folder_paths.get_output_directory(), "BSAI_Prism")
        os.makedirs(out_dir, exist_ok=True)
        _, _, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1]
        )
        video_path = os.path.join(out_dir, f"{filename_prefix}_{counter:05d}.mp4")
        save_video_with_audio(frames, audio_t, video_path, fps=fps, sample_rate=sr)
        _log(f"muxed: {video_path}")
        return {
            "ui": {"videos": [{
                "filename": os.path.basename(video_path),
                "subfolder": "BSAI_Prism",
                "type": "output",
            }]},
            "result": (video_path,),
        }


# ---------------------------------------------------------------------------
# 4. 权重下载节点
# ---------------------------------------------------------------------------

class BSAIPrismWeightsDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": (list(COMPONENT_PATTERNS.keys()), {
                    "default": ["base"],
                    "tooltip": "base = MOVA-360p base (~40GB); preview_alpha/beta are 65.3GB each. "
                               "Weights are always downloaded into the fixed folder "
                               "models/diffusers/BSAI-Prism; the destination is not configurable.",
                }),
                "use_mirror": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use HF_ENDPOINT=https://hf-mirror.com mirror for downloads.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("target_dir",)
    FUNCTION = "download"
    CATEGORY = "BSAI/Prism"

    def download(self, components, use_mirror=False, target_dir=None):
        # target_dir 仅为兼容老工作流 JSON 保留（已删除的旧控件），任何传入路径一律忽略：
        # 下载固定落到 models/diffusers/BSAI-Prism 根目录。
        results = []
        for comp in components:
            if comp not in COMPONENT_PATTERNS:
                raise ValueError(f"Unknown component: {comp}")
            # target_dir is intentionally ignored: downloads are pinned to the fixed Prism root.
            snapshot = download_component(comp, use_mirror=use_mirror)
            results.append(snapshot)
        return ("; ".join(results),)


# ---------------------------------------------------------------------------
# 5. 卸载节点
# ---------------------------------------------------------------------------

class BSAIPrismUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("released_count", "status")
    FUNCTION = "unload"
    CATEGORY = "BSAI/Prism"

    def unload(self):
        count = model_manager.unload_all()
        torch.cuda.empty_cache()
        return (count, f"Released {count} model cache(s)")


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BSAIPrismLoader": BSAIPrismLoader,
    "BSAIPrismSampler": BSAIPrismSampler,
    "BSAIPrismVideoEncode": BSAIPrismVideoEncode,
    "BSAIPrismWeightsDownload": BSAIPrismWeightsDownload,
    "BSAIPrismUnload": BSAIPrismUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIPrismLoader": "BSAI Prism Model Loader",
    "BSAIPrismSampler": "BSAI Prism Sampler (Video + Audio)",
    "BSAIPrismVideoEncode": "BSAI Prism Video/Audio Mux",
    "BSAIPrismWeightsDownload": "BSAI Prism Weights Download",
    "BSAIPrismUnload": "BSAI Prism Unload Models",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
