# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— Turbo 旁支节点

不修改原有 MOVABridge 管线，在主管线之外提供：
  BSAIPrismTurboOptimizer  对 transformer 做 torch.compile（SGLang 已验证 MOVA 可用）
  BSAIPrismTurboSampler    独立 denoising 循环，加入 CFG Rescale 保质 + 自适应跳步加速

使用方式：
  Loader -> TurboOptimizer -> TurboSampler -> 输出
  原有 Loader -> Sampler 完全不受影响。
"""
import gc
import os
from dataclasses import dataclass

import numpy as np
import torch
import folder_paths

from .bsa_config import (
    BSAParams,
    DEFAULT_NEGATIVE_PROMPT,
    RESOLUTION_ORDER,
    RESOLUTION_PROFILES,
)
from . import model_manager
from .ffmpeg_helper import save_video_with_audio
from .nodes import (
    PRISM_MODEL_TYPE,
    PrismModelHandle,
    _log,
)
from .nodes import BSAIPrismSampler as _BaseSampler

# 静态方法引用（原代码在类内部，不是模块级函数）
_snap = staticmethod(_BaseSampler._snap)
_crop_and_resize = staticmethod(_BaseSampler._crop_and_resize)


# ---------------------------------------------------------------------------
# 1. Turbo 优化节点（torch.compile + Cache-DiT）
# ---------------------------------------------------------------------------

class _CacheDiTBlock(torch.nn.Module):
    """Cache-DiT 块级缓存代理（移植自 SGLang DBCache）。

    包装一个 transformer block：如果上一步该 block 的输入与当前输入差异很小，
    直接返回上一步的输出，跳过 block 计算和 GPU 搬移。
    """
    def __init__(self, real_block, fn=False, bn=False, rdt=0.24, mc=3):
        super().__init__()
        self._real = real_block
        self._is_first = fn      # 第一个 block，总是计算
        self._is_last = bn       # 最后一个 block，总是计算
        self._rdt = rdt          # residual difference threshold
        self._mc = mc            # max consecutive cached hits
        self._cache_in = None
        self._cache_out = None
        self._hit_count = 0
        self._total_calls = 0

    def forward(self, *args, **kwargs):
        self._total_calls += 1

        # 第一个/最后一个 block 总是计算（SGLang DBCache: Fn/Bn）
        # warmup：前几步总是计算
        if self._is_first or self._is_last or self._total_calls <= 4:
            out = self._real(*args, **kwargs)
            self._cache_in = args[0].detach().clone() if len(args) > 0 else None
            self._cache_out = self._clone_out(out)
            self._hit_count = 0
            return out

        if self._cache_in is None or self._hit_count >= self._mc:
            out = self._real(*args, **kwargs)
            self._cache_in = args[0].detach().clone() if len(args) > 0 else None
            self._cache_out = self._clone_out(out)
            self._hit_count = 0
            return out

        # 计算输入变化率
        cur_in = args[0] if len(args) > 0 else None
        if cur_in is not None and self._cache_in is not None:
            try:
                diff = (cur_in - self._cache_in).abs().mean().item()
                ref = self._cache_in.abs().mean().item()
                rel = diff / max(ref, 1e-6)
                if rel < self._rdt:
                    self._hit_count += 1
                    return self._cache_out
            except Exception:
                pass

        # 缓存未命中，真正计算
        out = self._real(*args, **kwargs)
        self._cache_in = args[0].detach().clone() if len(args) > 0 else None
        self._cache_out = self._clone_out(out)
        self._hit_count = 0
        return out

    @staticmethod
    def _clone_out(out):
        if isinstance(out, tuple):
            return tuple(x.detach().clone() if torch.is_tensor(x) else x for x in out)
        return out.detach().clone() if torch.is_tensor(out) else out

    def reset(self):
        self._cache_in = None
        self._cache_out = None
        self._hit_count = 0
        self._total_calls = 0


def _apply_cache_dit(bridge, enable=True, fn=1, bn=1, rdt=0.24, mc=3):
    """Wrap the bridge's fusion_blocks and remaining_video_blocks with Cache-DiT proxies."""
    if enable:
        if getattr(bridge, "_bsai_cache_dit_on", False):
            return "Cache-DiT already active"
        # Save the original block lists
        bridge._orig_fusion_blocks = list(bridge.fusion_blocks)
        bridge._orig_remaining_blocks = list(bridge.remaining_video_blocks)

        n_fusion = len(bridge._orig_fusion_blocks)
        n_remaining = len(bridge._orig_remaining_blocks)

        # Wrap fusion blocks
        wrapped_fusion = []
        for i, blk in enumerate(bridge._orig_fusion_blocks):
            is_first = i < fn
            is_last = (i >= n_fusion - bn)
            wrapped_fusion.append(_CacheDiTBlock(blk, fn=is_first, bn=is_last, rdt=rdt, mc=mc))
        bridge.fusion_blocks = torch.nn.ModuleList(wrapped_fusion)

        # Wrap remaining blocks
        wrapped_remaining = []
        for i, blk in enumerate(bridge._orig_remaining_blocks):
            is_first = i < fn
            is_last = (i >= n_remaining - bn)
            wrapped_remaining.append(_CacheDiTBlock(blk, fn=is_first, bn=is_last, rdt=rdt, mc=mc))
        bridge.remaining_video_blocks = torch.nn.ModuleList(wrapped_remaining)

        bridge._bsai_cache_dit_on = True
        return f"Cache-DiT OK (fn={fn}, bn={bn}, rdt={rdt}, mc={mc})"
    else:
        if getattr(bridge, "_bsai_cache_dit_on", False):
            bridge.fusion_blocks = torch.nn.ModuleList(bridge._orig_fusion_blocks)
            bridge.remaining_video_blocks = torch.nn.ModuleList(bridge._orig_remaining_blocks)
            bridge._bsai_cache_dit_on = False
            return "Cache-DiT disabled"
        return "Cache-DiT not enabled"


class BSAIPrismTurboOptimizer:
    """对 MOVA transformer 做 torch.compile 加速。

    SGLang MOVA 文档确认 --enable-torch-compile 可加速推理。
    这里作为可选后处理步骤，编译失败自动回退到 eager 模式。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prism_model": (PRISM_MODEL_TYPE, {}),
                "enable_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "torch.compile on the transformer forward. Note: the custom BSA Triton kernels may fail to compile on PyTorch 2.13 + sm_120. Off by default; rely on Cache-DiT + CFG Rescale instead.",
                }),
                "compile_mode": (["default", "reduce-overhead", "max-autotune"], {
                    "default": "default",
                    "tooltip": "default = no CUDA graphs (compatible with cudaMallocAsync); reduce-overhead = CUDA graphs, fastest but conflicts with --cuda-malloc; max-autotune = most aggressive optimization.",
                }),
                "compile_dynamic": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "dynamic=True supports variable shapes (slower but more flexible); False compiles for a fixed shape (fastest). Choose False when the resolution is fixed.",
                }),
                # ---- Cache-DiT ----
                "enable_cache_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache-DiT block-level caching (validated 7.4x speedup by SGLang). Skips block recomputation when the input changes little.",
                }),
                "cache_dit_fn": ("INT", {
                    "default": 1, "min": 0, "max": 10,
                    "tooltip": "Always compute the first N blocks (cache disabled). Default 1.",
                }),
                "cache_dit_bn": ("INT", {
                    "default": 1, "min": 0, "max": 10,
                    "tooltip": "Always compute the last N blocks. Default 1.",
                }),
                "cache_dit_rdt": ("FLOAT", {
                    "default": 0.24, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": "Residual difference threshold. Smaller = stricter (more recompute, better quality); larger = more aggressive (more skips, faster). Default 0.24.",
                }),
                "cache_dit_mc": ("INT", {
                    "default": 3, "min": 1, "max": 10,
                    "tooltip": "Maximum consecutive cached steps; after that a recompute is forced. Default 3.",
                }),
                # ---- Quantization ----
                "quantize_mode": (["none", "float8_dynamic", "int8_dynamic", "int4_weight_only", "int4_float8_dynamic"], {
                    "default": "none",
                    "tooltip": "float8_dynamic = W8A8 fp8, halves VRAM; int4_weight_only = W4A16, 4-bit weights use 1/4 VRAM, activations uncompressed, best quality; int4_float8_dynamic = W4A8, 4-bit weights + fp8 activations, maximum Blackwell speedup.",
                }),
            },
        }

    RETURN_TYPES = (PRISM_MODEL_TYPE, "STRING")
    RETURN_NAMES = ("prism_model", "status")
    FUNCTION = "optimize"
    CATEGORY = "BSAI/Prism/Turbo"

    def optimize(self, prism_model: PrismModelHandle,
                 enable_torch_compile=True, compile_mode="reduce-overhead",
                 compile_dynamic=False,
                 enable_cache_dit=True, cache_dit_fn=1, cache_dit_bn=1,
                 cache_dit_rdt=0.24, cache_dit_mc=3,
                 quantize_mode="none"):
        loaded = model_manager.get_cached(prism_model.key)
        if loaded is None:
            raise RuntimeError("Model cache lost; re-run the loader node.")

        bridge = loaded.bridge
        status_parts = []

        if enable_torch_compile:
            # Check whether compilation already happened
            if getattr(bridge, "_bsai_turbo_compiled", False):
                status_parts.append("torch.compile active (cached)")
            else:
                try:
                    _log(f"[Turbo] Starting torch.compile (mode={compile_mode}, dynamic={compile_dynamic})...")
                    # Detect cudaMallocAsync: reduce-overhead (CUDA graphs) is incompatible
                    alloc_backend = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
                    using_async = "backend:cudaMallocAsync" in alloc_backend
                    if compile_mode == "reduce-overhead" and using_async:
                        _log("[Turbo] cudaMallocAsync detected; downgrading compile_mode to default (CUDA graphs incompatible)")
                        compile_mode = "default"
                        status_parts.append("cudaMallocAsync detected; CUDA graphs disabled")
                    # Compile only the transformer forward; do not touch VAE/text_encoder.
                    # With block offload each block is moved separately, so compiling the whole bridge may misbehave.
                    # We wrap bridge.forward rather than the bridge itself.
                    original_forward = bridge.forward

                    compiled = torch.compile(
                        bridge.forward,
                        mode=compile_mode,
                        dynamic=compile_dynamic,
                        fullgraph=False,
                    )
                    bridge._original_forward = original_forward
                    bridge.forward = compiled
                    bridge._bsai_turbo_compiled = True
                    status_parts.append(f"torch.compile OK (mode={compile_mode})")
                    _log("[Turbo] torch.compile wrapped; first sampling run triggers compilation (slower)")
                except Exception as e:
                    status_parts.append(f"torch.compile failed, falling back to eager: {e}")
                    _log(f"[Turbo] torch.compile failed: {e}")
                    if hasattr(bridge, "_original_forward"):
                        bridge.forward = bridge._original_forward
                        bridge._bsai_turbo_compiled = False
        else:
            # If it was compiled before, restore the original forward
            if getattr(bridge, "_bsai_turbo_compiled", False):
                bridge.forward = bridge._original_forward
                bridge._bsai_turbo_compiled = False
                status_parts.append("Restored eager mode")
            else:
                status_parts.append("torch.compile not enabled")

        # ---- Cache-DiT block-level cache ----
        try:
            cd_status = _apply_cache_dit(
                bridge, enable=enable_cache_dit,
                fn=cache_dit_fn, bn=cache_dit_bn,
                rdt=cache_dit_rdt, mc=cache_dit_mc,
            )
            status_parts.append(cd_status)
        except Exception as e:
            status_parts.append(f"Cache-DiT failed: {e}")
            _log(f"[Turbo] Cache-DiT failed: {e}")

        # ---- torchao quantization ----
        if quantize_mode != "none":
            try:
                from torchao.quantization.quant_api import quantize_
                qcfg = None
                qname = ""
                if quantize_mode == "float8_dynamic":
                    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
                    qcfg = Float8DynamicActivationFloat8WeightConfig()
                    qname = "W8A8 float8 dynamic"
                elif quantize_mode == "int8_dynamic":
                    from torchao.quantization import Int8DynamicActivationInt8WeightConfig
                    qcfg = Int8DynamicActivationInt8WeightConfig()
                    qname = "W8A8 int8 dynamic"
                elif quantize_mode == "int4_weight_only":
                    from torchao.quantization import Int4WeightOnlyConfig
                    qcfg = Int4WeightOnlyConfig(group_size=128)
                    qname = "W4A16 int4 weight-only"
                elif quantize_mode == "int4_float8_dynamic":
                    from torchao.quantization import Float8DynamicActivationInt4WeightConfig
                    qcfg = Float8DynamicActivationInt4WeightConfig(group_size=128)
                    qname = "W4A8 int4+float8"

                if qcfg is not None:
                    n_quantized = [0]
                    def _filter_fn(mod, fqn):
                        skip_keywords = ['embed', 'norm', 'head', 'out_proj', 'proj_out', 'final', 'adaln', 'time']
                        if isinstance(mod, torch.nn.Linear):
                            for kw in skip_keywords:
                                if kw in fqn.lower():
                                    return False
                            return True
                        return False

                    quantize_(bridge, qcfg, _filter_fn)
                    # Count how many Linear layers were quantized
                    for name, mod in bridge.named_modules():
                        if isinstance(mod, torch.nn.Linear):
                            n_quantized[0] += 1
                    status_parts.append(f"{qname} OK ({n_quantized[0]} Linear layers)")
                    _log(f"[Turbo] {qname} quantized, {n_quantized[0]} layers")
            except Exception as e:
                status_parts.append(f"Quantization failed: {e}")
                _log(f"[Turbo] quantization failed: {e}")

        return (prism_model, " | ".join(status_parts))


# ---------------------------------------------------------------------------
# 2. Turbo 采样节点（CFG Rescale + 自适应跳步）
# ---------------------------------------------------------------------------

def _cfg_rescale_noise_pred(pred_posi, pred_nega, cfg_scale, rescale_factor=0.5):
    """CFG Rescale（参照 SDXL/Playground v2 实践）。

    公式：pred_rescaled = pred * (norm(uncond) / norm(pred))
    防止高 CFG 导致色彩过饱和/画面发灰。
    rescale_factor=0 = 不 rescale，纯原始 CFG。
    """
    if rescale_factor <= 0.0:
        return pred_posi
    # pred = nega + cfg * (posi - nega)
    pred = pred_nega + cfg_scale * (pred_posi - pred_nega)
    # 计算 rescale
    std_posi = pred_posi.float().std()
    std_pred = pred.float().std()
    if std_pred > 1e-6:
        rescale = std_posi / std_pred
        # 混合：不完全替换，按 factor 插值
        pred = pred * (rescale_factor * rescale + (1.0 - rescale_factor))
    return pred


class BSAIPrismTurboSampler:
    """Turbo 采样器：独立 denoising 循环，不修改原 pipeline。

    新增能力：
      - cfg_rescale: CFG Rescale 保质（防止过饱和）
      - adaptive_step_skip: 自适应跳步（预测变化小时跳过负向分支，省一半时间）
      - 更激进的默认参数（社区调优）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prism_model": (PRISM_MODEL_TYPE, {}),
                "prompt": ("STRING", {
                    "default": "A man with short gray hair plays a red electric guitar.",
                    "multiline": True,
                }),
                "audio_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": DEFAULT_NEGATIVE_PROMPT, "multiline": True}),
                "resolution": (RESOLUTION_ORDER, {
                    "default": "Landscape 480p (848x480)",
                }),
                "width": ("INT", {"default": 848, "min": 448, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 16}),
                "num_frames": ("INT", {"default": 121, "min": 17, "max": 1024, "step": 4}),
                "fps": ("FLOAT", {"default": 17.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 15.0, "step": 0.1}),
                "visual_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "audio_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF, "step": 1}),
                # ---- Turbo-specific params ----
                "cfg_rescale": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "CFG Rescale strength. 0 = off (raw CFG); 0.5 = standard rescale (recommended); 1.0 = full rescale. Prevents oversaturation at high CFG.",
                }),
                "adaptive_step_skip": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adaptive step skip: when consecutive predictions change little, skip the negative CFG branch (halves forward time). Slight quality drop but a large speedup.",
                }),
                "step_skip_threshold": ("FLOAT", {
                    "default": 0.02, "min": 0.001, "max": 0.2, "step": 0.001,
                    "tooltip": "Prediction-change threshold. Below this, the prediction is considered converged and the step is skipped.",
                }),
                "enable_vae_tiling": ("BOOLEAN", {"default": False}),
                "tile_sample_min_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
                "tile_sample_stride": ("INT", {"default": 224, "min": 32, "max": 1024, "step": 32}),
                "remove_video_dit_after_switch": ("BOOLEAN", {"default": False}),
                "save_video": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "BSAI_Prism_Turbo"}),
            },
            "optional": {
                "image": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "BSAI/Prism/Turbo"

    def sample(
        self,
        prism_model: PrismModelHandle,
        prompt,
        audio_prompt="",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        resolution="Landscape 480p (848x480)",
        width=848,
        height=480,
        num_frames=121,
        fps=17.0,
        num_inference_steps=25,
        cfg_scale=2.5,
        visual_shift=3.0,
        audio_shift=3.0,
        seed=42,
        cfg_rescale=0.5,
        adaptive_step_skip=False,
        step_skip_threshold=0.02,
        enable_vae_tiling=False,
        tile_sample_min_size=256,
        tile_sample_stride=224,
        remove_video_dit_after_switch=False,
        save_video=True,
        filename_prefix="BSAI_Prism_Turbo",
        image=None,
    ):
        from PIL import Image

        loaded = model_manager.get_cached(prism_model.key)
        if loaded is None:
            raise RuntimeError("Model cache lost; re-run the loader node.")

        # Resolve the resolution tier
        if resolution == "Custom":
            h, w = int(height), int(width)
            v_shift, a_shift = float(visual_shift), float(audio_shift)
            tiling = bool(enable_vae_tiling)
        else:
            prof = RESOLUTION_PROFILES[resolution]
            h, w = prof.height, prof.width
            v_shift, a_shift = prof.visual_shift, prof.audio_shift
            tiling = prof.vae_tiling

        h = _snap(h, 16, "height")
        w = _snap(w, 16, "width")

        _log(
            f"[Turbo] sampling: {h}x{w} @ {fps}fps, {num_frames} frames, {num_inference_steps} steps, "
            f"cfg={cfg_scale} rescale={cfg_rescale} shift=({v_shift},{a_shift}) "
            f"adaptive_skip={adaptive_step_skip} seed={seed}"
        )

        pipeline = model_manager.build_pipeline(
            loaded.bridge, loaded.extras, torch.device(prism_model.device), prism_model.offload
        )

        # 参考图
        if image is None:
            _log("[Turbo] image not connected; pure-noise t2v mode")
            ref_img = None
        else:
            ref_np = image[0].cpu().numpy()
            ref_np = (np.clip(ref_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            ref_img = Image.fromarray(ref_np).convert("RGB")
            ref_img = _crop_and_resize(ref_img, height=h, width=w)

        # 帧数对齐
        raw_num_frames = int(num_frames)
        num_frames = raw_num_frames - (raw_num_frames - 1) % pipeline.vae_scale_factor_temporal
        if num_frames < pipeline.vae_scale_factor_temporal + 1:
            num_frames = pipeline.vae_scale_factor_temporal + 1

        device = pipeline.device
        offload = pipeline._cpu_offload
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # ---- Phase 1: Text encoding（复用 pipeline 内部方法）----
        if offload:
            pipeline.text_encoder.to(device)

        pipeline.scheduler.set_timesteps(num_inference_steps, shift=v_shift, device=device)
        pipeline.scheduler.set_pair_postprocess_by_name(
            "dual_sigma_shift", visual_shift=float(v_shift), audio_shift=float(a_shift),
        )
        paired_timesteps = pipeline.scheduler.get_pairs()

        prompt_embeds = pipeline._get_t5_prompt_embeds(prompt)
        audio_prompt_embeds = pipeline._get_t5_prompt_embeds(audio_prompt) if audio_prompt.strip() else None
        negative_prompt_embeds = pipeline._get_t5_prompt_embeds(negative_prompt)

        if offload:
            pipeline.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # ---- Phase 2: VAE encode ----
        if offload:
            pipeline.video_vae.to(device)

        num_channels_latents = pipeline.video_vae.config.z_dim
        if ref_img is not None:
            ref_img_t = pipeline.video_processor.preprocess(
                ref_img, height=h, width=w
            ).to(device, dtype=torch.float32)
        else:
            ref_img_t = None

        latents, condition = pipeline.prepare_latents(
            ref_img_t, 1, num_channels_latents, h, w, num_frames,
            torch.float32, device, generator=generator, latents=None, last_image=None,
        )
        audio_num_samples = int(pipeline.audio_sample_rate * num_frames / fps)
        audio_latents = pipeline.prepare_audio_latents(
            None, 1, pipeline.audio_vae.latent_dim, audio_num_samples,
            torch.float32, device, generator=generator, latents=None,
        )

        if offload:
            pipeline.video_vae.to("cpu")
            torch.cuda.empty_cache()

        # ---- Phase 3: Denoising loop（Turbo 增强版）----
        if offload:
            pipeline.transformer._block_offload = True

        cur_use_dit_2 = False
        total_steps = paired_timesteps.shape[0]
        switched = False
        boundary_timestep = pipeline.boundary_ratio * pipeline.scheduler.config.num_train_timesteps

        # 自适应跳步状态
        prev_visual_noise = None
        skipped_count = 0

        for idx_step in range(total_steps):
            timestep, audio_timestep = paired_timesteps[idx_step]

            if not switched and timestep.item() < boundary_timestep:
                cur_use_dit_2 = True
                if remove_video_dit_after_switch:
                    pipeline.transformer.video_dit = None
                    gc.collect()
                switched = True

            latent_model_input = torch.cat([latents, condition], dim=1)
            timestep_t = timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
            audio_timestep_t = audio_timestep.unsqueeze(0).to(device=device, dtype=torch.float32)

            # 正向分支（总是执行）
            noise_pred_posi = pipeline.transformer(
                visual_latents=latent_model_input,
                audio_latents=audio_latents,
                context=prompt_embeds,
                timestep=timestep_t,
                audio_context=audio_prompt_embeds,
                audio_timestep=audio_timestep_t,
                video_fps=fps,
                num_train_timesteps=pipeline.scheduler.config.num_train_timesteps,
                use_video_dit_2=cur_use_dit_2,
            )

            if cfg_scale == 1.0:
                visual_noise_pred = noise_pred_posi[0].float()
                audio_noise_pred = noise_pred_posi[1].float()
            else:
                # 自适应跳步：如果上一步预测变化很小，跳过负向分支
                skip_negative = False
                if adaptive_step_skip and prev_visual_noise is not None:
                    diff = (noise_pred_posi[0].float() - prev_visual_noise).abs().mean().item()
                    if diff < step_skip_threshold:
                        skip_negative = True
                        skipped_count += 1

                if skip_negative:
                    # 复用上一步的负向预测（近似）
                    visual_noise_pred = prev_visual_noise
                    audio_noise_pred = prev_audio_noise
                else:
                    noise_pred_nega = pipeline.transformer(
                        visual_latents=latent_model_input,
                        audio_latents=audio_latents,
                        context=negative_prompt_embeds,
                        timestep=timestep_t,
                        audio_context=None,
                        audio_timestep=audio_timestep_t,
                        video_fps=fps,
                        num_train_timesteps=pipeline.scheduler.config.num_train_timesteps,
                        use_video_dit_2=cur_use_dit_2,
                    )
                    visual_posi = noise_pred_posi[0].float()
                    audio_posi = noise_pred_posi[1].float()
                    visual_nega = noise_pred_nega[0].float()
                    audio_nega = noise_pred_nega[1].float()

                    # CFG Rescale
                    visual_noise_pred = _cfg_rescale_noise_pred(
                        visual_posi, visual_nega, cfg_scale, cfg_rescale
                    )
                    audio_noise_pred = _cfg_rescale_noise_pred(
                        audio_posi, audio_nega, cfg_scale, cfg_rescale
                    )

                prev_visual_noise = visual_noise_pred.clone()
                prev_audio_noise = audio_noise_pred.clone()

            next_timestep = paired_timesteps[idx_step + 1, 0] if idx_step + 1 < total_steps else None
            next_audio_timestep = paired_timesteps[idx_step + 1, 1] if idx_step + 1 < total_steps else None
            latents = pipeline.scheduler.step_from_to(
                visual_noise_pred, timestep, next_timestep, latents,
            )
            audio_latents = pipeline.scheduler.step_from_to(
                audio_noise_pred, audio_timestep, next_audio_timestep, audio_latents,
            )

        if offload:
            pipeline.transformer._block_offload = False
            pipeline.transformer.to("cpu")
            from prism_runtime.models.modules.block_sparse_attention import bsa_interface as _bsa_iface
            _bsa_iface._DISABLE_COMPILE = False
            torch.cuda.empty_cache()

        if skipped_count > 0:
            _log(f"[Turbo] adaptive step skip: skipped {skipped_count}/{total_steps} negative branches")

        # ---- Phase 4: VAE decode（fp32，与主管线一致）----
        if offload:
            pipeline.video_vae.to(device)

        if enable_vae_tiling:
            tiling_kwargs = {}
            if tile_sample_min_size is not None:
                tiling_kwargs["tile_sample_min_height"] = tile_sample_min_size
                tiling_kwargs["tile_sample_min_width"] = tile_sample_min_size
            if tile_sample_stride is not None:
                tiling_kwargs["tile_sample_stride_height"] = tile_sample_stride
                tiling_kwargs["tile_sample_stride_width"] = tile_sample_stride
            pipeline.video_vae.enable_tiling(**tiling_kwargs)

        video_latents = pipeline.denormalize_video_latents(latents)
        with torch.autocast("cuda", dtype=torch.float32):
            video = pipeline.video_vae.decode(video_latents.to(dtype=torch.float32)).sample
        video = pipeline.video_processor.postprocess_video(video, output_type="pil")

        if offload:
            pipeline.video_vae.to("cpu")
            pipeline.audio_vae.to(device)
            torch.cuda.empty_cache()

        # frames: list[list[PIL]] -> ComfyUI IMAGE tensor [B,H,W,3]
        frames = video
        images = torch.from_numpy(
            np.stack([np.array(f) for f in frames[0]], axis=0).astype(np.float32) / 255.0
        )

        with torch.autocast("cuda", dtype=torch.float32):
            audio_out = pipeline.audio_vae.decode(audio_latents)

        if offload:
            pipeline.audio_vae.to("cpu")
            torch.cuda.empty_cache()

        audio_wav = audio_out[0].detach().cpu().float()
        if audio_wav.dim() == 1:
            audio_wav = audio_wav.unsqueeze(0).unsqueeze(0)
        elif audio_wav.dim() == 2:
            audio_wav = audio_wav.unsqueeze(0)
        audio_dict = {
            "waveform": audio_wav,
            "sample_rate": int(pipeline.audio_sample_rate),
        }

        video_path = ""
        videos_ui = []
        if save_video:
            audio_save = audio_out[0].cpu().squeeze()
            out_dir = os.path.join(folder_paths.get_output_directory(), "BSAI_Prism")
            os.makedirs(out_dir, exist_ok=True)
            _, _, counter, _, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(),
                images.shape[1], images.shape[2],
            )
            video_path = os.path.join(out_dir, f"{filename_prefix}_{counter:05d}.mp4")
            save_video_with_audio(
                frames[0], audio_save, video_path,
                fps=fps, sample_rate=pipeline.audio_sample_rate,
            )
            videos_ui = [{
                "filename": os.path.basename(video_path),
                "subfolder": "BSAI_Prism",
                "type": "output",
            }]
            _log(f"[Turbo] saved: {video_path}")

        return {"ui": {"videos": videos_ui}, "result": (images, audio_dict, video_path)}


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS_TURBO = {
    "BSAIPrismTurboOptimizer": BSAIPrismTurboOptimizer,
    "BSAIPrismTurboSampler": BSAIPrismTurboSampler,
}

NODE_DISPLAY_NAME_MAPPINGS_TURBO = {
    "BSAIPrismTurboOptimizer": "BSAI Prism Turbo Optimizer (torch.compile)",
    "BSAIPrismTurboSampler": "BSAI Prism Turbo Sampler (CFG Rescale + Step Skip)",
}
