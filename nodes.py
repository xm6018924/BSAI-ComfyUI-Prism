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

from .bsa_config import (
    BSAParams,
    DEFAULT_NEGATIVE_PROMPT,
    RESOLUTION_ORDER,
    RESOLUTION_PROFILES,
)
from . import model_manager
from .ffmpeg_helper import save_video_with_audio
from .download_weights import COMPONENT_PATTERNS, download_component

PRISM_MODEL_TYPE = "BSAI_PRISM_MODEL"


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
        default_base = os.path.join(
            folder_paths.models_dir, "diffusers", "BSAI-Prism",
            "pretrained_models", "MOVA-360p",
        )
        devices = ["cuda:0"] if torch.cuda.is_available() else []
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return {
            "required": {
                "base_model_path": ("STRING", {
                    "default": default_base,
                    "multiline": False,
                    "tooltip": "MOVA-360p 基底目录（含 model_index.json 及 video_dit/"
                               "audio_dit/dual_tower_bridge/video_vae/audio_vae/"
                               "text_encoder/tokenizer/scheduler 子目录）。",
                }),
                "resume_ckpt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Prism 微调权重路径（preview_alpha 或 preview_beta 的 "
                               "diffusion_pytorch_model.safetensors）。留空 = 只用基底。",
                }),
                "device": (devices or ["N/A"], {
                    "default": "cuda:0" if devices else "N/A",
                    "tooltip": "运行设备。Prism 依赖 flash-attn，仅支持 NVIDIA CUDA。",
                }),
                "offload": (["cpu", "none"], {
                    "default": "cpu",
                    "tooltip": "cpu = 编码器/VAE 按阶段搬入搬出 GPU，单卡省显存（官方推荐）；"
                               "none = 全量驻留 GPU，峰值更高。",
                }),
                "enable_bsa": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启 Prism Block Sparse Attention（视频自注意力 top-k/top-p 稀疏）。",
                }),
                "bsa_sparsity": ("FLOAT", {
                    "default": 0.75, "min": 0.1, "max": 0.95, "step": 0.01,
                    "tooltip": "top-k 稀疏率（官方推荐 0.93 / 0.85 / 0.75；越小越稀疏、越快）。",
                }),
                "bsa_chunk_3d_shape_q": ("STRING", {
                    "default": "4 4 4", "multiline": False,
                    "tooltip": "视频自注意力 Q 的 3D 宏区形状 (T H W)，官方最优 4 4 4。",
                }),
                "bsa_chunk_3d_shape_k": ("STRING", {
                    "default": "4 4 4", "multiline": False,
                    "tooltip": "视频自注意力 K 的 3D 宏区形状 (T H W)，官方最优 4 4 4。",
                }),
                "bsa_cdf_threshold": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "混合 Top-k+Top-p 的 CDF 阈值（官方 0.16~0.4，最优 0.2）。"
                               "0 = 纯 Top-k。",
                }),
                "enable_ivpq_dynamic_block": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prism 核心：按局部内容方差与音视频耦合度自适应分配各向异性块形状。"
                               "开启后与音频/方差引导、Taylor/Rectified 互斥（冲突会直接报错）。",
                }),
                "enable_layer_adaptive_dynamic_block": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "浅层固定块 / 深层动态块 + 浅->深音频缓存（仅动态块开启时生效）。",
                }),
                "sparse_high_noise_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "true = 稀疏注意力只作用于高噪声专家 video_dit，"
                               "低噪声专家 video_dit_2 保持全注意力。",
                }),
                "enable_bsa_v2a": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "bridge v2a 交叉注意力 BSA（Q=audio，K=video 3D 分块）。",
                }),
                "bsa_v2a_sparsity": ("FLOAT", {
                    "default": 0.875, "min": 0.1, "max": 0.95, "step": 0.01,
                }),
                "bsa_v2a_audio_chunk_size": ("INT", {
                    "default": 64, "min": 32, "max": 512, "step": 32,
                    "tooltip": "audio Q 块大小（64 的倍数）。",
                }),
                "bsa_v2a_chunk_3d_shape_k": ("STRING", {
                    "default": "4 4 4", "multiline": False,
                }),
            },
            "optional": {
                "enable_audio_guidance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "音频引导 BSA（Path A，与动态块互斥）。",
                }),
                "audio_boost_gamma": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 4.0, "step": 0.1,
                }),
                "enable_variance_guidance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "通道方差引导 BSA（与动态块互斥）。",
                }),
                "variance_boost_gamma": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 4.0, "step": 0.1,
                }),
                "enable_taylor_sparse_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Taylor 稀疏注意力偏差校正（与 Rectified 互斥，与动态块互斥）。",
                }),
                "enable_rectified_sparse_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Rectified 稀疏注意力偏差校正（与 Taylor 互斥，与动态块互斥）。",
                }),
            },
        }

    RETURN_TYPES = (PRISM_MODEL_TYPE, "STRING")
    RETURN_NAMES = ("prism_model", "status")
    FUNCTION = "load"
    CATEGORY = "BSAI/Prism"

    def load(
        self,
        base_model_path,
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
        # ---- 运行环境校验（不静默降级）----
        if not torch.cuda.is_available():
            raise RuntimeError(
                "BSAI Prism 需要 NVIDIA CUDA（flash-attn 依赖）。"
                "请在 python_embeded_cuda 环境（torch+cu130）中运行 ComfyUI。"
            )
        if device in ("N/A", "") or not device.startswith("cuda"):
            raise ValueError(f"设备无效：{device}")
        dev_idx = int(device.split(":")[-1])
        if dev_idx >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA 设备索引 {dev_idx} 超出范围（共 {torch.cuda.device_count()} 个）。"
            )

        base_model_path = os.path.abspath(base_model_path)
        if not os.path.isfile(os.path.join(base_model_path, "model_index.json")):
            raise FileNotFoundError(
                f"找不到 MOVA 基底：{base_model_path}（需要包含 model_index.json）。"
                "可用 BSAIPrismWeightsDownload 节点下载，或检查路径。"
            )
        if resume_ckpt and not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"找不到 resume_ckpt：{resume_ckpt}")

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
            base_model_path, resume_ckpt, device, offload, bsa_hash
        )

        cached = model_manager.get_cached(key)
        if cached is not None:
            _log(f"命中缓存模型（{base_model_path} / {resume_ckpt or 'base-only'}）")
            handle = PrismModelHandle(
                key=key, base_model_path=base_model_path, resume_ckpt=resume_ckpt,
                device=device, offload=offload, bsa=bsa,
            )
            return (handle, f"cache-hit | device={device} offload={offload}")

        bridge, extras = model_manager.load_prism_model(
            base_model_path, resume_ckpt, device, offload, bsa
        )
        model_manager.cache_put(
            key,
            model_manager._LoadedModel(
                bridge=bridge, extras=extras, key=key,
                device=torch.device(device), offload=offload,
            ),
        )
        handle = PrismModelHandle(
            key=key, base_model_path=base_model_path, resume_ckpt=resume_ckpt,
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
                "image": ("IMAGE", {
                    "tooltip": "参考图（i2v 首帧）。会按目标分辨率中心裁剪并缩放。",
                }),
                "prompt": ("STRING", {
                    "default": "A man with short gray hair plays a red electric guitar.",
                    "multiline": True,
                    "tooltip": "视频提示词，支持结构化标签 <music> <sfx> <speech> <text> <lyrics>。",
                }),
                "audio_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "音频提示词，留空则复用视频提示词。",
                }),
                "negative_prompt": ("STRING", {
                    "default": DEFAULT_NEGATIVE_PROMPT,
                    "multiline": True,
                }),
                "resolution": (RESOLUTION_ORDER, {
                    "default": "720p (720x1280)",
                    "tooltip": "分辨率档位；custom 时使用下方 height/width/shift/tiling 值。",
                }),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 16}),
                "width": ("INT", {"default": 848, "min": 448, "max": 4096, "step": 16}),
                "num_frames": ("INT", {
                    "default": 205, "min": 17, "max": 1024, "step": 4,
                    "tooltip": "生成帧数（像素空间）。自动对齐到 (n-1) % 4 == 0。",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "num_inference_steps": ("INT", {
                    "default": 50, "min": 1, "max": 200, "step": 1,
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 15.0, "step": 0.1,
                    "tooltip": "分类器自由引导强度；1.0 = 无引导。",
                }),
                "visual_shift": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "视频噪声调度 shift。720p≈7.0，1080p≈9~13，2K≈13~17。",
                }),
                "audio_shift": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF, "step": 1}),
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "VAE 解码空间分块（2K/1080p 防 OOM）。档位预设会自动覆盖。",
                }),
                "tile_sample_min_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
                "tile_sample_stride": ("INT", {"default": 192, "min": 32, "max": 1024, "step": 32}),
                "remove_video_dit_after_switch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "边界切换后释放高噪声专家 video_dit，节省显存（不可逆，仅内部释放）。",
                }),
                "save_video": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "BSAI_Prism"}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "BSAI/Prism"

    def _resolve_geometry(self, resolution, height, width, visual_shift, audio_shift,
                          enable_vae_tiling, num_frames):
        """档位预设覆盖几何与 shift；custom 用控件值。"""
        if resolution == "custom":
            return int(height), int(width), float(visual_shift), float(
                audio_shift), bool(enable_vae_tiling)
        prof = RESOLUTION_PROFILES[resolution]
        return prof.height, prof.width, prof.visual_shift, prof.audio_shift, prof.vae_tiling

    @staticmethod
    def _snap(value, base, label):
        snapped = int(value) - (int(value) % base)
        if snapped != int(value):
            _log(f"{label} {value} -> {snapped}（需为 {base} 的倍数）")
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
        image,
        prompt,
        audio_prompt="",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        resolution="720p (720x1280)",
        height=480,
        width=848,
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
        save_video=True,
        filename_prefix="BSAI_Prism",
    ):
        from PIL import Image

        loaded = model_manager.get_cached(prism_model.key)
        if loaded is None:
            raise RuntimeError(
                "模型缓存丢失（可能已被 BSAIPrismUnload 释放或进程重启），请重新执行加载节点。"
            )

        h, w, v_shift, a_shift, tiling = self._resolve_geometry(
            resolution, height, width, visual_shift, audio_shift, enable_vae_tiling, num_frames
        )
        # 高度/宽度对齐 16（Wan VAE 空间压缩 8 倍，check_inputs 要求 16 的倍数）
        h = self._snap(h, 16, "height")
        w = self._snap(w, 16, "width")

        _log(
            f"采样参数：{h}x{w} @ {fps}fps，{num_frames} 帧，{num_inference_steps} 步，"
            f"cfg={cfg_scale}，visual_shift={v_shift}，audio_shift={a_shift}，seed={seed}，"
            f"tiling={tiling}，bsa_sparsity={prism_model.bsa.bsa_sparsity}"
        )

        pipeline = model_manager.build_pipeline(
            loaded.bridge, loaded.extras, torch.device(prism_model.device), prism_model.offload
        )

        # 参考图：ComfyUI IMAGE [B,H,W,3] -> PIL
        ref_np = image[0].cpu().numpy()
        ref_np = (np.clip(ref_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        ref_img = Image.fromarray(ref_np).convert("RGB")
        ref_img = self._crop_and_resize(ref_img, height=h, width=w)

        # 帧数对齐 (n-1) % 4 == 0
        raw_num_frames = int(num_frames)
        num_frames = raw_num_frames - (raw_num_frames - 1) % pipeline.vae_scale_factor_temporal
        if num_frames != raw_num_frames:
            _log(f"num_frames {raw_num_frames} -> {num_frames}（需满足 (n-1) % 4 == 0）")

        # 需要对齐到 check_inputs 的另一个条件：(n-1) % temporal == 0 已满足；
        # 若用户给 < 5 帧则兜底
        if num_frames < pipeline.vae_scale_factor_temporal + 1:
            num_frames = pipeline.vae_scale_factor_temporal + 1

        torch.manual_seed(seed)

        frames, audio = pipeline(
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

        # ---- 转 ComfyUI 格式 ----
        images = torch.from_numpy(
            np.stack([np.array(f) for f in frames[0]], axis=0).astype(np.float32) / 255.0
        )
        audio_wav = audio[0].detach().cpu().float()  # [1, T]（DAC 单声道）
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
        if save_video:
            audio_save = audio[0].cpu().squeeze()
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
            _log(f"已保存：{video_path}")

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
        _log(f"已合成：{video_path}")
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
        default_target = os.path.join(folder_paths.models_dir, "diffusers", "BSAI-Prism")
        return {
            "required": {
                "target_dir": ("STRING", {
                    "default": default_target,
                    "tooltip": "权重落盘根目录：<target>/pretrained_models/MOVA-360p 与 "
                               "<target>/preview_alpha|beta/...",
                }),
                "components": (list(COMPONENT_PATTERNS.keys()), {
                    "default": ["base"],
                    "tooltip": "base=MOVA-360p 基底（约 40GB）；preview_alpha/beta 各 65.3GB。",
                }),
                "use_mirror": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用 HF_ENDPOINT=https://hf-mirror.com 镜像下载。",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("target_dir",)
    FUNCTION = "download"
    CATEGORY = "BSAI/Prism"

    def download(self, target_dir, components, use_mirror=False):
        results = []
        for comp in components:
            if comp not in COMPONENT_PATTERNS:
                raise ValueError(f"未知组件：{comp}")
            snapshot = download_component(comp, target_dir, use_mirror=use_mirror)
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
        return (count, f"已释放 {count} 个模型缓存")


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
    "BSAIPrismLoader": "BSAI Prism 模型加载",
    "BSAIPrismSampler": "BSAI Prism 采样（视频+音频）",
    "BSAIPrismVideoEncode": "BSAI Prism 音视频合成",
    "BSAIPrismWeightsDownload": "BSAI Prism 权重下载",
    "BSAIPrismUnload": "BSAI Prism 卸载模型",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
