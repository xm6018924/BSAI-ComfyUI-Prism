# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— 分辨率档位与 BSA 稀疏注意力参数配置

参数口径全部来自 Prism 官方资料：
  - README / prism_infer.sh（推理脚本）中的 ENABLE_BSA / BSA_SPARSITY /
    BSA_CHUNK_3D_SHAPE_Q / BSA_CHUNK_3D_SHAPE_K / BSA_CDF_THRESHOLD /
    ENABLE_IVPQ_DYNAMIC_BLOCK / VISUAL_SHIFT / AUDIO_SHIFT / CFG_SCALE 等；
  - README 中 720p/1080p/2K 的 VISUAL_SHIFT 建议区间（720p=7.0，
    1080p=9.0~13.0，2K=13.0~17.0）。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionProfile:
    """一个分辨率档位对应的推理默认参数。"""

    name: str
    height: int
    width: int
    visual_shift: float
    audio_shift: float
    num_frames: int
    vae_tiling: bool
    note: str


RESOLUTION_PROFILES = {
    "480p (480x848)": ResolutionProfile(
        name="480p",
        height=480,
        width=848,
        visual_shift=7.0,
        audio_shift=7.0,
        num_frames=193,
        vae_tiling=False,
        note="官方最低档位；小显存也可跑，建议开启 CPU offload。",
    ),
    "720p (720x1280)": ResolutionProfile(
        name="720p",
        height=720,
        width=1280,
        visual_shift=7.0,
        audio_shift=7.0,
        num_frames=205,
        vae_tiling=False,
        note="官方推荐单 80GB 卡 + CPU offload 的档位。",
    ),
    "1080p (1072x1920)": ResolutionProfile(
        name="1080p",
        height=1072,
        width=1920,
        visual_shift=11.0,
        audio_shift=9.0,
        num_frames=205,
        vae_tiling=True,
        note="官方推荐 4x80GB FSDP；单卡约需 80GB 显存，VAE 解码自动开 tiling。",
    ),
    "2K (1440x2560)": ResolutionProfile(
        name="2k",
        height=1440,
        width=2560,
        visual_shift=15.0,
        audio_shift=11.0,
        num_frames=205,
        vae_tiling=True,
        note="官方推荐 4+ x80GB FSDP；单卡基本不可行，谨慎使用。",
    ),
    "custom": ResolutionProfile(
        name="custom",
        height=480,
        width=848,
        visual_shift=7.0,
        audio_shift=7.0,
        num_frames=193,
        vae_tiling=False,
        note="自定义：高度/宽度必须能被 16 整除，帧数需满足 (n-1) % 4 == 0（自动对齐）。",
    ),
}

# 分辨率预设的显示顺序（下拉框展示顺序）
RESOLUTION_ORDER = [
    "480p (480x848)",
    "720p (720x1280)",
    "1080p (1072x1920)",
    "2K (1440x2560)",
    "custom",
]


@dataclass
class BSAParams:
    """Prism Block Sparse Attention 参数（对应官方推理脚本开关）。"""

    enable_bsa: bool = True
    bsa_sparsity: float = 0.75          # top-k 稀疏率：0.93/0.85/0.75
    bsa_chunk_3d_shape_q: str = "4 4 4"  # 视频自注意力 Q 的 3D 宏区 (T H W)
    bsa_chunk_3d_shape_k: str = "4 4 4"  # 视频自注意力 K 的 3D 宏区 (T H W)
    bsa_cdf_threshold: float = 0.20     # top-p CDF 阈值，混合 top-k+top-p

    enable_bsa_v2a: bool = False        # bridge v2a 交叉注意力 BSA
    bsa_v2a_sparsity: float = 0.875
    bsa_v2a_audio_chunk_size: int = 64  # audio Q 块大小（64 的倍数）
    bsa_v2a_chunk_3d_shape_k: str = "4 4 4"

    enable_ivpq_dynamic_block: bool = True      # Prism 核心：各向异性动态块形状
    enable_layer_adaptive_dynamic_block: bool = False
    sparse_high_noise_only: bool = False        # 仅高噪声专家（video_dit）用稀疏

    # 音频引导（与动态块互斥，默认关闭）
    enable_audio_guidance: bool = False
    enable_audio_concentration_gate: bool = False
    enable_timestep_reliability_gate: bool = False
    audio_boost_gamma: float = 2.0
    enable_audio_weighted_pooling: bool = False
    audio_weighted_lambda: float = 2.0

    # 通道方差引导（独立，默认关闭）
    enable_variance_guidance: bool = False
    variance_boost_gamma: float = 2.0

    # 偏差校正（二者互斥，默认关闭）
    enable_taylor_sparse_attn: bool = False
    taylor_alpha_f: float = 0.5
    enable_rectified_sparse_attn: bool = False

    def chunk_q(self):
        return tuple(int(v) for v in self.bsa_chunk_3d_shape_q.split())

    def chunk_k(self):
        return tuple(int(v) for v in self.bsa_chunk_3d_shape_k.split())

    def chunk_k_v2a(self):
        return tuple(int(v) for v in self.bsa_v2a_chunk_3d_shape_k.split())

    def validate(self):
        """互斥与取值范围校验，任何冲突直接抛错（不静默降级）。"""
        if not (0.0 < self.bsa_sparsity <= 1.0):
            raise ValueError(f"bsa_sparsity 必须在 (0, 1]：{self.bsa_sparsity}")
        if self.bsa_cdf_threshold is not None and not (0.0 < self.bsa_cdf_threshold <= 1.0):
            raise ValueError(f"bsa_cdf_threshold 必须在 (0, 1]：{self.bsa_cdf_threshold}")
        if self.enable_taylor_sparse_attn and self.enable_rectified_sparse_attn:
            raise ValueError(
                "enable_taylor_sparse_attn 与 enable_rectified_sparse_attn 互斥，"
                "最多只能开启一个。"
            )
        if self.enable_ivpq_dynamic_block and self.enable_penalty_dynamic_block():
            raise ValueError("enable_ivpq_dynamic_block 与 enable_penalty_dynamic_block 互斥。")
        if self.dynamic_block_enabled() and (
            self.enable_audio_guidance
            or self.enable_audio_weighted_pooling
            or self.enable_variance_guidance
            or self.enable_taylor_sparse_attn
            or self.enable_rectified_sparse_attn
        ):
            raise ValueError(
                "动态块形状（IVPQ）与音频引导/方差引导/Taylor/Rectified 偏差校正互斥，"
                "请关闭其中一组。"
            )
        for label, s in (
            ("bsa_chunk_3d_shape_q", self.bsa_chunk_3d_shape_q),
            ("bsa_chunk_3d_shape_k", self.bsa_chunk_3d_shape_k),
            ("bsa_v2a_chunk_3d_shape_k", self.bsa_v2a_chunk_3d_shape_k),
        ):
            try:
                v = tuple(int(x) for x in s.split())
            except Exception as exc:
                raise ValueError(f"{label} 必须是 3 个整数（空格分隔），例如 \"4 4 4\"：{s}") from exc
            if len(v) != 3:
                raise ValueError(f"{label} 必须是 3 个整数（空格分隔），例如 \"4 4 4\"：{s}")

    def enable_penalty_dynamic_block(self):
        # 插件 v1 不暴露 Penalty（与 IVPQ 互斥的第二条路径），保留校验位
        return False

    def dynamic_block_enabled(self):
        return self.enable_ivpq_dynamic_block or self.enable_penalty_dynamic_block()


# 官方默认负向提示词（sample_mova_single.py 内置）
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)
