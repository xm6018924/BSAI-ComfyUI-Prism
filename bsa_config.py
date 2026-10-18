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
    # --- Landscape ---
    "Landscape 480p (848x480)": ResolutionProfile(
        name="480p",
        height=480,
        width=848,
        visual_shift=7.0,
        audio_shift=7.0,
        num_frames=193,
        vae_tiling=False,
        note="Official minimum tier; runs on small VRAM, CPU offload recommended.",
    ),
    "Landscape 720p (1280x720)": ResolutionProfile(
        name="720p",
        height=720,
        width=1280,
        visual_shift=7.0,
        audio_shift=7.0,
        num_frames=205,
        vae_tiling=False,
        note="Official recommended tier: single 80GB GPU + CPU offload.",
    ),
    "Landscape 1080p (1920x1072)": ResolutionProfile(
        name="1080p",
        height=1072,
        width=1920,
        visual_shift=11.0,
        audio_shift=9.0,
        num_frames=205,
        vae_tiling=True,
        note="Official recommended 4x80GB FSDP; single GPU needs ~80GB, VAE tiling auto-enabled.",
    ),
    "Landscape 2K (2560x1440)": ResolutionProfile(
        name="2k",
        height=1440,
        width=2560,
        visual_shift=15.0,
        audio_shift=11.0,
        num_frames=205,
        vae_tiling=True,
        note="Official recommended 4+ x80GB FSDP; not practical on a single GPU.",
    ),
    # --- Portrait ---
    "Portrait 480p (480x848)": ResolutionProfile(
        name="480p",
        height=848,
        width=480,
        visual_shift=7.0,
        audio_shift=7.0,
        num_frames=193,
        vae_tiling=False,
        note="Minimum portrait tier; for short videos / phone vertical screen.",
    ),
    "Portrait 720p (720x1280)": ResolutionProfile(
        name="720p",
        height=1280,
        width=720,
        visual_shift=7.0,
        audio_shift=7.0,
        num_frames=205,
        vae_tiling=False,
        note="Portrait 720p; for phone vertical playback.",
    ),
    "Portrait 1080p (1072x1920)": ResolutionProfile(
        name="1080p",
        height=1920,
        width=1072,
        visual_shift=11.0,
        audio_shift=9.0,
        num_frames=205,
        vae_tiling=True,
        note="Portrait 1080p; multi-GPU FSDP recommended.",
    ),
    "Portrait 2K (1440x2560)": ResolutionProfile(
        name="2k",
        height=2560,
        width=1440,
        visual_shift=15.0,
        audio_shift=11.0,
        num_frames=205,
        vae_tiling=True,
        note="Portrait 2K; requires 4+ x80GB FSDP.",
    ),
    # --- Custom ---
    "Custom": ResolutionProfile(
        name="custom",
        height=480,
        width=848,
        visual_shift=7.0,
        audio_shift=7.0,
        num_frames=193,
        vae_tiling=False,
        note="Custom: height/width must be divisible by 16, frames must satisfy (n-1) % 4 == 0 (auto-aligned).",
    ),
}

# Display order of resolution presets (dropdown order)
RESOLUTION_ORDER = [
    "Landscape 720p (1280x720)",
    "Landscape 480p (848x480)",
    "Landscape 1080p (1920x1072)",
    "Landscape 2K (2560x1440)",
    "Portrait 720p (720x1280)",
    "Portrait 480p (480x848)",
    "Portrait 1080p (1072x1920)",
    "Portrait 2K (1440x2560)",
    "Custom",
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
            raise ValueError(f"bsa_sparsity must be in (0, 1]: {self.bsa_sparsity}")
        if self.bsa_cdf_threshold is not None and not (0.0 < self.bsa_cdf_threshold <= 1.0):
            raise ValueError(f"bsa_cdf_threshold must be in (0, 1]: {self.bsa_cdf_threshold}")
        if self.enable_taylor_sparse_attn and self.enable_rectified_sparse_attn:
            raise ValueError(
                "enable_taylor_sparse_attn and enable_rectified_sparse_attn are mutually "
                "exclusive; enable at most one."
            )
        if self.enable_ivpq_dynamic_block and self.enable_penalty_dynamic_block():
            raise ValueError("enable_ivpq_dynamic_block and enable_penalty_dynamic_block are mutually exclusive.")
        if self.dynamic_block_enabled() and (
            self.enable_audio_guidance
            or self.enable_audio_weighted_pooling
            or self.enable_variance_guidance
            or self.enable_taylor_sparse_attn
            or self.enable_rectified_sparse_attn
        ):
            raise ValueError(
                "Dynamic block shape (IVPQ) is mutually exclusive with audio guidance / "
                "variance guidance / Taylor / Rectified bias correction; disable one group."
            )
        for label, s in (
            ("bsa_chunk_3d_shape_q", self.bsa_chunk_3d_shape_q),
            ("bsa_chunk_3d_shape_k", self.bsa_chunk_3d_shape_k),
            ("bsa_v2a_chunk_3d_shape_k", self.bsa_v2a_chunk_3d_shape_k),
        ):
            try:
                v = tuple(int(x) for x in s.split())
            except Exception as exc:
                raise ValueError(f"{label} must be 3 integers (space-separated), e.g. \"4 4 4\": {s}") from exc
            if len(v) != 3:
                raise ValueError(f"{label} must be 3 integers (space-separated), e.g. \"4 4 4\": {s}")

    def enable_penalty_dynamic_block(self):
        # 插件 v1 不暴露 Penalty（与 IVPQ 互斥的第二条路径），保留校验位
        return False

    def dynamic_block_enabled(self):
        return self.enable_ivpq_dynamic_block or self.enable_penalty_dynamic_block()


# Official default negative prompt (built into sample_mova_single.py)
DEFAULT_NEGATIVE_PROMPT = (
    "oversaturated colors, overexposed, static, blurry details, subtitles, watermarks, "
    "stylized artifacts, artwork, still frame, washed-out gray, worst quality, low quality, "
    "JPEG compression artifacts, ugly, mutilated, extra fingers"
)
