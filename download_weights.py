# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism -- Prism weight downloader (FrancisRing/Prism, ~208GB total)

Repo layout (huggingface.co/FrancisRing/Prism):
    pretrained_models/MOVA-360p/        # base model (video/audio DiT, dual-tower
                                        #   bridge, VAE, text encoder, tokenizer, scheduler)
    preview_alpha/diffusion_pytorch_model.safetensors   # 65.3GB, stable
    preview_beta/diffusion_pytorch_model.safetensors    # 65.3GB, motion
    config.json

Components can be downloaded selectively (check them in the download node).
Files are always written into the fixed Prism root under ComfyUI models/:
    <models_dir>/diffusers/BSAI-Prism/
    ├── pretrained_models/MOVA-360p/...
    ├── preview_alpha/diffusion_pytorch_model.safetensors
    └── preview_beta/diffusion_pytorch_model.safetensors

Security: the download node no longer accepts an arbitrary workflow-supplied
target directory. The download destination is pinned to the fixed Prism root so
a workflow cannot redirect multi-GB downloads to an arbitrary filesystem path.
"""
import os

REPO_ID = "FrancisRing/Prism"

# Fixed subdirectory under ComfyUI models/ where Prism weights are always stored.
PRISM_MODELS_SUBDIR = os.path.join("diffusers", "BSAI-Prism")

COMPONENT_PATTERNS = {
    "base": {
        "label": "MOVA-360p base (~40GB)",
        "allow": [
            "pretrained_models/MOVA-360p/*",
        ],
        "ignore": [],
    },
    "preview_alpha": {
        "label": "preview_alpha (stable, 65.3GB)",
        "allow": ["preview_alpha/*", "config.json"],
        "ignore": [],
    },
    "preview_beta": {
        "label": "preview_beta (motion, 65.3GB)",
        "allow": ["preview_beta/*", "config.json"],
        "ignore": [],
    },
}


def default_prism_root() -> str:
    """Return the fixed Prism weight root under ComfyUI models/."""
    import folder_paths
    return os.path.join(folder_paths.models_dir, PRISM_MODELS_SUBDIR)


def resolve_target_dir(target_dir: str = None) -> str:
    """Resolve the download destination.

    Any caller-supplied ``target_dir`` is deliberately ignored; downloads are
    always pinned to the fixed Prism root to prevent a workflow from writing
    weights to an arbitrary folder.
    """
    return os.path.abspath(default_prism_root())


def download_component(component: str, target_dir: str = None, use_mirror: bool = False) -> str:
    """Download ``component`` into the fixed Prism root; returns the snapshot dir.

    ``target_dir`` is accepted for backwards compatibility but ignored -- the
    destination is always :func:`default_prism_root`.
    """
    import huggingface_hub

    if component not in COMPONENT_PATTERNS:
        raise ValueError(f"Unknown component: {component}; options: {list(COMPONENT_PATTERNS)}")

    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    target = resolve_target_dir(target_dir)
    os.makedirs(target, exist_ok=True)

    spec = COMPONENT_PATTERNS[component]
    print(f"[BSAI Prism] Downloading {spec['label']} -> {target}")
    snapshot = huggingface_hub.snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=target,
        allow_patterns=spec["allow"],
        ignore_patterns=spec["ignore"] or None,
        max_workers=4,
    )
    print(f"[BSAI Prism] Done: {spec['label']} ({snapshot})")
    return snapshot


def scan_checkpoints(base_root: str):
    """Scan ``<base_root>/pretrained_models/MOVA-360p`` and ``preview_*``; returns (base_dir, [ckpt_paths])."""
    base_root = os.path.abspath(base_root)
    found_base = os.path.join(base_root, "pretrained_models", "MOVA-360p")
    base_ok = os.path.isfile(os.path.join(found_base, "model_index.json"))
    ckpts = []
    for name in ("preview_alpha", "preview_beta"):
        p = os.path.join(base_root, name, "diffusion_pytorch_model.safetensors")
        if os.path.isfile(p):
            ckpts.append(p)
    return found_base if base_ok else None, ckpts
