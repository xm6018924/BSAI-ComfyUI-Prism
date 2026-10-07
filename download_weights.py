# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— Prism 权重下载（FrancisRing/Prism，共约 208GB）

仓库目录结构（huggingface.co/FrancisRing/Prism）：
    pretrained_models/MOVA-360p/        # 基底模型（视频/音频 DiT、双塔桥、VAE、
                                        #   text encoder、tokenizer、scheduler）
    preview_alpha/diffusion_pytorch_model.safetensors   # 65.3GB，stable 版
    preview_beta/diffusion_pytorch_model.safetensors    # 65.3GB，motion 版
    config.json

组件可单独下载（下载节点勾选即可），落盘结构：
    <target_dir>/
    ├── pretrained_models/MOVA-360p/...
    ├── preview_alpha/diffusion_pytorch_model.safetensors
    └── preview_beta/diffusion_pytorch_model.safetensors
"""
import os

REPO_ID = "FrancisRing/Prism"

COMPONENT_PATTERNS = {
    "base": {
        "label": "MOVA-360p 基底（约 40GB）",
        "allow": [
            "pretrained_models/MOVA-360p/*",
        ],
        "ignore": [],
    },
    "preview_alpha": {
        "label": "preview_alpha（stable，65.3GB）",
        "allow": ["preview_alpha/*", "config.json"],
        "ignore": [],
    },
    "preview_beta": {
        "label": "preview_beta（motion，65.3GB）",
        "allow": ["preview_beta/*", "config.json"],
        "ignore": [],
    },
}


def resolve_target_dir(target_dir: str) -> str:
    return os.path.abspath(os.path.expanduser(target_dir))


def download_component(component: str, target_dir: str, use_mirror: bool = False) -> str:
    """下载指定组件到 target_dir，返回实际落盘目录。"""
    import huggingface_hub

    if component not in COMPONENT_PATTERNS:
        raise ValueError(f"未知组件：{component}，可选 {list(COMPONENT_PATTERNS)}")

    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    target = resolve_target_dir(target_dir)
    os.makedirs(target, exist_ok=True)

    spec = COMPONENT_PATTERNS[component]
    print(f"[BSAI Prism] 开始下载 {spec['label']} -> {target}")
    snapshot = huggingface_hub.snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=target,
        allow_patterns=spec["allow"],
        ignore_patterns=spec["ignore"] or None,
        max_workers=4,
    )
    print(f"[BSAI Prism] 完成：{spec['label']}（{snapshot}）")
    return snapshot


def scan_checkpoints(base_root: str):
    """扫描 <target>/pretrained_models/MOVA-360p 与 preview_*，返回可选列表。"""
    base_root = os.path.abspath(base_root)
    found_base = os.path.join(base_root, "pretrained_models", "MOVA-360p")
    base_ok = os.path.isfile(os.path.join(found_base, "model_index.json"))
    ckpts = []
    for name in ("preview_alpha", "preview_beta"):
        p = os.path.join(base_root, name, "diffusion_pytorch_model.safetensors")
        if os.path.isfile(p):
            ckpts.append(p)
    return found_base if base_ok else None, ckpts
