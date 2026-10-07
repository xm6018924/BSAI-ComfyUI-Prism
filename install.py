# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— ComfyUI-Manager 安装钩子

Prism 运行时强依赖 CUDA 生态组件（flash-attn / triton / diffusers /
transformers / descript-audiotools），且 ComfyUI 本身也依赖 diffusers。
因此本插件**不自动重装/降级**任何已存在的包，只做缺失校验并给出安装命令。
"""
import subprocess
import sys


def _installed(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


def install():
    missing = []
    checks = [
        ("flash_attn", "flash-attn>=2.8.0"),
        ("triton", "triton>=3.3.0"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers>=4.57.0"),
        ("audiotools", "descript-audiotools>=0.7.2"),
        ("imageio_ffmpeg", "imageio[ffmpeg]>=2.37.0"),
        ("omegaconf", "omegaconf>=2.3.0"),
        ("einops", "einops>=0.8.0"),
        ("safetensors", "safetensors>=0.6.2"),
        ("huggingface_hub", "huggingface_hub"),
    ]
    for mod, dist_name in checks:
        if not _installed(mod):
            missing.append(dist_name)

    if missing:
        print("BSAI ComfyUI Prism 缺失依赖：", ", ".join(missing))
        print("请在 ComfyUI 的 CUDA Python 环境执行：")
        print(f"  {sys.executable} -m pip install {' '.join(missing)}")
        # 不自动安装：避免在 ComfyUI 主环境里破坏现有 diffusers 版本
        return False
    print("BSAI ComfyUI Prism：依赖检查通过，无需安装。")
    return True


if __name__ == "__main__":
    install()
