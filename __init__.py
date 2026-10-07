# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— ComfyUI 插件入口

基于腾讯混元 + 复旦 + 浙大的 Prism（动态稀疏注意力联合视频-音频生成模型，
原生 720p/1080p/2K）权重（FrancisRing/Prism）设计的 ComfyUI 插件：

  - BSAI Prism 模型加载：MOVA-360p 基底 + preview_alpha/beta 微调权重，
    完整暴露 Prism Block Sparse Attention 配置（含互斥校验）；
  - BSAI Prism 采样：参考图 + 文本 -> 视频 + 音频（同一 denoising 流程）；
  - BSAI Prism 音视频合成：IMAGE + AUDIO -> mp4；
  - BSAI Prism 权重下载：选择性下载 208GB 权重（支持 hf-mirror）；
  - BSAI Prism 卸载模型：释放进程内模型缓存。

运行环境：ComfyUI 的 CUDA Python 环境（python_embeded_cuda），
依赖 diffusers/transformers/flash-attn/triton/descript-audiotools 等。
"""
import importlib.util
import os
import sys

WEB_DIRECTORY = "./web" if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")) else None

# prism_runtime 内部使用绝对导入（from prism_runtime.xxx import ...）。
# ComfyUI 的自定义节点加载器不会把插件目录加入 sys.path，因此在入口显式
# 以顶层包名注册 prism_runtime（不污染 sys.path，子模块经其 __path__ 解析）。
_rt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prism_runtime")
if "prism_runtime" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "prism_runtime", os.path.join(_rt_dir, "__init__.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["prism_runtime"] = _mod
    _spec.loader.exec_module(_mod)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
