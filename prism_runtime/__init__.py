# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— Prism 推理运行时（vendored）

本包是 Prism 官方推理运行时（Tencent-Hunyuan/Prism 仓库 `hymm/` 目录）的
ComfyUI 适配副本，MIT 许可，来源保留如下：

    Prism: Dynamic Sparse Attention for Native 2K Joint Video-Audio
    Generation Model Training
    Shuyuan Tu, Qi Tian, Yinming Huang, Yue Wu, Xintong Han, Kaihang Pan,
    Weijie Kong, Jiangfeng Xiong, Jian-Wei Zhang, Zuxuan Wu, Yu-Gang Jiang
    (Fudan University / Tencent Hunyuan / Zhejiang University)
    https://github.com/Tencent-Hunyuan/Prism
    License: MIT

适配内容（相对官方 hymm/ 的差异，全部为工程性调整，不影响数值结果）：
  1. 包名 `hymm` -> `prism_runtime`，避免与用户自建 Prism 目录/环境冲突；
  2. 移除 wan_video_dit.py 模块导入期的 print 调试输出；
  3. 不打包训练/FSDP/数据管线相关模块，只保留单卡推理必需的最小依赖闭包；
  4. 未做任何数值或接口语义改动：denoising 循环、BSA 稀疏注意力、
     flow-match 调度、VAE 编解码路径与官方 sample_mova_single.py 一致。

运行前提（由 BSAI Prism 加载节点在运行时校验并给出明确报错）：
  - CUDA >= 12.4（flash-attn 必需），ComfyUI 使用 python_embeded_cuda 环境；
  - diffusers / transformers / flash_attn / triton / descript-audiotools
    等依赖已存在于该环境（见插件根目录 requirements.txt）。
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
