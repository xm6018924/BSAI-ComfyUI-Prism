<div align="center">

# BSAI ComfyUI Prism

**Native 2K Joint Video-Audio Generation with Block Sparse Attention — ComfyUI Plugin**

**原生 2K 联合视频-音频生成 + 动态块稀疏注意力 —— ComfyUI 插件**

[![BSAI](https://img.shields.io/badge/BSAI-ComfyUI-blue?style=flat-square)]()
[![Prism](https://img.shields.io/badge/Model-Prism-green?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.4+-green?style=flat-square)]()

**Reference Image + Text → Synchronized Video + Audio · Native 720p / 1080p / 2K**

**参考图 + 文本 → 同步生成视频 + 音频 · 原生支持 720p / 1080p / 2K**

---

[**English**](#english) | [**中文**](#中文)

</div>

---

<a name="english"></a>
## 📖 English

### Introduction

BSAI ComfyUI Prism is a ComfyUI plugin built on top of the **Prism** model — a native joint video-audio generation diffusion model jointly developed by Tencent Hunyuan, Fudan University, and Zhejiang University (arXiv:2610.05416). Prism's core innovation is **BSA (Block Sparse Attention)**, a dynamic sparse attention mechanism that achieves **2.5× training speedup** with improved quality compared to full attention.

This plugin wraps the official Prism inference runtime into 5 easy-to-use ComfyUI nodes, exposing the full BSA parameter space with mutual-exclusion validation.

### ✨ Features

- 🎬 **Native Video-Audio Joint Generation** — Video and audio generated simultaneously in a single denoising pipeline
- ⚡ **Block Sparse Attention (BSA)** — Dynamic sparse attention with IVPQ adaptive block shapes, Top-k + Top-p hybrid sparsity
- 🎯 **5 ComfyUI Nodes** — Model loader, sampler, video encoder, weight downloader, model unloader
- 📐 **Resolution Presets** — 480p / 720p / 1080p / 2K with automatic shift & VAE tiling configuration
- 💾 **Smart Memory Management** — CPU offload, VAE tiling, model caching, expert freeing after switch
- 🔊 **Structured Audio Prompts** — Support `<music> <sfx> <speech> <text> <lyrics>` tags
- 🇨🇳 **Built-in Chinese Negative Prompt** — Official default negative prompt in Chinese

### 📦 Nodes

| Node | Description |
|------|-------------|
| **BSAIPrismLoader** | Load MOVA base + preview checkpoint, full BSA configuration |
| **BSAIPrismSampler** | Reference image + text → video + audio with full sampling control |
| **BSAIPrismVideoEncode** | IMAGE + AUDIO → MP4 (standalone video encoding) |
| **BSAIPrismWeightsDownload** | Selective weight download with hf-mirror support |
| **BSAIPrismUnload** | Release all model cache from GPU memory |

### 🔧 Installation

#### Requirements

| Item | Requirement |
|------|-------------|
| OS | Windows / Linux |
| Python | ≥ 3.10 |
| GPU | NVIDIA CUDA ≥ 12.4 (required for flash-attn) |
| VRAM | 480p/720p: 24GB+ with CPU offload; 1080p/2K: 80GB+ recommended |
| Disk | ~208 GB for full weights |

#### Step 1: Install the Plugin

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BSAI-Official/BSAI-ComfyUI-Prism.git
```

Then restart ComfyUI. Nodes are located under **BSAI/Prism** menu.

#### Step 2: Install Dependencies

Most dependencies are already included in standard ComfyUI CUDA environments. Use `install.py` to verify:

```bash
cd BSAI-ComfyUI-Prism
python install.py
```

Key dependencies:
```
diffusers>=0.33.0  transformers>=4.57.0  accelerate  safetensors  einops
flash-attn>=2.8.0  triton>=3.3.0  descript-audiotools>=0.7.2
imageio[ffmpeg]  omegaconf  ftfy  huggingface_hub
```

#### Step 3: Download Weights (~208 GB)

**Option A — From ComfyUI:** Use the `BSAIPrismWeightsDownload` node to download components selectively.

**Option B — Command line:**
```bash
python -m huggingface_hub download --resume-download FrancisRing/Prism \
  --local-dir ComfyUI/models/diffusers/BSAI-Prism \
  --include "pretrained_models/MOVA-360p/*" "preview_alpha/*" "preview_beta/*"
```

Expected directory structure:
```
ComfyUI/models/diffusers/BSAI-Prism/
├── pretrained_models/MOVA-360p/
│   ├── video_dit/  video_dit_2/  audio_dit/  dual_tower_bridge/
│   ├── video_vae/  audio_vae/  text_encoder/  tokenizer/  scheduler/
│   └── model_index.json
├── preview_alpha/diffusion_pytorch_model.safetensors   # 65.3GB
└── preview_beta/diffusion_pytorch_model.safetensors    # 65.3GB
```

### 🧩 Model Overview

Prism uses a **base + fine-tune** two-layer structure. You need **1 base + 1 checkpoint** to run inference.

| Folder | Size | Role | Description |
|--------|------|------|-------------|
| **pretrained_models/MOVA-360p** | ~47 GB | 🏗️ Base (skeleton) | Required. Provides the model architecture: video_dit, audio_dit, VAE, text_encoder, etc. Think of it as the "house frame". |
| **preview_alpha** | ~65 GB | 🎨 Fine-tune A | Choose one. First preview release, stable and reliable. The "standard finish". |
| **preview_beta** | ~65 GB | 🎨 Fine-tune B | Choose one. Second preview release, typically higher quality and more cinematic. The "premium finish". |

**Valid combinations:**
- ✅ `MOVA-360p` + `preview_alpha` — works, recommended starting point
- ✅ `MOVA-360p` + `preview_beta` — works, higher quality
- ⚠️ `MOVA-360p` only (no fine-tune) — works but very poor quality, not recommended
- ❌ `preview_alpha/beta` only (no base) — won't work at all

**alpha vs beta — which to choose?**

| Aspect | preview_alpha | preview_beta |
|--------|---------------|--------------|
| Version | 1st preview | 2nd preview |
| Quality | Solid, reliable | Often more refined & cinematic |
| Style | Realistic leaning | Can be more stylized / filmic |
| Recommendation | Start here | Try when chasing best quality |

> 💡 **Tip:** Both checkpoints share the same base. You can switch between alpha and beta in the dropdown without reloading the base model (it stays cached).

### 🚀 Quick Start

1. Download weights (see above)
2. Load `examples/BSAI_Prism_i2v_720p_workflow.json` into ComfyUI
3. Select a reference image in `LoadImage`
4. Configure model paths in `BSAI Prism Loader`
5. Enter your prompt in `BSAI Prism Sampler` and run
6. Output: `output/BSAI_Prism/*.mp4` (with audio), plus IMAGE frames and AUDIO tensor

**Prompt example (official style):**
```
Continuous <sfx>rhythmic percussive drum beats</sfx>.
A traffic police officer on a sunlit city street, directing cars with dynamic movements.
Deep rhythmic male vocalizations: <speech>hm, ooh, ah, ha-ya</speech>.
```

### ⚙️ Parameter Reference

| Preset | Resolution (H×W) | visual_shift | audio_shift | VAE Tiling | Recommendation |
|--------|-------------------|--------------|-------------|------------|----------------|
| 480p | 480×848 | 7.0 | 7.0 | off | Any 24GB+ GPU (offload) |
| 720p | 720×1280 | 7.0 | 7.0 | off | 80GB + offload |
| 1080p | 1072×1920 | 9~13 | 9.0 | on | 4×80GB FSDP |
| 2K | 1440×2560 | 13~17 | 11.0 | on | 4+×80GB FSDP |

BSA sparsity: `0.93` (fastest) → `0.85` → `0.75` (official default, balanced).

### 🧠 Architecture

```
Reference Image → Video VAE Encode ──┐
                                      ├──▶ MOVABridge (Dual-Tower DiT + BSA)
Text Prompt → UMT5 → Embeddings ────┬┘       │
Audio Prompt → UMT5 → Embeddings ───┤   50-step denoising + CFG
Negative Prompt → UMT5 → Embeddings ─┘        │
                                               ▼
                                  Video VAE Decode + DAC Audio Decode
                                               │
                                               ▼
                                  PIL Frames + Audio → FFmpeg → MP4
```

### 🗺️ Roadmap

- [x] Single-GPU load / sample / encode / download / unload nodes
- [x] Full BSA parameter exposure with mutual-exclusion validation
- [x] Resolution presets with automatic shift & tiling
- [ ] FSDP multi-GPU 1080p/2K inference
- [ ] XPU/CPU VAE decode routing (multi-hardware coordination)
- [ ] Preview checkpoint hot-swap (alpha ↔ beta without reloading base)

### 📄 License

- Plugin code: MIT License
- Prism model & official code: MIT License (Tencent Hunyuan / Fudan / Zhejiang University)
- Weights: MIT License (FrancisRing/Prism @ Hugging Face)

### 🙏 Acknowledgments

- **Prism Team** — Tencent Hunyuan / Fudan University / Zhejiang University
  - [Paper](https://arxiv.org/abs/2610.05416) · [Code](https://github.com/Tencent-Hunyuan/Prism) · [Project Page](https://francis-rings.github.io/Prism/)
- **BSAI ComfyUI Team** — Plugin adaptation and integration

---

<a name="中文"></a>
## 📖 中文

### 介绍

BSAI ComfyUI Prism 是基于 **Prism** 模型打造的 ComfyUI 插件。Prism 是由腾讯混元、复旦大学、浙江大学联合发布的原生联合视频-音频生成扩散模型（arXiv:2610.05416），其核心创新是 **BSA（Block Sparse Attention，动态块稀疏注意力）**，相比全注意力实现了 **2.5 倍训练加速**，同时质量更优。

本插件将官方 Prism 推理运行时封装为 5 个易用的 ComfyUI 节点，完整暴露 BSA 全部参数并内置互斥校验。

### ✨ 特性

- 🎬 **原生联合视频-音频生成** — 视频和音频在同一去噪流水线中同步生成
- ⚡ **动态块稀疏注意力 (BSA)** — IVPQ 自适应块形状、Top-k + Top-p 混合稀疏
- 🎯 **5 个 ComfyUI 节点** — 模型加载、采样、音视频合成、权重下载、模型卸载
- 📐 **分辨率档位预设** — 480p / 720p / 1080p / 2K，自动配置 shift 和 VAE tiling
- 💾 **智能显存管理** — CPU offload、VAE 分块解码、模型缓存、切换后释放专家
- 🔊 **结构化音频提示词** — 支持 `<music> <sfx> <speech> <text> <lyrics>` 标签
- 🇨🇳 **内置中文负向提示** — 官方默认中文负向提示词

### 📦 节点说明

| 节点 | 说明 |
|------|------|
| **BSAIPrismLoader** | 加载 MOVA 基底 + preview 检查点，完整 BSA 配置 |
| **BSAIPrismSampler** | 参考图 + 文本 → 视频 + 音频，完整采样控制 |
| **BSAIPrismVideoEncode** | IMAGE + AUDIO → MP4（独立音视频合成） |
| **BSAIPrismWeightsDownload** | 选择性权重下载，支持 hf-mirror 镜像 |
| **BSAIPrismUnload** | 释放 GPU 中全部模型缓存 |

### 🔧 安装

#### 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows / Linux |
| Python | ≥ 3.10 |
| GPU | NVIDIA CUDA ≥ 12.4（flash-attn 必须） |
| 显存 | 480p/720p：24GB+ 配合 CPU offload；1080p/2K：推荐 80GB+ |
| 磁盘空间 | 完整权条约 208 GB |

#### 第一步：安装插件

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BSAI-Official/BSAI-ComfyUI-Prism.git
```

重启 ComfyUI。节点位于菜单 **BSAI/Prism** 下。

#### 第二步：安装依赖

标准 ComfyUI CUDA 环境通常已包含大部分依赖。使用 `install.py` 校验：

```bash
cd BSAI-ComfyUI-Prism
python install.py
```

核心依赖：
```
diffusers>=0.33.0  transformers>=4.57.0  accelerate  safetensors  einops
flash-attn>=2.8.0  triton>=3.3.0  descript-audiotools>=0.7.2
imageio[ffmpeg]  omegaconf  ftfy  huggingface_hub
```

#### 第三步：下载权重（约 208 GB）

**方式 A — 插件内下载：** 使用 `BSAIPrismWeightsDownload` 节点选择性下载。

**方式 B — 命令行：**
```bash
python -m huggingface_hub download --resume-download FrancisRing/Prism ^
  --local-dir ComfyUI\models\diffusers\BSAI-Prism ^
  --include "pretrained_models/MOVA-360p/*" "preview_alpha/*" "preview_beta/*"
```

预期目录结构：
```
ComfyUI/models/diffusers/BSAI-Prism/
├── pretrained_models/MOVA-360p/
│   ├── video_dit/  video_dit_2/  audio_dit/  dual_tower_bridge/
│   ├── video_vae/  audio_vae/  text_encoder/  tokenizer/  scheduler/
│   └── model_index.json
├── preview_alpha/diffusion_pytorch_model.safetensors   # 65.3GB
└── preview_beta/diffusion_pytorch_model.safetensors    # 65.3GB
```

### 🧩 模型说明

Prism 采用 **基底 + 微调** 的双层结构。运行推理需要 **1 个基底 + 1 个微调权重**。

| 目录 | 大小 | 角色 | 说明 |
|------|------|------|------|
| **pretrained_models/MOVA-360p** | ~47 GB | 🏗️ 基底（骨架） | 必须有。提供模型整体结构：video_dit、audio_dit、VAE、text_encoder 等。相当于"毛坯房"。 |
| **preview_alpha** | ~65 GB | 🎨 微调版本 A | 二选一。第一代预览权重，稳定可靠。相当于"标准装修"。 |
| **preview_beta** | ~65 GB | 🎨 微调版本 B | 二选一。第二代预览权重，画质更精细、更有电影感。相当于"轻奢装修"。 |

**正确搭配：**
- ✅ `MOVA-360p` + `preview_alpha` — 可用，推荐先试这个
- ✅ `MOVA-360p` + `preview_beta` — 可用，质量更高
- ⚠️ 只有 `MOVA-360p`（无微调） — 能跑但效果很差，不推荐
- ❌ 只有 `preview_alpha/beta`（无基底） — 完全不能用

**alpha 和 beta 怎么选？**

| 对比项 | preview_alpha | preview_beta |
|--------|---------------|--------------|
| 版本 | 第一代预览 | 第二代预览 |
| 画质 | 稳定、可靠 | 通常更精细、更有电影感 |
| 风格 | 偏真实感 | 可能更有风格化/胶片感 |
| 推荐 | 先试这个 | 追求最佳效果时再试 |

> 💡 **小贴士：** 两个微调权重共用同一个基底。在下拉框切换 alpha/beta 时，基底会复用缓存，不需要重新加载。

### 🚀 快速开始

1. 下载权重（见上文）
2. 将 `examples/BSAI_Prism_i2v_720p_workflow.json` 拖入 ComfyUI
3. 在 `LoadImage` 中选择参考图
4. 在 `BSAI Prism 模型加载` 中配置模型路径
5. 在 `BSAI Prism 采样` 中输入提示词并运行
6. 输出：`output/BSAI_Prism/*.mp4`（含音频），以及 IMAGE 帧序列和 AUDIO 张量

**提示词示例（官方风格）：**
```
Continuous <sfx>rhythmic percussive drum beats</sfx>.
一名交警在阳光明媚的城市街道上指挥交通，动作充满活力。
低沉有节奏的男性发声：<speech>hm, ooh, ah, ha-ya</speech>.
```

### ⚙️ 参数速查

| 档位 | 分辨率 (高×宽) | visual_shift | audio_shift | VAE 分块 | 推荐配置 |
|------|----------------|--------------|-------------|----------|----------|
| 480p | 480×848 | 7.0 | 7.0 | 关 | 任意 24GB+ 显卡（offload） |
| 720p | 720×1280 | 7.0 | 7.0 | 关 | 80GB + offload |
| 1080p | 1072×1920 | 9~13 | 9.0 | 开 | 4×80GB FSDP |
| 2K | 1440×2560 | 13~17 | 11.0 | 开 | 4+×80GB FSDP |

BSA 稀疏率：`0.93`（最快）→ `0.85` → `0.75`（官方默认，均衡）。

### 🧠 架构

```
参考图 → 视频VAE编码 ─────────────┐
                                  ├──▶ MOVABridge（双塔DiT + BSA稀疏注意力）
文本提示 → UMT5 → 嵌入向量 ──────┬┘       │
音频提示 → UMT5 → 嵌入向量 ──────┤   50步去噪 + CFG
负向提示 → UMT5 → 嵌入向量 ──────┘        │
                                           ▼
                              视频VAE解码 + DAC音频解码
                                           │
                                           ▼
                              PIL帧序列 + 音频 → FFmpeg → MP4
```

### 🗺️ 路线图

- [x] 单卡加载 / 采样 / 合成 / 下载 / 卸载节点
- [x] BSA 全参数透出（含互斥校验）
- [x] 分辨率档位预设 + 自动 shift/tiling
- [ ] FSDP 多卡 1080p/2K 推理
- [ ] XPU/CPU VAE 解码路由（多硬件协调）
- [ ] Preview 检查点热切换（alpha ↔ beta 不重载基底）

### 📄 许可证

- 插件代码：MIT License
- Prism 模型与官方代码：MIT License（腾讯混元 / 复旦大学 / 浙江大学）
- 模型权重：MIT License（FrancisRing/Prism @ Hugging Face）

### 🙏 致谢

- **Prism 团队** — 腾讯混元 / 复旦大学 / 浙江大学
  - [论文](https://arxiv.org/abs/2610.05416) · [代码](https://github.com/Tencent-Hunyuan/Prism) · [项目主页](https://francis-rings.github.io/Prism/)
- **BSAI ComfyUI 团队** — 插件适配与整合

---

<div align="center">

**Made with ❤️ by BSAI Team**

[BSAI 百声AI](https://github.com/BSAI-Official) · ComfyUI 整合包

</div>
