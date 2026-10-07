# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Prism —— 音视频合成（mp4 mux）与 ffmpeg 定位

与官方 sample_mova_single.save_video_with_audio 的差异：
  - 官方直接调用系统 PATH 上的 `ffmpeg`；本模块优先使用
    imageio_ffmpeg 自带的 ffmpeg 可执行文件（ComfyUI 环境必然存在），
    找不到时再回退到系统 PATH。合成命令参数与官方一致：
    -c:v copy -c:a aac -b:a 192k -movflags +faststart -shortest。
"""
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
import torch


def find_ffmpeg() -> str:
    """返回可用的 ffmpeg 可执行文件路径。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError(
        "未找到 ffmpeg：请安装系统级 ffmpeg（并加入 PATH），或安装 imageio[ffmpeg]。"
    )


def save_video_with_audio(
    frames,
    audio,
    save_path: str,
    fps: float,
    sample_rate: int = 44100,
) -> None:
    """把 PIL 帧序列 + 音频张量合成为 mp4。

    Args:
        frames: PIL.Image 列表（官方 pipeline 输出格式）。
        audio:  [B, 1, T] 或 [1, T] 或 [T] 的 float/int16 张量（-1..1）。
        save_path: 输出 .mp4 路径。
        fps: 视频帧率。
        sample_rate: 音频采样率（Prism 固定 44100，来自 DAC）。
    """
    import imageio

    ffmpeg = find_ffmpeg()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bsai_prism_") as tmp_dir:
        tmp_video = os.path.join(tmp_dir, "video.mp4")
        tmp_audio = os.path.join(tmp_dir, "audio.wav")

        writer = imageio.get_writer(tmp_video, fps=fps, quality=9)
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)
        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]
        channels, samples = audio_np.shape[0], audio_np.shape[1]
        if channels > 2:
            audio_np = audio_np[:2, :]
            channels = 2
        if np.issubdtype(audio_np.dtype, np.floating):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767.0).astype(np.int16)
        elif audio_np.dtype != np.int16:
            audio_np = np.clip(audio_np, -32768, 32767).astype(np.int16)
        if channels == 1:
            interleaved = audio_np.reshape(-1)
        else:
            interleaved = audio_np.T.reshape(-1)
        with wave.open(tmp_audio, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(interleaved.tobytes(order="C"))

        cmd = [
            ffmpeg, "-y",
            "-i", tmp_video,
            "-i", tmp_audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            save_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            # 与官方行为一致：ffmpeg 失败时退化为只保存无声视频，不丢结果
            err = exc.stderr.decode(errors="ignore")[:500] if exc.stderr else ""
            print(f"[BSAI Prism] ffmpeg 合成失败（{err}），退回保存无声视频。")
            shutil.copyfile(tmp_video, save_path)
