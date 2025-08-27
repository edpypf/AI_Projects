import os, json
import whisper
import subprocess
import pytesseract
import cv2
from yt_dlp import YoutubeDL
from tqdm import tqdm

video_urls = [
    "https://www.youtube.com/watch?v=dh0pJdgY6Lc"]

output_file = "talks_transcripts.json"
asr_model = whisper.load_model("base")  # 或 tiny / small / medium

def download_audio_and_frames(url, index):
    audio_file = f"audio_{index}.mp3"
    frames_dir = f"frames_{index}"
    os.makedirs(frames_dir, exist_ok=True)

    # 下載音訊
    subprocess.run([
        "yt-dlp", "-f", "bestaudio", "-x", "--audio-format", "mp3",
        "-o", audio_file, url
    ])

    # 下載影片逐幀儲存（每 2 秒一幀）
    subprocess.run([
        "ffmpeg", "-i", audio_file.replace(".mp3", ".mp4"),
        "-vf", "fps=0.5", f"{frames_dir}/frame_%03d.jpg"
    ])

    return audio_file, frames_dir

def extract_ocr_from_images(frames_dir):
    ocr_texts = []
    for fname in sorted(os.listdir(frames_dir)):
        path = os.path.join(frames_dir, fname)
        img = cv2.imread(path)
        if img is not None:
            text = pytesseract.image_to_string(img)
            ocr_texts.append(text.strip())
    return list(filter(None, ocr_texts))

def transcribe_audio(audio_path):
    result = asr_model.transcribe(audio_path)
    return result["segments"]  # 含 timestamps 和 text

def process_video(url, index):
    audio_path, frame_dir = download_audio_and_frames(url, index)
    ocr_lines = extract_ocr_from_images(frame_dir)
    asr_segments = transcribe_audio(audio_path)

    results = []
    for seg in asr_segments:
        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "asr_text": seg["text"],
            "ocr_text": ocr_lines  # 簡化示例（也可與時間對齊）
        })
    return results

# 主執行流程
with open(output_file, "w", encoding="utf-8") as f_out:
    for i, url in enumerate(video_urls):
        segments = process_video(url, i)
        for seg in segments:
            f_out.write(json.dumps(seg, ensure_ascii=False) + "\n")

print(f"✅ 所有影片處理完畢，儲存於 {output_file}")
