import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher
import cv2
import pytesseract
import whisper
import yt_dlp

# Configuration
FRAME_RATE = 0.2  # Capture 1 frame every 5 seconds
SIMILARITY_THRESHOLD = 0.7  # For slide deduplication
WHISPER_MODEL = "base"  # Balance of speed and accuracy
SUPPORTED_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv')

def is_youtube_url(source: str) -> bool:
    """Check if source is a YouTube/online video URL"""
    patterns = [
        r'^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$',
        r'^https?:\/\/(www\.)?(vimeo|dailymotion)\.com\/.+$'
    ]
    return any(re.match(pattern, source, re.IGNORECASE) for pattern in patterns)

def extract_audio(video_path: str) -> str:
    """Extract audio from video to WAV format using ffmpeg"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio_path = temp_audio.name
    
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # Disable video
        '-acodec', 'pcm_s16le',  # Audio codec
        '-ar', '16000',  # Sample rate
        '-ac', '1',  # Mono channel
        '-y',  # Overwrite
        audio_path
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        os.unlink(audio_path)
        raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")

def download_youtube_video(url: str) -> tuple:
    """Download YouTube video and return (video_path, audio_path, metadata)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Audio download configuration
        audio_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(tmpdir, 'audio_%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': True,
        }
        
        # Video download configuration
        video_opts = {
            'format': 'bestvideo[ext=mp4]',
            'outtmpl': os.path.join(tmpdir, 'video_%(id)s.%(ext)s'),
            'quiet': True,
        }
        
        # Download audio
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            audio_info = ydl.extract_info(url)
            audio_id = audio_info['id']
            audio_path = os.path.join(tmpdir, f'audio_{audio_id}.wav')
        
        # Download video
        with yt_dlp.YoutubeDL(video_opts) as ydl:
            video_info = ydl.extract_info(url)
            video_id = video_info['id']
            video_path = os.path.join(tmpdir, f'video_{video_id}.mp4')
        
        metadata = {
            'id': audio_id,
            'title': audio_info.get('title', 'Unknown'),
            'source': 'youtube',
            'url': url
        }
        
        return video_path, audio_path, metadata

def transcribe_audio(audio_path: str) -> list:
    """Transcribe audio using Whisper and return segments"""
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, word_timestamps=False)
    return [
        {"start": seg["start"], "end": seg["end"], "type": "speech", "text": seg["text"].strip()}
        for seg in result["segments"]
    ]

def extract_slide_text(video_path: str) -> list:
    """Extract deduplicated slide text from video frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS
    
    frame_interval = max(1, int(fps / FRAME_RATE))
    slide_segments = []
    prev_text = None
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Progress indicator
            if frame_count % (frame_interval * 10) == 0:
                percent = frame_count / total_frames * 100
                print(f"  OCR progress: {percent:.1f}%", end='\r')
            
            # Preprocess frame for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # OCR with Tesseract
            text = pytesseract.image_to_string(gray, config='--psm 6',lang="chi_sim").strip()
            
            # Deduplicate similar slides
            if text and (prev_text is None or 
                SequenceMatcher(None, prev_text, text).ratio() < SIMILARITY_THRESHOLD):
                timestamp = frame_count / fps
                slide_segments.append({
                    "start": timestamp,
                    "type": "slide",
                    "text": re.sub(r'\s+', ' ', text)  # Clean whitespace
                })
                prev_text = text

        frame_count += 1

    cap.release()
    print("  OCR progress: 100%   ")
    return slide_segments

def process_source(source: str, output_file: str, model_size: str) -> bool:
    """Process either local file or YouTube URL"""
    try:
        # YouTube/online video processing
        if is_youtube_url(source):
            print(f"🌐 Processing YouTube video: {source}")
            video_path, audio_path, metadata = download_youtube_video(source)
            print(f"  Downloaded: {metadata['title']}")
        
        # Local video processing
        elif os.path.exists(source):
            if not source.lower().endswith(SUPPORTED_EXTENSIONS):
                raise ValueError(f"Unsupported file format: {source}")
                
            print(f"📁 Processing local video: {os.path.basename(source)}")
            audio_path = extract_audio(source)
            video_path = source
            metadata = {
                'id': os.path.splitext(os.path.basename(source))[0],
                'title': os.path.basename(source),
                'source': 'local',
                'path': os.path.abspath(source)
            }
        
        else:
            raise FileNotFoundError(f"Source not found: {source}")
        
        # Common processing pipeline
        print("💬 Transcribing speech with Whisper...")
        segments = transcribe_audio(audio_path)
        
        print("🖼️ Extracting slide text with OCR...")
        segments.extend(extract_slide_text(video_path))
        segments.sort(key=lambda x: x["start"])
        
        # Write results
        with open(output_file, "a", encoding="utf-8") as f:
            for seg in segments:
                seg.update(metadata)  # Add metadata to each segment
                f.write(json.dumps(seg) + "\n")
        
        # Cleanup
        if is_youtube_url(source):
            os.unlink(audio_path)
            os.unlink(video_path)
        else:
            os.unlink(audio_path)  # Remove temporary audio file
        
        print(f"✅ Successfully processed: {metadata['title']}")
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe videos from local files or YouTube URLs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'sources', 
        nargs='+',
        help='List of YouTube URLs or local video paths'
    )
    parser.add_argument(
        '-o', '--output',
        default='video_transcripts.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size'
    )
    args = parser.parse_args()

    global WHISPER_MODEL
    WHISPER_MODEL = args.model

    # Clear previous output if exists
    if os.path.exists(args.output):
        os.remove(args.output)
    
    success_count = 0
    for source in args.sources:
        print(f"\n{'=' * 50}")
        if process_source(source, args.output, args.model):
            success_count += 1
    
    print(f"\n{'=' * 50}")
    print(f"Processed {success_count}/{len(args.sources)} sources successfully")
    print(f"Transcripts saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()