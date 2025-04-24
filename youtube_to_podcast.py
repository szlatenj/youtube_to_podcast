#!/usr/bin/env python3
import argparse
import os
import sys
import json
import asyncio
import whisper
import numpy as np
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from tqdm import tqdm
import edge_tts
import tempfile
import re 

def find_ffmpeg_windows():
    """Try to find ffmpeg in common Windows locations"""
    possible_paths = [
        os.path.expandvars(r"%ProgramFiles%\ffmpeg\bin\ffmpeg.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\ffmpeg\bin\ffmpeg.exe"),
        os.path.expandvars(r"%ChocolateyInstall%\bin\ffmpeg.exe"),
        # Scoop usually installs here
        os.path.expandvars(r"%USERPROFILE%\scoop\shims\ffmpeg.exe"),
    ]
    
    for path in possible_paths:
        if os.path.isfile(path):
            return os.path.dirname(path)
    return None

def setup_ffmpeg_path(ffmpeg_path):
    """Setup ffmpeg path in environment variables"""
    if ffmpeg_path:
        # Add ffmpeg to PATH for whisper
        if sys.platform == 'win32':
            os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
        return ffmpeg_path
    return None

def download_youtube_audio(url, output_path, ffmpeg_location=None):
    """Download audio from YouTube video"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
    }
    
    if ffmpeg_location:
        ydl_opts['ffmpeg_location'] = ffmpeg_location
    
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            return info.get('title', 'video')
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

async def text_to_speech(text, voice, output_file):
    """Convert text to speech using Edge TTS"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def create_silent_audio(duration_ms):
    """Create a silent audio segment"""
    return AudioSegment.silent(duration=duration_ms)

def extract_speaker_segments(text):
    """Extract speaker segments from Whisper output"""
    # Pattern to match speaker labels like "Speaker 1:" or [Speaker 1]
    pattern = r'(?:^|\n)(?:\[?(?:Speaker|SPEAKER)\s*[\d\w]+\]?:?\s*)(.*?)(?=(?:\n\[?(?:Speaker|SPEAKER)|$))'
    segments = []
    current_speaker = "Speaker 1"
    
    # Split by common speaker indicators
    parts = re.split(r'\n(?:\[?(?:Speaker|SPEAKER)\s*[\d\w]+\]?:?\s*)', text)
    if len(parts) <= 1:  # No speaker labels found
        # Try splitting by double newlines or other indicators
        parts = re.split(r'\n\n+|\.\s+(?=[A-Z])', text)
        
    for i, part in enumerate(parts):
        if part.strip():
            segments.append({
                "speaker": f"Speaker {(i % 2) + 1}",  # Alternate between Speaker 1 and 2
                "text": part.strip()
            })
    
    return segments

def process_audio_segments(segments, base_output_path):
    """Process audio segments with different voices for each speaker"""
    # Available voices for different speakers
    voices = {
        "Speaker 1": "en-US-ChristopherNeural",  # Male voice
        "Speaker 2": "en-US-JennyNeural",        # Female voice
        "Speaker 3": "en-US-GuyNeural",          # Alternative male voice
        "Speaker 4": "en-US-AriaNeural",         # Alternative female voice
    }
    
    # Create a temporary directory for audio segments
    temp_dir = tempfile.mkdtemp()
    final_audio = AudioSegment.empty()
    
    # Process each segment
    for i, segment in enumerate(segments):
        speaker = segment["speaker"]
        voice = voices.get(speaker, voices["Speaker 1"])
        
        # Generate speech for the segment
        temp_file = os.path.join(temp_dir, f"segment_{i}.mp3")
        asyncio.run(text_to_speech(segment["text"], voice, temp_file))
        
        # Add small pause between segments
        if i > 0:
            final_audio += create_silent_audio(500)  # 500ms pause
        
        # Add the segment to final audio
        segment_audio = AudioSegment.from_file(temp_file)
        final_audio += segment_audio
        
        # Clean up temporary file
        os.remove(temp_file)
    
    # Save final audio
    final_audio.export(base_output_path + "_translated.mp3", format="mp3")
    
    # Clean up temporary directory
    os.rmdir(temp_dir)

def translate_audio(input_file, output_file, source_lang=None, model_name="medium"):
    """Translate audio using Whisper and generate multi-speaker output"""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print("Transcribing and translating audio...")
    result = model.transcribe(
        input_file,
        task="translate",
        language=source_lang,
        verbose=True
    )
    
    # Extract speaker segments from the translation
    print("Processing speaker segments...")
    segments = extract_speaker_segments(result["text"])
    
    # Process segments and create final audio
    print("Generating translated audio with multiple voices...")
    process_audio_segments(segments, output_file.rsplit('.', 1)[0])
    
    # Save the translated text for reference
    text_file = output_file.rsplit('.', 1)[0] + '.txt'
    with open(text_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    
    print(f"Translation saved to: {text_file}")
    return segments

def main():
    parser = argparse.ArgumentParser(description='Convert YouTube videos to translated podcasts')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--output', '-o', help='Output directory', default='output')
    parser.add_argument('--model', '-m', help='Whisper model to use', default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--source-lang', '-s', help='Source language (optional)')
    parser.add_argument('--ffmpeg-path', help='Path to ffmpeg executable directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Handle ffmpeg path
    ffmpeg_path = args.ffmpeg_path
    if not ffmpeg_path and sys.platform == 'win32':
        ffmpeg_path = find_ffmpeg_windows()
        if ffmpeg_path:
            print(f"Found ffmpeg at: {ffmpeg_path}")
        else:
            print("Warning: Could not find ffmpeg automatically. Please install it or specify path with --ffmpeg-path")
            return
    
    # Setup ffmpeg in PATH for both yt-dlp and whisper
    ffmpeg_path = setup_ffmpeg_path(ffmpeg_path)
    if not ffmpeg_path:
        print("Error: ffmpeg path not set")
        return
    
    # Generate output paths
    base_path = os.path.join(args.output, 'video')
    audio_output = f"{base_path}.mp3"
    
    # Download YouTube video audio
    print("Downloading YouTube video...")
    video_title = download_youtube_audio(args.url, base_path, ffmpeg_path)
    if not video_title:
        return
    
    # Translate the audio
    print("\nTranslating audio...")
    translated_segments = translate_audio(
        audio_output,
        audio_output,
        source_lang=args.source_lang,
        model_name=args.model
    )
    
    print("\nProcess completed!")
    print(f"Translated audio saved as: {os.path.splitext(audio_output)[0]}_translated.mp3")

if __name__ == "__main__":
    main() 