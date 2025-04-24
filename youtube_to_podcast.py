#!/usr/bin/env python3
import argparse
import os
import sys
import json
import asyncio
import whisper
import numpy as np
import torch
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from tqdm import tqdm
import edge_tts
import tempfile
import re
from datasets import Dataset

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print("GPU acceleration enabled!")
    torch.cuda.empty_cache()  # Clear GPU memory before starting
else:
    print("Running on CPU. Install CUDA for faster processing.")

# Dictionary of supported languages with their codes
LANGUAGES = {
    "Afrikaans": "af", "Arabic": "ar", "Armenian": "hy", "Azerbaijani": "az",
    "Belarusian": "be", "Bosnian": "bs", "Bulgarian": "bg", "Catalan": "ca",
    "Chinese": "zh", "Croatian": "hr", "Czech": "cs", "Danish": "da",
    "Dutch": "nl", "English": "en", "Estonian": "et", "Finnish": "fi",
    "French": "fr", "Galician": "gl", "German": "de", "Greek": "el",
    "Hebrew": "he", "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is",
    "Indonesian": "id", "Italian": "it", "Japanese": "ja", "Kannada": "kn",
    "Kazakh": "kk", "Korean": "ko", "Latvian": "lv", "Lithuanian": "lt",
    "Macedonian": "mk", "Malay": "ms", "Marathi": "mr", "Maori": "mi",
    "Nepali": "ne", "Norwegian": "no", "Persian": "fa", "Polish": "pl",
    "Portuguese": "pt", "Romanian": "ro", "Russian": "ru", "Serbian": "sr",
    "Slovak": "sk", "Slovenian": "sl", "Spanish": "es", "Swahili": "sw",
    "Swedish": "sv", "Tagalog": "tl", "Tamil": "ta", "Thai": "th",
    "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", "Vietnamese": "vi",
    "Welsh": "cy"
}

# Edge TTS voices for different languages
VOICES = {
    "en": ["en-US-ChristopherNeural", "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural"],
    "es": ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"],
    "fr": ["fr-FR-HenriNeural", "fr-FR-DeniseNeural"],
    "de": ["de-DE-ConradNeural", "de-DE-KatjaNeural"],
    "it": ["it-IT-DiegoNeural", "it-IT-ElsaNeural"],
    "ja": ["ja-JP-KeitaNeural", "ja-JP-NanamiNeural"],
    "ko": ["ko-KR-InJoonNeural", "ko-KR-SunHiNeural"],
    "zh": ["zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"],
    "ru": ["ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"],
    "pt": ["pt-BR-AntonioNeural", "pt-BR-FranciscaNeural"],
    "hu": ["hu-HU-TamasNeural", "hu-HU-NoemiNeural"],  # Hungarian voices
    # Default fallback for other languages
    "default": ["en-US-ChristopherNeural", "en-US-JennyNeural"]
}

def get_voices_for_language(lang_code):
    """Get appropriate voices for a language"""
    return VOICES.get(lang_code.split('-')[0], VOICES["default"])

def list_languages():
    """Print available languages and their codes"""
    print("\nAvailable languages:")
    for name, code in sorted(LANGUAGES.items()):
        print(f"{name}: {code}")

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

def process_audio_segments(segments, base_output_path, target_lang):
    """Process audio segments with different voices for each speaker"""
    # Get appropriate voices for the target language
    voices = get_voices_for_language(target_lang)
    
    # Create a temporary directory for audio segments
    temp_dir = tempfile.mkdtemp()
    final_audio = AudioSegment.empty()
    
    # Process each segment
    for i, segment in enumerate(segments):
        speaker = segment["speaker"]
        voice = voices[i % len(voices)]  # Alternate between available voices
        
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

def translate_audio(input_file, output_file, source_lang=None, target_lang="en", model_name="medium"):
    """Translate audio using Whisper and generate multi-speaker output"""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device=DEVICE)
    
    print(f"Translating audio to {target_lang}...")
    # Directly use Whisper's translate task
    result = model.transcribe(
        input_file,
        language=source_lang,
        task="translate",
        verbose=True,
        fp16=DEVICE == "cuda"  # Use half-precision on GPU
    )
    
    translated_text = result["text"]
    
    # If target language is not English, translate from English
    if target_lang != "en":
        try:
            print(f"Converting English translation to {target_lang}...")
            from transformers import pipeline
            
            # Create a dataset from the text chunks for batch processing
            chunks = [translated_text[i:i+500] for i in range(0, len(translated_text), 500)]
            dataset = Dataset.from_dict({"text": chunks})
            
            # Initialize translation pipeline
            translator = pipeline("translation",
                               model=f"Helsinki-NLP/opus-mt-en-{target_lang}",
                               device=0 if DEVICE == "cuda" else -1,
                               batch_size=8 if DEVICE == "cuda" else 1)
            
            # Translate in batches
            print("Translating text chunks...")
            translations = []
            for batch in dataset.iter(batch_size=8):
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()  # Clear GPU memory before each batch
                batch_translations = translator(batch["text"])
                translations.extend([t['translation_text'] for t in batch_translations])
            
            translated_text = " ".join(translations)
            
            # Save final translation
            final_text_file = output_file.rsplit('.', 1)[0] + f'_translated.txt'
            with open(final_text_file, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            print(f"Translation saved to: {final_text_file}")
            
        except Exception as e:
            print(f"Warning: Translation to {target_lang} failed. Error: {e}")
            print("Falling back to English translation")
    
    # Extract speaker segments from the translation
    print("Processing speaker segments...")
    segments = extract_speaker_segments(translated_text)
    
    # Save segments with speaker information
    segments_file = output_file.rsplit('.', 1)[0] + '_segments.json'
    with open(segments_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"Speaker segments saved to: {segments_file}")
    
    # Process segments and create final audio
    print("Generating translated audio with multiple voices...")
    process_audio_segments(segments, output_file.rsplit('.', 1)[0], target_lang)
    
    # Clear GPU memory after processing
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return segments

def main():
    parser = argparse.ArgumentParser(description='Convert YouTube videos to translated podcasts')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--output', '-o', help='Output directory', default='output')
    parser.add_argument('--model', '-m', help='Whisper model to use', default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--source-lang', '-s', help='Source language code (optional)')
    parser.add_argument('--target-lang', '-t', help='Target language code (default: en)',
                        default='en')
    parser.add_argument('--ffmpeg-path', help='Path to ffmpeg executable directory')
    parser.add_argument('--list-languages', '-l', action='store_true',
                        help='List available languages and exit')
    
    args = parser.parse_args()
    
    # Show available languages if requested
    if args.list_languages:
        list_languages()
        return
    
    # Validate target language
    target_lang = args.target_lang.lower()
    if target_lang not in [code.lower() for code in LANGUAGES.values()]:
        print(f"Error: Unsupported target language code: {target_lang}")
        list_languages()
        return
    
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
        target_lang=target_lang,
        model_name=args.model
    )
    
    print("\nProcess completed!")
    print(f"Translated audio saved as: {os.path.splitext(audio_output)[0]}_translated.mp3")

if __name__ == "__main__":
    main() 