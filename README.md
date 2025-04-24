# YouTube to Podcast Translator

A command-line tool that converts YouTube videos into translated audio podcasts using OpenAI's Whisper model.
 
## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system

### Installing FFmpeg

#### On Debian/Ubuntu:
```bash
sudo apt update && sudo apt install ffmpeg
```

#### On Windows:
Using Chocolatey:
```bash
choco install ffmpeg
```
Or download from [FFmpeg's official website](https://ffmpeg.org/download.html)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Youtube_to_podcast.git
cd Youtube_to_podcast
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python youtube_to_podcast.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Advanced options:
```bash
python youtube_to_podcast.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output output_directory \
    --model medium \
    --source-lang ja \
    --ffmpeg-path "C:\Program Files\ffmpeg\bin"
```

### Arguments

- `url`: YouTube video URL (required)
- `--output`, `-o`: Output directory (default: 'output')
- `--model`, `-m`: Whisper model to use (choices: tiny, base, small, medium, large; default: medium)
- `--source-lang`, `-s`: Source language code (optional, auto-detected if not specified)
- `--ffmpeg-path`: Path to ffmpeg executable directory (optional, auto-detected in common locations)

### Output

The script will create:
1. An MP3 file containing the audio
2. A text file containing the translated transcript

## Notes

- The larger the model, the better the translation quality but slower the processing
- The tool requires an internet connection to download the YouTube video and the Whisper model
- First-time use will download the specified Whisper model (several GB depending on model size)
- On Windows, if ffmpeg is not found automatically, you'll need to specify its location using --ffmpeg-path 