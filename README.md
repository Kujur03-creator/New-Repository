# Video Background Remover

## Features
- Upload a video
- AI segmentation on every frame
- Transparent background output as WebM (VP9 alpha)
- Green-screen output
- Custom solid-color background
- Preserves original resolution and aspect ratio, including 9:16 and 16:9
- Attempts to preserve original audio

## Run locally
1. Install Python 3.10 or newer.
2. Install FFmpeg and make sure `ffmpeg` is available in your PATH.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start:
   `python app.py`
5. Open the local URL shown in the terminal.

## Important limitations
- “Every pixel” background removal is not mathematically perfect. The app uses an AI segmentation model to predict foreground/background per frame.
- Hair, motion blur, fast movement, shadows, transparent objects, and multiple overlapping people can need better models or manual refinement.
- Transparent video compatibility varies by browser/player. WebM VP9 with alpha is used because normal MP4/H.264 does not reliably support transparency.
- Long/high-resolution videos can require substantial CPU/GPU power and memory.
