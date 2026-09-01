
import os, uuid, shutil, subprocess, tempfile
from pathlib import Path
import gradio as gr
import cv2
import numpy as np
from rembg import remove, new_session

session = None

def get_session():
    global session
    if session is None:
        session = new_session("u2net_human_seg")
    return session

def process_video(video_path, mode, bg_color, output_format, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not read this video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    work = Path(tempfile.mkdtemp())
    silent = work / "processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise gr.Error("Could not create output video.")

    # Parse #RRGGBB for replacement background.
    rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
    bg = np.full((h, w, 3), rgb[::-1], dtype=np.uint8)

    model = get_session()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Analyze every frame at pixel level through segmentation model.
        rgba = remove(frame, session=model, alpha_matting=False)
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        fg = rgba[:, :, :3]

        if mode == "Green screen":
            replacement = np.full((h, w, 3), (0, 255, 0), dtype=np.uint8)
        else:
            replacement = bg

        out = (fg * alpha[..., None] + replacement * (1 - alpha[..., None])).astype(np.uint8)
        writer.write(out)
        i += 1
        if i % 2 == 0:
            progress(i / total, desc=f"Processing frame {i}/{total}")

    cap.release()
    writer.release()

    # Re-encode and copy original audio where possible.
    final = work / f"background_removed_{uuid.uuid4().hex}.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(silent), "-i", video_path,
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        shutil.copy(silent, final)

    # True transparent video support depends heavily on browser/player support.
    # MP4/H.264 does not preserve alpha reliably. For transparent output use WebM VP9.
    if mode == "Transparent":
        transparent = work / f"transparent_{uuid.uuid4().hex}.webm"
        # Re-process into PNG sequence with alpha for reliable VP9 alpha.
        cap = cv2.VideoCapture(video_path)
        frames_dir = work / "frames"
        frames_dir.mkdir()
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgba = remove(frame, session=model)
            cv2.imwrite(str(frames_dir / f"{idx:08d}.png"), rgba)
            idx += 1
        cap.release()
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "%08d.png"),
            "-i", video_path, "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
            "-c:a", "libopus", "-shortest", str(transparent)
        ]
        subprocess.run(cmd, check=True)
        return str(transparent)

    return str(final)

with gr.Blocks(title="Video Background Remover") as demo:
    gr.Markdown("""
    # Video Background Remover
    Upload a video, remove the background frame-by-frame, and export with:
    - **Transparent background** (WebM/VP9 alpha)
    - **Green screen**
    - **Custom solid-color background**

    Both **9:16 and 16:9** are preserved automatically because the app keeps the original dimensions.
    """)
    with gr.Row():
        video = gr.Video(label="Upload video")
        output = gr.Video(label="Processed video")
    with gr.Row():
        mode = gr.Radio(
            ["Transparent", "Green screen", "Custom color"],
            value="Green screen", label="Background mode"
        )
        bg_color = gr.ColorPicker(value="#202020", label="Custom background color")
    run = gr.Button("Remove Background", variant="primary")
    run.click(process_video, [video, mode, bg_color, gr.State("auto")], output)

demo.launch()
