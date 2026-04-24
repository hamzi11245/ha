import os, io, tempfile, time, json, base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import streamlit as st
import whisper
from google import genai
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, ColorClip
from pydub import AudioSegment
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont

# ================== PAGE CONFIG (SEO-friendly title) ==================
st.set_page_config(
    page_title="AI Video Clipper Pro – Viral Shorts with Smart Captions",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

WHISPER_MODEL = "tiny"   # fastest for real-time use

# ================== 22 CAPTION PRESETS (Opus/Vizard style) ==================
TEMPLATES = {
    "Karaoke": {
        "position": "bottom", "font": "Arial-Bold", "fontsize": 50,
        "color": "white", "stroke_color": "black", "stroke_width": 3,
        "highlight_color": "#FFD700", "bg_color": None
    },
    "Beasty": {
        "position": "bottom", "font": "Impact", "fontsize": 65,
        "color": "#00FF00", "stroke_color": "black", "stroke_width": 5,
        "highlight_color": "#FF00FF", "bg_color": "#333333cc"
    },
    "Deep Diver": {
        "position": "center", "font": "Arial-Black", "fontsize": 55,
        "color": "#00FFFF", "stroke_color": "#000033", "stroke_width": 4,
        "highlight_color": "#FFAA00", "bg_color": "#00002280"
    },
    "Youshaei": {
        "position": "bottom", "font": "Courier New", "fontsize": 48,
        "color": "#FF3366", "stroke_color": "white", "stroke_width": 2,
        "highlight_color": "#FFFF00", "bg_color": "#11111180"
    },
    "Pod P": {
        "position": "top", "font": "Georgia", "fontsize": 44,
        "color": "#FFCC00", "stroke_color": "black", "stroke_width": 2,
        "highlight_color": "#FFFFFF", "bg_color": "#22222290"
    },
    "Mozi": {
        "position": "bottom", "font": "Verdana", "fontsize": 52,
        "color": "#ffffff", "stroke_color": "#6600FF", "stroke_width": 3,
        "highlight_color": "#FF8800", "bg_color": None
    },
    "Popline": {
        "position": "center", "font": "Comic Sans MS", "fontsize": 58,
        "color": "#FF0099", "stroke_color": "white", "stroke_width": 2,
        "highlight_color": "#00FFCC", "bg_color": "#FFFFFF30"
    },
    "Glitch Infinite": {
        "position": "bottom", "font": "Arial-Bold", "fontsize": 54,
        "color": "#FF0000", "stroke_color": "blue", "stroke_width": 2,
        "highlight_color": "#00FF00", "bg_color": "#00000080"
    },
    "Seamless Bounce": {
        "position": "bottom", "font": "Arial-Bold", "fontsize": 50,
        "color": "white", "stroke_color": "#FF00FF", "stroke_width": 3,
        "highlight_color": "#00FFFF", "bg_color": "#22222280"
    },
    "Baby Earthquake": {
        "position": "center", "font": "Arial-Black", "fontsize": 62,
        "color": "#FFAA00", "stroke_color": "black", "stroke_width": 5,
        "highlight_color": "#FF0000", "bg_color": "#33000080"
    },
    "Blur Switch": {
        "position": "top", "font": "Arial", "fontsize": 45,
        "color": "black", "stroke_color": "white", "stroke_width": 3,
        "highlight_color": "#FF00FF", "bg_color": "#FFFFFF80"
    },
    "Highlighter Box": {
        "position": "bottom", "font": "Arial-Bold", "fontsize": 48,
        "color": "white", "stroke_color": "black", "stroke_width": 2,
        "highlight_color": "#FFFF00", "bg_color": "#FFFF0040"
    },
    "Focus": {
        "position": "center", "font": "Georgia", "fontsize": 60,
        "color": "#FFFFFF", "stroke_color": "black", "stroke_width": 4,
        "highlight_color": "#FFD700", "bg_color": "#00000080"
    },
    "Blur In": {
        "position": "bottom", "font": "Arial", "fontsize": 46,
        "color": "#FFFFFF", "stroke_color": "#000000", "stroke_width": 2,
        "highlight_color": "#00FFAA", "bg_color": "#00000060"
    },
    "With Backdrop": {
        "position": "top", "font": "Arial-Bold", "fontsize": 52,
        "color": "#FFD700", "stroke_color": "black", "stroke_width": 3,
        "highlight_color": "#FF00FF", "bg_color": "#22222280"
    },
    "Soft Landing": {
        "position": "center", "font": "Arial", "fontsize": 50,
        "color": "#FFFFFF", "stroke_color": "#444444", "stroke_width": 2,
        "highlight_color": "#FFAA00", "bg_color": "#11111180"
    },
    "Baby Steps": {
        "position": "bottom", "font": "Arial-Bold", "fontsize": 56,
        "color": "#FF3366", "stroke_color": "white", "stroke_width": 3,
        "highlight_color": "#FFFFFF", "bg_color": None
    },
    "Grow": {
        "position": "bottom", "font": "Impact", "fontsize": 60,
        "color": "#00FF00", "stroke_color": "black", "stroke_width": 5,
        "highlight_color": "#FFAA00", "bg_color": "#00330080"
    },
    "Breathe": {
        "position": "center", "font": "Arial", "fontsize": 54,
        "color": "#00CCFF", "stroke_color": "#003366", "stroke_width": 3,
        "highlight_color": "#FFCC00", "bg_color": "#00224480"
    },
    "Bold Bottom White": {
        "position": "bottom", "font": "Arial-Bold", "fontsize": 48,
        "color": "white", "stroke_color": "black", "stroke_width": 3,
        "highlight_color": "#FFD700", "bg_color": None
    },
    "Minimal Top": {
        "position": "top", "font": "Arial", "fontsize": 42,
        "color": "black", "stroke_color": "white", "stroke_width": 2,
        "highlight_color": "#00FFFF", "bg_color": "#ffffff80"
    },
    "TikTok Style": {
        "position": "bottom", "font": "Arial-Bold", "fontsize": 54,
        "color": "white", "stroke_color": "black", "stroke_width": 3,
        "highlight_color": "#FF0000", "bg_color": "#00000080"
    }
}

QUALITIES = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}

# ================== AI / ENERGY HIGHLIGHT DETECTION ==================
def ai_find_top_moments(video_path, transcript, clip_duration, api_key, custom_prompt, top_n):
    client = genai.Client(api_key=api_key)
    ext = os.path.splitext(video_path)[1].lower()
    mime_map = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}
    mime_type = mime_map.get(ext, "video/mp4")
    with open(video_path, "rb") as f:
        video_file = client.files.upload(file=f, config={"mime_type": mime_type})
    while video_file.state == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)

    prompt = f"""You are a pro video editor AI. Analyse the video (visuals + audio + transcript).
Find the top {top_n} most engaging {clip_duration}-second segments.
Consider emotional peaks, visual excitement, key dialogue, laughter, scene changes.
{custom_prompt}
Return ONLY a JSON list of objects with 'start' and 'end' times in seconds.
Example: [{{"start": 10.5, "end": 70.5}}]
Transcript: {transcript}"""
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=[prompt, video_file]
    )
    try:
        moments = json.loads(response.text.strip().replace("'", '"'))
        return moments[:top_n]
    except:
        moments = []
        parts = response.text.replace("[","").replace("]","").split("},{")
        for p in parts:
            nums = [x.strip() for x in p.replace("{","").replace("}","").split(",")]
            if len(nums) >= 2:
                start = float(nums[0].split(":")[-1].strip())
                end = float(nums[1].split(":")[-1].strip())
                moments.append({"start": start, "end": end})
        return moments[:top_n]

def energy_find_top_moments(video_path, clip_duration, top_n):
    audio = AudioSegment.from_file(video_path).set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = 16000
    window = int(sr * 0.5)
    energies, times = [], []
    for i in range(0, len(samples) - window, window):
        chunk = samples[i:i+window]
        energies.append(np.sqrt(np.mean(chunk ** 2)))
        times.append(i / sr)
    energies = gaussian_filter1d(energies, sigma=2)
    clip_frames = int(clip_duration / 0.5)
    moments, used = [], []
    for _ in range(top_n):
        best_score, best_idx = -1, 0
        for i in range(len(energies) - clip_frames):
            overlap = False
            for st_u, end_u in used:
                if not (times[i] + clip_duration <= st_u or times[i] >= end_u):
                    overlap = True
                    break
            if overlap:
                continue
            avg = np.mean(energies[i:i+clip_frames])
            if avg > best_score:
                best_score = avg
                best_idx = i
        if best_score == -1:
            break
        start = times[best_idx]
        moments.append({"start": start, "end": start + clip_duration})
        used.append((start, start + clip_duration))
    return moments

# ================== CAPTION RENDERING (FULLY WORKING) ==================
def _get_pil_font(font_name, size):
    try:
        return ImageFont.truetype(font_name, size)
    except:
        return ImageFont.load_default()

def _hex_to_rgba(hex_color):
    """Convert '#rrggbbaa' to (r,g,b,a) 0-255; if no alpha return (r,g,b,255)"""
    if hex_color is None:
        return (0,0,0,0)
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r,g,b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
        return (r,g,b,255)
    elif len(hex_color) == 8:
        r,g,b,a = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16), int(hex_color[6:8],16)
        return (r,g,b,a)
    return (0,0,0,255)

def make_caption_line(words, template, video_size):
    font = template["font"]
    fs = template["fontsize"]
    color = template["color"]
    stroke = template["stroke_color"]
    sw = template["stroke_width"]
    hi_color = template["highlight_color"]
    bg = template["bg_color"]

    pil_font = _get_pil_font(font, fs)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))

    line_text = " ".join([w["word"] for w in words])
    bbox_line = dummy_draw.textbbox((0,0), line_text, font=pil_font)
    line_w = bbox_line[2] - bbox_line[0]
    line_h = bbox_line[3] - bbox_line[1]

    # Background
    bg_clip = None
    if bg:
        rgba = _hex_to_rgba(bg)
        bg_clip = ColorClip(size=(line_w+20, line_h+20), color=(rgba[0],rgba[1],rgba[2]),
                            duration=words[-1]["end"]).set_opacity(rgba[3]/255)
    else:
        # Transparent background (required for compositing)
        bg_clip = ColorClip(size=(line_w+20, line_h+20), color=(0,0,0,0),
                            duration=words[-1]["end"]).set_mask(None)

    # Full static line
    static_txt = TextClip(txt=line_text, font=font, font_size=fs, color=color,
                         stroke_color=stroke, stroke_width=sw, method='caption')
    static_txt = static_txt.set_duration(words[-1]["end"]).set_position("center")

    line_canvas = CompositeVideoClip([bg_clip, static_txt], size=(line_w+20, line_h+20))

    # Highlight words
    for idx, w in enumerate(words):
        prefix = " ".join([pw["word"] for pw in words[:idx]]) + (" " if idx>0 else "")
        bbox_prefix = dummy_draw.textbbox((0,0), prefix, font=pil_font)
        prefix_w = bbox_prefix[2] - bbox_prefix[0]

        hi_txt = TextClip(txt=w["word"], font=font, font_size=fs, color=hi_color,
                         stroke_color=stroke, stroke_width=sw, method='caption')
        hi_txt = hi_txt.set_duration(w["end"] - w["start"]).set_start(w["start"])

        x_pos = 10 + prefix_w   # 10 = padding/2
        hi_txt = hi_txt.set_position((x_pos, "center"))
        line_canvas = CompositeVideoClip([line_canvas, hi_txt], size=(line_w+20, line_h+20))

    # Position on video
    y_map = {"bottom": video_size[1]*0.85, "top": video_size[1]*0.1, "center": "center"}
    y = y_map[template["position"]]
    return line_canvas.set_position(("center", y))

def add_captions(video_clip, words, template):
    lines, cur, cnt = [], [], 0
    for i, seg in enumerate(words):
        cur.append(seg)
        cnt += 1
        if cnt >= 6 or (i < len(words)-1 and words[i+1]["start"] - seg["end"] > 0.5):
            lines.append(cur)
            cur, cnt = [], 0
    if cur: lines.append(cur)
    caps = []
    for lw in lines:
        caps.append(make_caption_line(lw, template, video_clip.size))
    return CompositeVideoClip([video_clip] + caps)

# ================== PROCESS ONE CLIP ==================
def process_clip(full_video, start, end, words, template, aspect, quality, out_path, add_broll=False):
    clip = full_video.subclip(start, end)
    target_w, target_h = QUALITIES[quality]
    if aspect == "9:16":
        w, h = target_h, target_w
    else:
        w, h = target_w, target_h

    clip = clip.resize(height=h)
    if clip.w > w:
        x_center = clip.w/2
        clip = clip.crop(x1=x_center-w/2, x2=x_center+w/2)
    else:
        clip = clip.resize(width=w)

    cwords = []
    for wd in words:
        if wd["start"] >= start and wd["end"] <= end:
            cwords.append({"word": wd["word"], "start": wd["start"]-start, "end": wd["end"]-start})

    final = add_captions(clip, cwords, template)
    if add_broll:
        broll = ColorClip(size=(w, h), color=(255,255,255), duration=final.duration).set_opacity(0.15)
        final = CompositeVideoClip([final, broll])

    final.write_videofile(out_path, codec="libx264", audio_codec="aac",
                          preset="medium", bitrate="8000k", fps=24, threads=2, logger=None)
    return out_path

# ================== STREAMLIT UI – PRO LEVEL GLOW THEME ==================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: #06060d;
    }
    .stApp {
        background: radial-gradient(circle at 0% 0%, #1a1a2e, #0a0a14);
    }
    /* Sidebar glow effect */
    section[data-testid="stSidebar"] {
        background: rgba(10,10,20,0.85);
        backdrop-filter: blur(24px);
        border-right: 2px solid transparent;
        animation: borderGlow 5s ease infinite;
    }
    @keyframes borderGlow {
        0% { border-color: #ff0066; box-shadow: 0 0 10px #ff0066; }
        25% { border-color: #00ffff; box-shadow: 0 0 10px #00ffff; }
        50% { border-color: #ff00ff; box-shadow: 0 0 10px #ff00ff; }
        75% { border-color: #00ff66; box-shadow: 0 0 10px #00ff66; }
        100% { border-color: #ff0066; box-shadow: 0 0 10px #ff0066; }
    }
    /* Cards */
    .glass-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 0 40px rgba(107,44,245,0.6);
    }
    /* Buttons */
    .primary-btn {
        background: linear-gradient(135deg, #6B2CF5, #d4507b);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 12px 32px;
        font-weight: 600;
        box-shadow: 0 0 25px rgba(107,44,245,0.5);
        transition: all 0.2s;
    }
    .primary-btn:hover {
        transform: scale(1.02);
        box-shadow: 0 0 45px rgba(212,80,123,0.9);
    }
    .virality-badge {
        background: linear-gradient(45deg, #FF416C, #FF4B2B);
        border-radius: 50px;
        padding: 4px 16px;
        font-weight: 700;
        color: white;
        display: inline-block;
        box-shadow: 0 0 20px #FF416C;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #6B2CF5, #d4507b);
        box-shadow: 0 0 15px #d4507b;
    }
    h1, h2, h3 {
        background: linear-gradient(135deg, #6B2CF5, #d4507b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("<h2 style='color:white;'>🎥 Studio Pro</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload video", type=["mp4","mov","mkv","avi"])
    api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIza...")
    st.markdown("---")
    st.markdown("### ⚙️ Clip Settings")
    clip_model = st.radio("Clip model", ["⚡ Energy (fast)", "🧠 AI (Gemini)"], horizontal=True)
    clip_duration = st.slider("Clip length (sec)", 15, 120, 60)
    aspect = st.radio("Aspect ratio", ["9:16", "16:9", "1:1"], horizontal=True)
    quality = st.selectbox("Output quality", ["1080p", "720p", "480p"], index=0)
    custom_prompt = st.text_input("🎯 Find moments about", placeholder="e.g., laughter, product mentions")
    add_broll = st.checkbox("Auto B‑roll overlay", value=False)
    st.markdown("---")
    st.markdown("### ✨ Caption Style")
    # Template selection with beautiful previews
    selected_tpl = st.session_state.get("selected_template", list(TEMPLATES.keys())[0])
    cols = st.columns(2)
    for i, (name, tpl) in enumerate(TEMPLATES.items()):
        with cols[i%2]:
            # create preview image
            try:
                pil_font = _get_pil_font(tpl["font"], tpl["fontsize"])
            except:
                pil_font = ImageFont.load_default()
            img = Image.new("RGBA", (200, 50), (0,0,0,0))
            draw = ImageDraw.Draw(img)
            # background
            if tpl["bg_color"]:
                rgba = _hex_to_rgba(tpl["bg_color"])
                draw.rectangle([(0,0), (200,50)], fill=rgba)
            # text
            draw.text((10,25), "Sample", fill=tpl["color"], font=pil_font, anchor="lm",
                      stroke_width=tpl["stroke_width"], stroke_fill=tpl["stroke_color"])
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; cursor:pointer;"
                 onclick="document.querySelectorAll('[data-testid=stButton] button')[arguments[0]].click()">
                <img src="data:image/png;base64,{img_b64}" width="100%">
                <p style="color:white; margin:0;">{name}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select", key=f"tpl_{i}"):
                st.session_state["selected_template"] = name
    num_clips = st.slider("🔢 Clips to generate", 1, 8, 3)

# ---- MAIN PAGE ----
st.markdown("""
<div style="text-align:center; padding:1rem 0;">
    <h1 style="font-size:3rem;">🎬 AI Video Clipper Pro</h1>
    <p style="color:#aaa; font-size:1.1rem;">Intelligent highlights · Animated captions · Virality scoring</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file and st.button("✨ Generate Viral Clips", use_container_width=True):
    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # 1. Transcription
    with st.status("🔊 Transcribing audio...", expanded=True) as status:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(video_path, word_timestamps=True)
        transcript = result["text"]
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        status.update(label="✅ Transcription complete", state="complete")

    # 2. Find moments
    with st.status("🎯 Detecting best moments...", expanded=True) as status:
        full_video = VideoFileClip(video_path)
        if clip_model == "🧠 AI (Gemini)" and api_key:
            try:
                moments = ai_find_top_moments(video_path, transcript, clip_duration,
                                              api_key, custom_prompt, num_clips)
            except Exception as e:
                st.warning(f"AI fell back to energy: {e}")
                moments = energy_find_top_moments(video_path, clip_duration, num_clips)
        else:
            moments = energy_find_top_moments(video_path, clip_duration, num_clips)
        status.update(label=f"✅ Found {len(moments)} moments", state="complete")

    template_name = st.session_state.get("selected_template", list(TEMPLATES.keys())[0])
    template = TEMPLATES[template_name]

    # 3. Parallel rendering
    progress_bar = st.progress(0)
    output_files = []
    with ThreadPoolExecutor(max_workers=min(4, len(moments))) as executor:
        futures = {}
        for i, moment in enumerate(moments):
            out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            fut = executor.submit(process_clip, full_video, moment["start"], moment["end"],
                                  words, template, aspect, quality, out_path, add_broll)
            futures[fut] = (i, moment, out_path)

        for idx, fut in enumerate(as_completed(futures)):
            i, moment, out_path = futures[fut]
            # Virality score
            audio = AudioSegment.from_file(video_path).set_channels(1).set_frame_rate(16000)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            s_s = int(moment["start"] * 16000)
            e_s = int(moment["end"] * 16000)
            energy = np.sqrt(np.mean(samples[s_s:e_s] ** 2)) if e_s <= len(samples) else 0
            score = min(100, int(energy * 12))
            output_files.append((out_path, moment, score))
            progress_bar.progress((idx + 1) / len(futures))

    full_video.close()
    st.success("All clips generated!")

    # Gallery
    st.markdown("## 🎞️ Your Viral Clips")
    cols = st.columns(3)
    for idx, (file_path, moment, virality) in enumerate(output_files):
        col = cols[idx % 3]
        with col:
            st.video(file_path)
            st.markdown(f"**Moment {idx+1}:** {moment['start']:.1f}s – {moment['end']:.1f}s")
            st.markdown(f'<span class="virality-badge">🔥 Virality {virality}%</span>', unsafe_allow_html=True)
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download", f, file_name=f"clip_{idx+1}.mp4", key=f"dl_{idx}")

st.markdown("<br><div style='text-align:center; color:#555;'>🔒 100% local processing – your videos stay private.</div>", unsafe_allow_html=True)
