import streamlit as st
import sqlite3
import hashlib
import os
from contextlib import closing
import requests
import io
import requests
import tempfile

# ---------------------- APP CONFIG ----------------------
st.set_page_config(layout="wide")
DB_NAME = "users.db"

LOGIN_AUDIO_FILE = "login.mp3"
SIGNUP_AUDIO_FILE = "signup.mp3"
HOME_AUDIO_FILE = None

# ---------------------- DATABASE HELPERS ----------------------
def get_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL
        );
    """)
    return conn


def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${h}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, h = stored_hash.split("$")
    except ValueError:
        return False

    new_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return new_hash == h


def create_user(name, email, password):
    conn = get_connection()
    with closing(conn):
        try:
            conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, hash_password(password))
            )
            conn.commit()
            return True, None
        except sqlite3.IntegrityError:
            return False, "User already exists."

from contextlib import closing

def get_user_by_username(name):
    conn = get_connection()
    with closing(conn):
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, email, password_hash FROM users WHERE name = ?",
            (name,)
        )
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "password_hash": row[3]
            }
        return None


# ---------------------- NAVBAR ----------------------
def navbar(current_view: str):
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown("## welcome to application")

    if current_view == "signup":
        audio = SIGNUP_AUDIO_FILE
    elif current_view == "login":
        audio = LOGIN_AUDIO_FILE
    else:
        # audio = HOME_AUDIO_FILE
        pass

    with cols[1]:
        if st.button("🔊 Audio"):
            if audio is None:
                st.info("No audio available.")
            elif os.path.exists(audio):
                st.audio(audio)
            else:
                st.warning(f"Audio '{audio}' not found.")

import streamlit as st
import speech_recognition as sr

# --- Voice-to-Text Function ---
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text.strip().lower()
    except:
        return ""


# ---------------------- SIGNUP PAGE ----------------------

def signup_page():
    st.subheader("Create Account")

    # Initialize default values
    name = st.session_state.get("v_name", "")
    email = st.session_state.get("v_email", "")
    pw = st.session_state.get("v_pw", "")
    cpw = st.session_state.get("v_cpw", "")

    with st.form("signup"):
        col1, col2 = st.columns([4, 1])
        with col1:
            name = st.text_input("Name", value=name)
        with col2:
            if st.form_submit_button("🎙️", key="mic_name"):
                st.session_state.v_name = voice_input()
                st.rerun()

        col3, col4 = st.columns([4, 1])
        with col3:
            email = st.text_input("Email", value=email)
        with col4:
            if st.form_submit_button("🎙️", key="mic_email"):
                st.session_state.v_email = voice_input()
                st.rerun()

        col5, col6 = st.columns([4, 1])
        with col5:
            pw = st.text_input("Password", type="password", value=pw)
        with col6:
            if st.form_submit_button("🎙️", key="mic_pw"):
                st.session_state.v_pw = voice_input()
                st.rerun()

        col7, col8 = st.columns([4, 1])
        with col7:
            cpw = st.text_input("Confirm Password", type="password", value=cpw)
        with col8:
            if st.form_submit_button("🎙️", key="mic_cpw"):
                st.session_state.v_cpw = voice_input()
                st.rerun()

        btn = st.form_submit_button("Sign Up")

    if btn:
        if not name or not email or not pw or not cpw:
            st.error("All fields required.")
        elif pw != cpw:
            st.error("Passwords do not match.")
        else:
            # Replace with your user creation logic
            ok, err = create_user(name, email, pw)
            if ok:
                st.success("✅ Account created! Please login.")
                st.session_state.view = "login"
                st.rerun()
            else:
                st.error(err)

    if st.button("Already have an account? Login"):
        st.session_state.view = "login"
        st.rerun()
         

# ---------------------- LOGIN PAGE ----------------------

import streamlit as st

def login_page():
    st.subheader("Login")

    # keep values on rerun
    username = st.session_state.get("l_username", "")
    pw = st.session_state.get("l_pw", "")

    with st.form("login"):

        col1, col2 = st.columns([4, 1])
        with col1:
            username = st.text_input("Username", value=username)
        with col2:
            if st.form_submit_button("🎙️", key="mic_login_username"):
                st.session_state.l_username = voice_input()
                st.rerun()

        col3, col4 = st.columns([4, 1])
        with col3:
            pw = st.text_input("Password", type="password", value=pw)
        with col4:
            if st.form_submit_button("🎙️", key="mic_login_pw"):
                st.session_state.l_pw = voice_input()
                st.rerun()

        btn = st.form_submit_button("Login")

    if btn:
        if not username or not pw:
            st.error("Username and Password required.")
        else:
            # change this function to your username lookup
            user = get_user_by_username(username)
            if not user:
                st.error("User not found.")
            elif not verify_password(pw, user["password_hash"]):
                st.error("Wrong password.")
            else:
                st.session_state.user = user
                st.session_state.view = "home"
                st.rerun()

    if st.button("New user? Sign Up"):
        st.session_state.view = "signup"
        st.rerun()



# ---------------------- HOME PAGE (Protected) ----------------------
def home_page():
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from PIL import Image
        import tempfile
        from gtts import gTTS
        import os
        import base64 as b64mod
        import io
        import numpy as np
        import cv2
        from ultralytics import YOLO

        api_key = "AIzaSyBU2grB_Txh2eloFGbwqqs4L3CYJ3Jwa78"
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY", "")  # works on Streamlit Cloud/local secrets
        except Exception:
            api_key = "AIzaSyBU2grB_Txh2eloFGbwqqs4L3CYJ3Jwa78"

        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY", "")

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            st.warning("GOOGLE_API_KEY not found. Set it in Streamlit secrets or environment variable.")

        # ------------------- LLM MODEL -------------------
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        SCENE_PROMPT = """You are an advanced AI model designed to assist visually impaired individuals.
        When provided with an image, describe the scene in detail, including:
        - Objects present in the image
        - Colors and visual characteristics
        - Actions or interactions
        - Spatial relationships
        - Any visible text or labels

        Important:
        - Write a concise, clear description between 50 and 250 words.
        - Use simple sentences that are easy to understand when spoken aloud.
        - Do NOT exceed 150 words.
        """

        # ------------------- HELPERS -------------------
        def encode(image: Image.Image) -> str:
            """Convert PIL image to base64 PNG string."""
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return b64mod.b64encode(buffer.getvalue()).decode("utf-8")


        def chain(prompt: str):
            template = ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", [
                    {"type": "text", "text": "{prompt}"},
                    {"type": "image_url", "image_url": "{base64}"}
                ])
            ])

            return (
                RunnablePassthrough.assign(
                    base64=lambda x: f"data:image/png;base64,{x['base64']}"
                )
                | template
                | model
                | StrOutputParser()
            )


        def speak(text: str) -> bytes:
            temp = tempfile.mktemp(suffix=".mp3", prefix="speech_")
            try:
                tts = gTTS(text=text, lang="en")
                tts.save(temp)
                with open(temp, "rb") as audio:
                    data = audio.read()
                return data
            finally:
                try:
                    if os.path.exists(temp):
                        os.unlink(temp)
                except Exception:
                    pass


        @st.cache_resource
        def load_yolo(weights: str):
            """Load YOLO model once and cache it (per weights)."""
            return YOLO(weights)


        def detect_objects(
            image: Image.Image,
            weights: str = "yolov8x-oiv7.pt",
            conf_thres: float = 0.25,
            iou_thres: float = 0.7,
            imgsz: int = 640,
            topk_summary: int = 15,
        ):
            """
            Run YOLO on a single image and return:
            - annotated PIL image
            - summary string (ALL classes)
            - person_count (kept for compatibility; not used)
            - vehicle_count (kept for compatibility; not used)
            - class_counts (dict: label -> count) for ALL detected classes
            - detections_list (list of dicts: label, conf, x1,y1,x2,y2)
            """
            model = load_yolo(weights)

            img_rgb = np.array(image)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            results = model.predict(img_rgb, conf=conf_thres, iou=iou_thres, imgsz=imgsz, verbose=False)[0]
            names = results.names

            class_counts = {}
            detections_list = []

            # (kept only to preserve your existing return signature)
            person_count = 0
            vehicle_count = 0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if label == "person":
                    person_count += 1

                class_counts[label] = class_counts.get(label, 0) + 1
                detections_list.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                # Draw ALL detections the same way
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                caption = f"{label} {conf:.2f}"
                cv2.putText(
                    img_bgr, caption, (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )

            annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            annotated_img = Image.fromarray(annotated_rgb)

            total_objects = sum(class_counts.values())
            unique_classes = len(class_counts)

            # Summary = top classes
            top_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:topk_summary]
            top_text = ", ".join([f"{k}: {v}" for k, v in top_items])
            summary = f"Detected {total_objects} object(s) across {unique_classes} class(es). Top: {top_text}" if top_items else "No objects detected."

            return annotated_img, summary, person_count, vehicle_count, class_counts, detections_list



        def get_video_frame_at(path: str, t_sec: float):
            """Extract a single frame at time t_sec from video (for 'pause and detect' style analysis)."""
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None, 0.0, 0, 0

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

            target_frame = int(t_sec * fps)
            target_frame = max(0, min(target_frame, max(0, total_frames - 1)))

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None, duration, total_frames, fps

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb), duration, total_frames, fps


        def process_video_file(
            path: str,
            weights: str,
            conf_thres: float = 0.25,
            iou_thres: float = 0.7,
            imgsz: int = 640,
            frame_skip: int = 1,
            max_frames_to_show: int = 8,
            export_annotated_video: bool = True,
        ):
            """
            Process a video file:
            - Run YOLO on every 'frame_skip'-th frame (set 1 = full frames)
            - Collect sample annotated frames
            - Track max persons/vehicles seen in a single frame
            - Optionally export full annotated video (all frames)
            - Return (sample_frames, summary, first_frame_pil, annotated_video_path_or_none, total_class_counts)
            """
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return [], "Error: Could not open video.", None, None, {}

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            sampled_frames = []
            first_frame_pil = None
            max_persons = 0
            max_vehicles = 0
            idx = 0

            total_class_counts = {}

            out_path = None
            writer = None
            if export_annotated_video and width > 0 and height > 0:
                out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_path = out_tmp.name
                out_tmp.close()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            prog = st.progress(0)
            status = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save first frame (original) for LLM
                if idx == 0:
                    first_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    first_frame_pil = Image.fromarray(first_rgb)

                do_detect = (idx % max(1, frame_skip) == 0)

                if do_detect:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)

                    annotated, _, p_cnt, v_cnt, class_counts, _ = detect_objects(
                        pil_img,
                        weights=weights,
                        conf_thres=conf_thres,
                        iou_thres=iou_thres,
                        imgsz=imgsz
                    )

                    max_persons = max(max_persons, p_cnt)
                    max_vehicles = max(max_vehicles, v_cnt)

                    for k, v in class_counts.items():
                        total_class_counts[k] = total_class_counts.get(k, 0) + v

                    # Write annotated frame to output video
                    if writer is not None:
                        ann_rgb = np.array(annotated)
                        ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
                        writer.write(ann_bgr)

                    if len(sampled_frames) < max_frames_to_show:
                        sampled_frames.append(annotated)

                else:
                    # If not detecting on this frame, still write original frame (to keep video length correct)
                    if writer is not None:
                        writer.write(frame)

                idx += 1

                if total_frames > 0 and (idx % 10 == 0):
                    prog.progress(min(1.0, idx / total_frames))
                    status.write(f"Processing frame {idx}/{total_frames}...")

            cap.release()
            if writer is not None:
                writer.release()

            prog.progress(1.0)
            status.write("Done.")

            summary = (
                # f"Processed video (frame_skip={frame_skip}). "
                # f"Max in a single analyzed frame: {max_persons} person(s), {max_vehicles} vehicle(s). "
                # f"Total detections (summed across analyzed frames) for top classes: "
                " "
            )

            # Show top 12 by frequency
            top_items = sorted(total_class_counts.items(), key=lambda x: x[1], reverse=True)[:12]
            summary += ", ".join([f"{k}: {v}" for k, v in top_items]) if top_items else "None."

            return sampled_frames, summary, first_frame_pil, out_path, total_class_counts

        import time

        def _video_job_cleanup():
            job = st.session_state.get("video_job", {})
            try:
                if job.get("cap") is not None:
                    job["cap"].release()
            except Exception:
                pass
            try:
                if job.get("writer") is not None:
                    job["writer"].release()
            except Exception:
                pass
            job["cap"] = None
            job["writer"] = None
            st.session_state["video_job"] = job


        def video_job_start(video_path: str, weights: str, conf_thres: float, iou_thres: float,
                            imgsz: int, frame_skip: int, max_frames_to_show: int, export_video: bool):
            # reset old job
            st.session_state["video_job"] = {
                "processing": True,
                "cancel": False,
                "done": False,
                "idx": 0,
                "weights": weights,
                "conf_thres": conf_thres,
                "iou_thres": iou_thres,
                "imgsz": imgsz,
                "frame_skip": max(1, int(frame_skip)),
                "max_frames_to_show": int(max_frames_to_show),
                "sampled_frames": [],
                "total_class_counts": {},
                "out_path": None,
                "cap": None,
                "writer": None,
                "fps": 0.0,
                "total_frames": 0,
            }

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.session_state["video_job"]["processing"] = False
                st.session_state["video_job"]["done"] = True
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            writer = None
            out_path = None
            if export_video and width > 0 and height > 0:
                out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_path = out_tmp.name
                out_tmp.close()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            job = st.session_state["video_job"]
            job["cap"] = cap
            job["writer"] = writer
            job["out_path"] = out_path
            job["fps"] = float(fps)
            job["total_frames"] = int(total_frames)
            st.session_state["video_job"] = job


        def video_job_cancel():
            job = st.session_state.get("video_job", {})
            job["cancel"] = True
            job["processing"] = False
            st.session_state["video_job"] = job
            _video_job_cleanup()


        def video_job_step(chunk_frames: int = 20):
            job = st.session_state.get("video_job", {})
            if not job or not job.get("processing") or job.get("done"):
                return

            cap = job.get("cap", None)
            if cap is None:
                job["processing"] = False
                job["done"] = True
                st.session_state["video_job"] = job
                return

            for _ in range(int(chunk_frames)):
                if job.get("cancel"):
                    break

                ret, frame = cap.read()
                if not ret:
                    job["done"] = True
                    job["processing"] = False
                    break

                idx = job["idx"]
                do_detect = (idx % job["frame_skip"] == 0)

                if do_detect:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)

                    annotated, _, _, _, class_counts, _ = detect_objects(
                        pil_img,
                        weights=job["weights"],
                        conf_thres=job["conf_thres"],
                        iou_thres=job["iou_thres"],
                        imgsz=job["imgsz"],
                    )

                    # accumulate class counts
                    total = job["total_class_counts"]
                    for k, v in class_counts.items():
                        total[k] = total.get(k, 0) + v
                    job["total_class_counts"] = total

                    # keep some sample frames
                    if len(job["sampled_frames"]) < job["max_frames_to_show"]:
                        job["sampled_frames"].append(annotated)

                    # write annotated frame
                    if job.get("writer") is not None:
                        ann_rgb = np.array(annotated)
                        ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
                        job["writer"].write(ann_bgr)
                else:
                    # keep timing correct by writing original frame when skipping detection
                    if job.get("writer") is not None:
                        job["writer"].write(frame)

                job["idx"] += 1

            # finish / cleanup
            if job.get("done") or job.get("cancel"):
                job["processing"] = False
                st.session_state["video_job"] = job
                _video_job_cleanup()
            else:
                st.session_state["video_job"] = job







        # ------------------- STREAMLIT UI -------------------
        st.title("AI Powered Assistance for Visually Impaired Individuals")

        # ---- YOLO model chooser (best overall objects = OpenImagesV7 weights: 601 classes) ----
        # Ultralytics supports OpenImagesV7 pretrained YOLOv8 by using "-oiv7" weights. :contentReference[oaicite:1]{index=1}
        st.sidebar.header("YOLO Settings")

        weights = st.sidebar.selectbox(
            "Choose YOLO weights",
            options=[
                "yolov8x-oiv7.pt",  # 601 classes (best overall object coverage)
                "yolov8l-oiv7.pt",
                "yolov8m-oiv7.pt",
                "yolov8s-oiv7.pt",
                "yolov8n-oiv7.pt",
                # You can also choose YOLO11 COCO (80 classes), but fewer categories:
                "yolo11x.pt",
                "yolo11l.pt",
                "yolo11m.pt",
                "yolo11s.pt",
                "yolo11n.pt",
            ],
            index=0
        )

        conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.01)
        iou_thres = st.sidebar.slider("IoU threshold", 0.10, 0.95, 0.70, 0.01)
        imgsz = st.sidebar.selectbox("Image size (imgsz)", [320, 480, 640, 960, 1280], index=0)
        if st.sidebar.button("Log Out"):
            st.session_state.user = None
            st.session_state.view = "login"
            st.rerun()

        mode = st.radio("Select input type:", ["Image", "Video"], horizontal=True)

        # ------------------- IMAGE MODE -------------------
        if mode == "Image":
            img = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

            if img:
                image = Image.open(img).convert("RGB")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)

                annotated_img, summary, person_cnt, vehicle_cnt, class_counts, det_list = detect_objects(
                    image,
                    weights=weights,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    imgsz=imgsz
                )

                with col2:
                    st.subheader("YOLO Detection (All Objects)")
                    st.image(annotated_img, caption=summary, use_container_width=True)
                    st.write(f"**Persons:** {person_cnt}")
                    st.write(f"**Vehicles:** {vehicle_cnt}")

                    st.write("**All detected classes (counts):**")
                    st.json(class_counts)

                    st.write("**Bounding boxes (label + confidence + coords):**")
                    st.dataframe(det_list, use_container_width=True)

                # LLM scene description (short & sweet) + audio
                img_b64 = encode(image)

                st.subheader("Scene Description (LLM)")
                scene_chain = chain(SCENE_PROMPT)
                scene = scene_chain.invoke({
                    "prompt": SCENE_PROMPT,
                    "base64": img_b64
                })
                st.write(scene)

                if st.button("🔊 Listen to Description"):
                    audio = speak(scene)
                    st.audio(audio, format="audio/mp3")

        # ------------------- VIDEO MODE -------------------
        else:
            video_file = st.file_uploader(
                "Upload a video", type=["mp4", "mov", "avi", "mkv"]
            )

            if video_file is not None:
                # Save to temp file
                video_bytes = video_file.read()
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_bytes)
                tfile.close()

                st.subheader("Original Video")
                st.video(video_bytes)

                # ---- "Pause video and detect" (manual frame selection) ----
                st.subheader("Pause & Detect (Select a time → analyze that frame)")
                key_frame_pil, duration, total_frames, fps = get_video_frame_at(tfile.name, 0.0)

                if duration > 0:
                    t_sec = st.slider("Select time (seconds)", 0.0, float(duration), 0.0, 0.05)
                else:
                    t_sec = st.slider("Select time (seconds)", 0.0, 60.0, 0.0, 0.05)

                if st.button("Analyze selected frame"):
                    key_frame_pil, duration2, total_frames2, fps2 = get_video_frame_at(tfile.name, t_sec)
                    if key_frame_pil is None:
                        st.error("Could not read that frame from the video.")
                    else:
                        st.write(f"Frame time: **{t_sec:.2f}s** | FPS: **{fps2:.2f}** | Total frames: **{total_frames2}**")
                        ann, summ, p_cnt, v_cnt, counts, det_list = detect_objects(
                            key_frame_pil,
                            weights=weights,
                            conf_thres=conf_thres,
                            iou_thres=iou_thres,
                            imgsz=imgsz
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(key_frame_pil, caption="Selected Frame (Original)", use_container_width=True)
                        with c2:
                            st.image(ann, caption=summ, use_container_width=True)
                        st.write("**Bounding boxes (label + confidence + coords):**")
                        st.dataframe(det_list, use_container_width=True)



                        # Frame Summary and Detected Classes
                        st.markdown("### Frame Summary")
                        total_objects = sum(counts.values())
                        unique_classes = len(counts)
                        st.write(f"Detected **{total_objects}** object(s) from **{unique_classes}** class(es).")

                        st.markdown("### Detected Classes in This Frame")
                        st.json(counts)






                st.divider()

                # ---- Full video detection settings ----
                st.subheader("Full Video Object Detection (All Frames)")
                st.caption("Set frame_skip=1 to detect on every frame (slow but full coverage).")

                # frame_skip = st.number_input("frame_skip (1 = every frame)", min_value=1, max_value=120, value=1, step=1)
                max_frames_to_show = st.number_input("Max sample frames to display", min_value=1, max_value=50, value=4, step=1)
                # export_video = st.checkbox("Export annotated video (all frames)", value=True)



                frame_skip = st.number_input(
                "frame_skip (1 = every frame)",
                min_value=1, max_value=120,
                value=4, step=1
        )
                # YOLO on video (FULL FRAMES if frame_skip=1)
                with st.spinner("Running YOLO object detection on video..."):
                    frames, video_summary, first_frame, annotated_video_path, total_class_counts = process_video_file(
                        tfile.name,
                        weights=weights,
                        conf_thres=conf_thres,
                        iou_thres=iou_thres,
                        imgsz=imgsz,
                        frame_skip=int(frame_skip),
                        max_frames_to_show=int(max_frames_to_show),
                        # export_annotated_video=export_video
                    )

                if frames:
                    st.subheader("Sample Annotated Frames")
                    st.image(frames, use_container_width=True)

                # LLM description from key frame + audio
                if first_frame is not None:
                    img_b64 = encode(first_frame)

                    st.subheader("Scene Description from Video (Key Frame)")
                    scene_chain = chain(SCENE_PROMPT)
                    scene = scene_chain.invoke({
                        "prompt": SCENE_PROMPT,
                        "base64": img_b64
                    })
                    st.write(scene)

                    if st.button("🔊 Listen to Description (Video)", key="listen_video_desc"):
                        audio = speak(scene)
                        st.audio(audio, format="audio/mp3")

# ---------------------- MAIN ROUTER ----------------------
def main():
    if "view" not in st.session_state:
        st.session_state.view = "login"

    if "user" not in st.session_state:
        st.session_state.user = None

    navbar(st.session_state.view)

    if st.session_state.view == "signup":
        signup_page()
    elif st.session_state.view == "home":
        if st.session_state.user:
            home_page()
        else:
            st.session_state.view = "login"
            st.rerun()
    else:
        login_page()


if __name__ == "__main__":
    main()
