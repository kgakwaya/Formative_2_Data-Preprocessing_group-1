"""Streamlit Multimodal Biometric Verification & Product Recommendation App.

Start -> Facial Recognition (fail: Access Denied)
      -> Product Recommendation Model
      -> Voice Validation (fail: Access Denied)
      -> Display Predicted Product
"""
import html
import io
import time

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.face_engine import verify_face, DEFAULT_THRESHOLD as FACE_THRESHOLD
from src.voice_engine import verify_voice, DEFAULT_THRESHOLD as VOICE_THRESHOLD
from src.recommender_engine import recommend_product, get_default_profile, CATEGORY_METADATA

MAX_ATTEMPTS = 3
TRANSITION_PAUSE = 0.7

FACE_SAMPLES = [
    ("Audric", "images/raw/audric_neutral.jpg"),
    ("Ketia", "images/raw/ketia_neutral.jpg"),
    ("Twariq", "images/raw/twariq_neutral.jpg"),
    ("Stranger", "images/raw/stranger_sample.jpg"),
    ("Audric (smiling)", "images/raw/audric_smiling.jpg"),
    ("Ketia (smiling)", "images/raw/ketia_smiling.jpg"),
    ("Twariq (surprised)", "images/raw/twariq_surprised.jpg"),
]

VOICE_SAMPLES = {
    "Audric - Approve": "audio/audric_approve.flac",
    "Audric - Confirm": "audio/audric_confirm.flac",
    "Ketia - Approve": "audio/ketia_approve.flac",
    "Ketia - Confirm": "audio/ketia_confirm.flac",
    "Twariq - Approve": "audio/twariq_approve.flac",
    "Twariq - Confirm": "audio/twariq_confirm.flac",
    "Unknown - Approve": "audio/unknown1_approve.flac",
    "Unknown - Confirm": "audio/unknown1_confirm.flac",
}

STEPS = ["Face ID", "Product model", "Voice ID", "Your product"]
STEP_INDEX = {"start": -1, "face": 0, "recommend": 1, "voice": 2, "result": 4}

st.set_page_config(page_title="Biometric Product Recommender", layout="centered")

st.markdown("""
<style>
.stepper { display: flex; gap: 6px; margin: 0.25rem 0 1.5rem; }
.step { flex: 1; text-align: center; font-size: 0.82rem; font-weight: 600; }
.step .bar { height: 6px; border-radius: 99px; margin-bottom: 6px; background: rgba(148,163,184,0.3); }
.step.pending { opacity: 0.55; }
.step.active .bar { background: #3B82F6; }
.step.done .bar { background: #10B981; }
.step.failed .bar { background: #EF4444; }
.step.failed { color: #EF4444; }

.card { padding: 1.25rem 1.5rem; border-radius: 14px; margin: 0.75rem 0 1.25rem; color: #1E293B; }
.card h2, .card h3 { margin: 0 0 0.35rem; color: inherit; padding: 0; }
.card p { margin: 0.2rem 0; }
.card-denied { background: #FEF2F2; border: 1px solid #FCA5A5; color: #7F1D1D; text-align: center; }
.card-locked { background: #FFFBEB; border: 1px solid #FCD34D; color: #78350F; }
.card-success { background: #ECFDF5; border: 1px solid #6EE7B7; color: #064E3B; }
.card-product { background: linear-gradient(135deg, #EEF2FF, #F5F3FF); border: 1px solid #C7D2FE; }
.muted { opacity: 0.75; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

ss = st.session_state
DEFAULTS = {
    "step": "start",
    "denied_stage": None,
    "attempts": {"face": 0, "voice": 0},
    "face_input": None,
    "face_result": None,
    "voice_input": None,
    "voice_result": None,
    "profile": None,
    "rec": None,
    "face_threshold": FACE_THRESHOLD,
    "voice_threshold": VOICE_THRESHOLD,
}
for k, v in DEFAULTS.items():
    if k not in ss:
        ss[k] = v.copy() if isinstance(v, dict) else v


def reset_journey():
    for k, v in DEFAULTS.items():
        if k not in ("face_threshold", "voice_threshold"):
            ss[k] = v.copy() if isinstance(v, dict) else v
    for k in ("cam", "face_upload", "mic", "voice_upload", "voice_pick"):
        ss.pop(k, None)


def go(step):
    ss.step = step
    st.rerun()


def member_name():
    return ss.face_result["member"].capitalize()


@st.cache_data
def thumbnail(path):
    return ImageOps.fit(Image.open(path).convert("RGB"), (220, 220))


def bar_chart(data, value, label, highlight, threshold=None):
    df = pd.DataFrame([{label: k.capitalize(), value: v} for k, v in data.items()])
    bars = alt.Chart(df).mark_bar(cornerRadius=4).encode(
        x=alt.X(f"{value}:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y(f"{label}:N", sort="-x", title=None),
        color=alt.condition(highlight, alt.value("#10B981"), alt.value("#94A3B8")),
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=".1%")],
    )
    if threshold is not None:
        rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="#EF4444", strokeDash=[4, 4]).encode(x="t:Q")
        bars = bars + rule
    st.altair_chart(bars.properties(height=34 * len(df) + 20), width="stretch")


def face_details(res):
    c1, c2 = st.columns(2)
    with c1:
        st.image(res["annotated_image"], caption="Detected face & landmarks", width="stretch")
    with c2:
        st.metric("Closest match", (res["member"] or "None").capitalize(), f"{res['similarity']:.1%} similarity")
        st.caption(f"Required: {res['threshold']:.1%} (red line)")
        bar_chart(res["similarities"], "Similarity", "Member",
                  alt.datum.Similarity >= res["threshold"], res["threshold"])


def voice_details(res):
    expected = (res["expected_member"] or "").capitalize()
    st.metric("Predicted speaker", res["predicted_member"].capitalize(), f"{res['confidence']:.1%} confidence")
    st.caption(f"Expected speaker: {expected}, required confidence {res['threshold']:.0%} (red line)")
    bar_chart(res["probabilities"], "Probability", "Speaker",
              alt.datum.Speaker == expected, res["threshold"])


def render_stepper():
    idx = STEP_INDEX.get(ss.step)
    if ss.step == "denied":
        idx = 0 if ss.denied_stage == "face" else 2
    parts = []
    for i, label in enumerate(STEPS):
        if ss.step == "denied" and i == idx:
            state = "failed"
        elif i < idx:
            state = "done"
        elif i == idx:
            state = "active"
        else:
            state = "pending"
        parts.append(f'<div class="step {state}"><div class="bar"></div>{label}</div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def method_picker(key, options):
    choice = st.segmented_control("Input method", options, default=options[0], key=key, label_visibility="collapsed")
    return choice or options[0]


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("System control")
    st.caption("Registered members: Audric, Ketia, Twariq")
    with st.expander("Security thresholds"):
        st.slider("Face similarity threshold", 0.20, 0.60, step=0.01, key="face_threshold",
                  help="OpenCV's recommended SFace same-person threshold is 0.363.")
        st.slider("Voice confidence threshold", 0.50, 0.95, step=0.05, key="voice_threshold",
                  help="Default 0.70 for the voiceprint model.")
    st.button("Restart journey", width="stretch", on_click=reset_journey)


st.title("Biometric Product Recommender")
render_stepper()


# ------------------------------------------------------------------------------
# Start
# ------------------------------------------------------------------------------
if ss.step == "start":
    st.markdown("#### Get a personalised product recommendation in three quick steps")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Face check**  \nWe confirm you're a registered member.")
    c2.markdown("**2. Prediction**  \nOur model picks a product for you.")
    c3.markdown("**3. Voice check**  \nSay a phrase to unlock your result.")
    st.write("")
    st.caption("Failing either biometric check denies access. You get "
               f"{MAX_ATTEMPTS} attempts per check.")
    if st.button("Get started", type="primary", width="stretch"):
        go("face")


# ------------------------------------------------------------------------------
# Step 1: Facial recognition
# ------------------------------------------------------------------------------
elif ss.step == "face":
    st.subheader("Step 1: Verify your face")
    st.markdown('<p class="muted">Choose a sample photo, take a selfie, or upload a picture.</p>',
                unsafe_allow_html=True)

    def set_face(src, label):
        ss.face_input = {"src": src, "label": label}

    method = method_picker("face_method", ["Sample photos", "Camera", "Upload"])

    if method == "Sample photos":
        selected = ss.face_input["src"] if ss.face_input else None
        cols = st.columns(4)
        for i, (label, path) in enumerate(FACE_SAMPLES):
            with cols[i % 4]:
                st.image(thumbnail(path), width="stretch")
                is_sel = selected == path
                st.button(label, key=f"pick_{path}", width="stretch",
                          type="primary" if is_sel else "secondary", on_click=set_face, args=(path, label))
    elif method == "Camera":
        cam = st.camera_input("Look at the camera and take a photo", key="cam")
        if cam is not None:
            set_face(cam.getvalue(), "Webcam photo")
    else:
        up = st.file_uploader("Upload a clear, front-facing photo", type=["jpg", "jpeg", "png"], key="face_upload")
        if up is not None:
            set_face(up.getvalue(), up.name)

    st.divider()
    if ss.face_input is None:
        st.button("Verify my face", type="primary", width="stretch", disabled=True)
        st.caption("Select a photo above to continue.")
    else:
        src = ss.face_input["src"]
        img = Image.open(src if isinstance(src, str) else io.BytesIO(src)).convert("RGB")
        pc1, pc2 = st.columns([1, 3], vertical_alignment="center")
        pc1.image(ImageOps.fit(img, (160, 160)))
        pc2.markdown(f"**Selected:** {html.escape(ss.face_input['label'])}")
        if pc2.button("Verify my face", type="primary", width="stretch"):
            with st.status("Verifying your face...", expanded=True) as status:
                st.write("Detecting face and extracting embedding...")
                res = verify_face(img, threshold=ss.face_threshold)
                st.write("Comparing against registered members...")
                ss.face_result = res
                if res["approved"]:
                    status.update(label=f"Face verified. Welcome, {member_name()}.", state="complete")
                else:
                    status.update(label="Face not recognised", state="error")
            time.sleep(TRANSITION_PAUSE)
            if res["approved"]:
                go("recommend")
            ss.attempts["face"] += 1
            ss.denied_stage = "face"
            go("denied")


# ------------------------------------------------------------------------------
# Step 2: Product recommendation model (runs automatically)
# ------------------------------------------------------------------------------
elif ss.step == "recommend":
    st.subheader(f"Welcome back, {member_name()}")
    with st.status("Running product recommendation model...", expanded=True) as status:
        st.write("Loading your customer profile...")
        ss.profile = get_default_profile(ss.face_result["member"])
        st.write("Scoring every product category...")
        ss.rec = recommend_product(ss.profile)
        time.sleep(TRANSITION_PAUSE)
        status.update(label="Recommendation ready. Locked until voice check.", state="complete")
    time.sleep(TRANSITION_PAUSE)
    go("voice")


# ------------------------------------------------------------------------------
# Step 3: Voice validation
# ------------------------------------------------------------------------------
elif ss.step == "voice":
    name = member_name()
    st.markdown(f"""
    <div class="card card-locked">
        <p class="muted">Face verified as {html.escape(name)}. Prediction ready.</p>
        <h3 style="margin-top:0.4rem">Your recommendation is ready</h3>
        <p>Confirm it's really you with your voice to unlock it.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Step 3: Verify your voice")
    st.markdown('Say **"Approve"** or **"Confirm"** clearly.')

    def set_voice(src, label):
        ss.voice_input = {"src": src, "label": label}

    def on_pick():
        pick = ss.get("voice_pick")
        ss.voice_input = {"src": VOICE_SAMPLES[pick], "label": pick} if pick else None

    method = method_picker("voice_method", ["Sample clips", "Record", "Upload"])

    if method == "Sample clips":
        st.pills("Sample clips", list(VOICE_SAMPLES), key="voice_pick", on_change=on_pick,
                 label_visibility="collapsed")
    elif method == "Record":
        rec = st.audio_input("Press the mic and speak", key="mic")
        if rec is not None:
            set_voice(rec.getvalue(), "Live recording")
    else:
        up = st.file_uploader("Upload a voice clip", type=["flac", "wav", "mp3"], key="voice_upload")
        if up is not None:
            set_voice(up.getvalue(), up.name)

    st.divider()
    if ss.voice_input is None:
        st.button("Verify my voice", type="primary", width="stretch", disabled=True)
        st.caption("Choose or record a clip above to continue.")
    else:
        st.markdown(f"**Selected:** {html.escape(ss.voice_input['label'])}")
        st.audio(ss.voice_input["src"])
        if st.button("Verify my voice", type="primary", width="stretch"):
            with st.status("Analysing your voiceprint...", expanded=True) as status:
                st.write("Extracting MFCC features...")
                res = verify_voice(ss.voice_input["src"], expected_member=ss.face_result["member"],
                                   threshold=ss.voice_threshold)
                st.write(f"Matching against {name}'s voiceprint...")
                ss.voice_result = res
                if res["approved"]:
                    status.update(label="Voice verified. Unlocking your product.", state="complete")
                else:
                    status.update(label="Voice not verified", state="error")
            time.sleep(TRANSITION_PAUSE)
            if res["approved"]:
                go("result")
            ss.attempts["voice"] += 1
            ss.denied_stage = "voice"
            go("denied")


# ------------------------------------------------------------------------------
# Access denied (either gate)
# ------------------------------------------------------------------------------
elif ss.step == "denied":
    stage = ss.denied_stage
    res = ss.face_result if stage == "face" else ss.voice_result
    left = MAX_ATTEMPTS - ss.attempts[stage]
    what = "Face" if stage == "face" else "Voice"
    attempts_msg = (f"{left} attempt{'s' if left != 1 else ''} remaining" if left > 0
                    else "Too many failed attempts. This session is locked.")

    st.markdown(f"""
    <div class="card card-denied">
        <h2>Access denied</h2>
        <p><strong>{what} verification failed.</strong></p>
        <p>{html.escape(res["denial_reason"] or "")}</p>
        <p style="margin-top:0.6rem; font-weight:600">{attempts_msg}</p>
    </div>
    """, unsafe_allow_html=True)

    def retry():
        if stage == "face":
            ss.face_input = ss.face_result = None
            ss.pop("cam", None)
            ss.pop("face_upload", None)
        else:
            ss.voice_input = ss.voice_result = None
            for k in ("mic", "voice_upload", "voice_pick"):
                ss.pop(k, None)
        ss.step = stage

    b1, b2 = st.columns(2)
    if left > 0:
        b1.button(f"Try {what.lower()} again", type="primary", width="stretch", on_click=retry)
    b2.button("Start over", width="stretch", on_click=reset_journey)

    with st.expander("Why was I denied? See what the model saw"):
        face_details(res) if stage == "face" else voice_details(res)


# ------------------------------------------------------------------------------
# Step 4: Display predicted product
# ------------------------------------------------------------------------------
elif ss.step == "result":
    name = member_name()
    rec, prof = ss.rec, ss.profile
    meta = rec["metadata"]

    st.markdown(f"""
    <div class="card card-success">
        <strong>Identity confirmed, {html.escape(name)}.</strong>
        Face {ss.face_result['similarity']:.0%} match, voice {ss.voice_result['confidence']:.0%} confidence
    </div>
    <div class="card card-product" style="border-left: 8px solid {meta['color']}">
        <div style="color:{meta['color']}; font-weight:700; text-transform:uppercase; letter-spacing:1px; font-size:0.85rem">
            Recommended for you
        </div>
        <div style="font-size:2.2rem; font-weight:800; margin:0.15rem 0">{html.escape(rec['recommended_category'])}</div>
        <div style="font-size:1.05rem; font-weight:500; opacity:0.85">{meta['tagline']}</div>
        <p style="margin-top:0.75rem">{meta['desc']}</p>
        <p style="margin-top:0.75rem; font-weight:700">Model confidence: {rec['confidence']:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    tab_dist, tab_profile, tab_whatif, tab_auth = st.tabs(
        ["All categories", "Your profile", "What-if", "Verification"])

    with tab_dist:
        df = pd.DataFrame([{"Category": c, "Probability": p} for c, p in rec["all_probabilities"].items()])
        chart = alt.Chart(df).mark_bar(cornerRadius=4).encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Category:N", sort="-x", title=None),
            color=alt.Color("Category:N", legend=None, scale=alt.Scale(
                domain=list(CATEGORY_METADATA), range=[m["color"] for m in CATEGORY_METADATA.values()])),
            tooltip=["Category", alt.Tooltip("Probability:Q", format=".1%")],
        ).properties(height=220)
        st.altair_chart(chart, width="stretch")

    with tab_profile:
        p1, p2, p3 = st.columns(3)
        p1.metric("Customer ID", prof.get("customer_id", "-"))
        p2.metric("Platform", prof.get("platform", "-"))
        p3.metric("Sentiment", prof.get("sentiment", "-"))
        p4, p5, p6 = st.columns(3)
        p4.metric("Engagement", f"{prof.get('engagement_mean', 0):.1f}/100")
        p5.metric("Purchase interest", f"{prof.get('interest_mean', 0):.1f}/5")
        p6.metric("Prior purchases", prof.get("prior_purchases", 0))

    with tab_whatif:
        st.caption("Tweak the profile to see how the recommendation changes.")
        platforms = ["LinkedIn", "Instagram", "TikTok", "Facebook", "Twitter"]
        sentiments = ["Positive", "Neutral", "Negative"]
        w1, w2 = st.columns(2)
        sim = prof.copy()
        sim["platform"] = w1.selectbox("Platform", platforms,
                                       index=platforms.index(prof["platform"]) if prof.get("platform") in platforms else 0)
        sim["sentiment"] = w2.selectbox("Sentiment", sentiments,
                                        index=sentiments.index(prof["sentiment"]) if prof.get("sentiment") in sentiments else 0)
        sim["engagement_mean"] = w1.slider("Engagement", 10.0, 100.0, float(prof.get("engagement_mean", 75.0)))
        sim["interest_mean"] = w2.slider("Purchase interest", 1.0, 5.0, float(prof.get("interest_mean", 4.0)))
        sim["prior_purchases"] = w1.slider("Prior purchases", 0, 15, int(prof.get("prior_purchases", 5)))
        sim["prior_avg_amount"] = w2.slider("Prior avg amount ($)", 20.0, 500.0, float(prof.get("prior_avg_amount", 200.0)))
        sim_rec = recommend_product(sim)
        changed = sim_rec["recommended_category"] != rec["recommended_category"]
        st.info(f"Simulated: **{sim_rec['recommended_category']}** "
                f"({sim_rec['confidence']:.1%}){' (changed)' if changed else ''}")

    with tab_auth:
        st.markdown("**Face check**")
        face_details(ss.face_result)
        st.markdown("**Voice check**")
        voice_details(ss.voice_result)

    st.write("")
    st.button("Start a new session", type="primary", width="stretch", on_click=reset_journey)
