"""Facial Recognition Engine for the Streamlit Application.
Uses OpenCV YuNet for face detection and OpenCV SFace for face recognition.
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FEATURES_CSV = os.path.join(ROOT_DIR, "features", "image_features.csv")

YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

DEFAULT_THRESHOLD = 0.363  # OpenCV recommended cosine similarity threshold for SFace

# Initialize models
_detector = None
_recognizer = None
_member_reference_embeddings = {}


def get_detector():
    global _detector
    if _detector is None:
        if not os.path.exists(YUNET_PATH):
            raise FileNotFoundError(f"YuNet model not found at {YUNET_PATH}")
        _detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000)
    return _detector


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        if not os.path.exists(SFACE_PATH):
            raise FileNotFoundError(f"SFace model not found at {SFACE_PATH}")
        _recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    return _recognizer


def get_reference_embeddings():
    """Load reference centroid embeddings from features/image_features.csv for original photos."""
    global _member_reference_embeddings
    if not _member_reference_embeddings:
        if not os.path.exists(FEATURES_CSV):
            raise FileNotFoundError(f"Image features CSV not found at {FEATURES_CSV}")
        df = pd.read_csv(FEATURES_CSV)
        emb_cols = [f"emb_{i:03d}" for i in range(128)]
        orig_df = df[df["augmentation"] == "original"]
        for member, group in orig_df.groupby("member"):
            # Compute centroid of original embeddings and normalize
            mean_emb = group[emb_cols].values.mean(axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            _member_reference_embeddings[member] = mean_emb
    return _member_reference_embeddings


def to_bgr(image_input):
    """Convert various image inputs (path, PIL, or numpy array) to OpenCV BGR format."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image from {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        elif image_input.shape[2] == 3:
            # Assume already BGR or RGB; if values look like RGB from PIL, user might pass RGB
            return image_input
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def verify_face(image_input, threshold=DEFAULT_THRESHOLD):
    """
    Run facial recognition on an input image.

    Returns:
        dict containing:
            - 'approved': bool (True if matched authorized member >= threshold)
            - 'member': str or None (best matched member name)
            - 'similarity': float (cosine similarity score, 0 to 1)
            - 'similarities': dict of {member: score}
            - 'threshold': float (the decision threshold used)
            - 'annotated_image': PIL.Image with bounding box and landmarks
            - 'aligned_face': PIL.Image of 112x112 face crop (if detected)
            - 'denial_reason': str (explanation if access is denied)
    """
    detector = get_detector()
    recognizer = get_recognizer()
    references = get_reference_embeddings()

    img = to_bgr(image_input)
    h, w = img.shape[:2]

    # Resize if image is exceptionally large to keep detection robust & fast
    max_side = 1000
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    detector.setInputSize((w, h))
    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {
            "approved": False,
            "member": None,
            "similarity": 0.0,
            "similarities": {m: 0.0 for m in references},
            "threshold": threshold,
            "annotated_image": Image.fromarray(rgb_img),
            "aligned_face": None,
            "denial_reason": "No face detected in the image. Please ensure your face is well-lit and centered.",
        }

    # Best detected face
    face = faces[np.argmax(faces[:, -1])]
    bbox = face[:4].astype(int)
    landmarks = face[4:14].reshape((5, 2)).astype(int)
    face_score = float(face[-1])

    # Align and crop face to 112x112
    aligned = recognizer.alignCrop(img, face)
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    # Extract 128-d feature embedding and normalize
    emb = recognizer.feature(aligned).flatten().astype(float)
    emb = emb / np.linalg.norm(emb)

    # Compute cosine similarities
    similarities = {}
    for member, ref_emb in references.items():
        sim = float(np.dot(emb, ref_emb))
        similarities[member] = max(0.0, sim)

    best_member = max(similarities, key=similarities.get)
    best_similarity = similarities[best_member]

    # Check threshold
    approved = bool(best_similarity >= threshold)

    # Create annotated visual image
    annotated = img.copy()
    x, y, bw, bh = bbox
    color = (0, 200, 0) if approved else (0, 0, 220)  # Green if pass, Red if fail
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 3)

    # Draw 5 facial landmarks (eyes, nose, mouth corners)
    landmark_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    for pt, c in zip(landmarks, landmark_colors):
        cv2.circle(annotated, (pt[0], pt[1]), 4, c, -1)

    label_text = f"{best_member.capitalize()} ({best_similarity:.1%})" if approved else f"Denied ({best_similarity:.1%})"
    cv2.putText(annotated, label_text, (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    denial_reason = None
    if not approved:
        denial_reason = (
            f"Face recognized as closest to '{best_member.capitalize()}' with similarity {best_similarity:.1%}, "
            f"which is below the required threshold of {threshold:.1%}. Access Denied."
        )

    return {
        "approved": approved,
        "member": best_member,
        "similarity": best_similarity,
        "similarities": similarities,
        "threshold": threshold,
        "annotated_image": Image.fromarray(annotated_rgb),
        "aligned_face": Image.fromarray(aligned_rgb),
        "denial_reason": denial_reason,
    }

