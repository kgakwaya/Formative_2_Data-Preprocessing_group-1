"""Voice Validation Engine for the Streamlit Application.
Uses librosa / soundfile to extract MFCC features and evaluate against voice_model.joblib.
"""
import os
import io
import tempfile
import numpy as np
import librosa
import soundfile as sf
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "verify_voice", "voice_model.joblib")

DEFAULT_THRESHOLD = 0.70  # Standard confidence threshold from formative2 voiceprint task
SR = 16000
N_MFCC = 13

_voice_bundle = None


def get_voice_bundle():
    global _voice_bundle
    if _voice_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Voice model not found at {MODEL_PATH}")
        _voice_bundle = joblib.load(MODEL_PATH)
    return _voice_bundle


def load_audio_signal(audio_input, sr=SR):
    """
    Load an audio input (filepath, bytes, or file-like object) into a mono float array at `sr`.
    """
    if isinstance(audio_input, (bytes, bytearray)):
        with io.BytesIO(audio_input) as bio:
            y, file_sr = sf.read(bio)
    elif hasattr(audio_input, "read"):
        # UploadedFile or buffer
        y, file_sr = sf.read(audio_input)
    elif isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        y, file_sr = sf.read(audio_input)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample to target sample rate if needed
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    # Trim leading and trailing silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Normalize amplitude
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y


def extract_mfcc_features(y, sr=SR):
    """Extract 12 MFCC means (mfcc2_mean to mfcc13_mean)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats = {}
    for i in range(1, N_MFCC):
        feats[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
    return feats


def verify_voice(audio_input, expected_member=None, threshold=DEFAULT_THRESHOLD):
    """
    Run voice validation on an input audio signal.

    Args:
        audio_input: path to audio file, raw bytes, or file buffer.
        expected_member: str (optional, e.g. member authenticated during face recognition).
        threshold: float (minimum confidence score required, default 0.70).

    Returns:
        dict containing:
            - 'approved': bool
            - 'predicted_member': str
            - 'confidence': float (0 to 1)
            - 'probabilities': dict of {class: prob}
            - 'threshold': float
            - 'expected_member': str or None
            - 'identity_match': bool (True if predicted == expected_member)
            - 'denial_reason': str or None
    """
    bundle = get_voice_bundle()
    model = bundle["model"]
    columns = bundle["feature_columns"]

    try:
        y = load_audio_signal(audio_input)
    except Exception as e:
        return {
            "approved": False,
            "predicted_member": "unknown",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in getattr(model, "classes_", ["audric", "ketia", "twariq", "unknown"])},
            "threshold": threshold,
            "expected_member": expected_member,
            "identity_match": False,
            "denial_reason": f"Failed to process audio signal: {str(e)}",
        }

    sr_sample_minimum = int(SR * 0.3)
    if len(y) < sr_sample_minimum:
        return {
            "approved": False,
            "predicted_member": "unknown",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in model.classes_},
            "threshold": threshold,
            "expected_member": expected_member,
            "identity_match": False,
            "denial_reason": "Audio clip is too short (less than 0.3 seconds of detected speech).",
        }

    feats = extract_mfcc_features(y, sr=SR)
    x = np.array([[feats.get(c, 0.0) for c in columns]])

    proba = model.predict_proba(x)[0]
    best_idx = int(np.argmax(proba))
    predicted_member = model.classes_[best_idx]
    confidence = float(proba[best_idx])
    probabilities = {cls: float(p) for cls, p in zip(model.classes_, proba)}

    # Security validation conditions:
    # 1. Must not be 'unknown'
    # 2. Confidence must be >= threshold (70%)
    # 3. If expected_member is specified, the voice must belong to that member!
    is_known = (predicted_member != "unknown")
    cleared_threshold = (confidence >= threshold)
    identity_match = (expected_member is None or predicted_member.lower() == expected_member.lower())

    approved = is_known and cleared_threshold and identity_match

    denial_reason = None
    if not approved:
        if not is_known:
            denial_reason = f"Voice classified as 'Unknown Speaker' (confidence {confidence:.1%}). Access Denied."
        elif not cleared_threshold:
            denial_reason = (
                f"Voice identified as '{predicted_member.capitalize()}', but confidence ({confidence:.1%}) "
                f"did not meet the required {threshold:.1%} threshold. Access Denied."
            )
        elif not identity_match:
            denial_reason = (
                f"Biometric Identity Mismatch! Facial recognition authenticated '{expected_member.capitalize()}', "
                f"but voice biometrics matched '{predicted_member.capitalize()}' ({confidence:.1%}). Access Denied."
            )

    return {
        "approved": approved,
        "predicted_member": predicted_member,
        "confidence": confidence,
        "probabilities": probabilities,
        "threshold": threshold,
        "expected_member": expected_member,
        "identity_match": identity_match,
        "denial_reason": denial_reason,
    }

