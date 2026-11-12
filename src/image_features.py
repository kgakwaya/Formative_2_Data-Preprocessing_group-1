"""Face features for the command-line app.

Same steps as task2_image_processing.ipynb: load the photo, find the face,
align it to 112x112, then compute the SFace embedding, HSV colour histogram,
LBP texture histogram, brightness and contrast (164 numbers in total).
"""
import os
import urllib.request

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")
YUNET = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODELS, "face_recognition_sface_2021dec.onnx")
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"

os.makedirs(MODELS, exist_ok=True)
if not os.path.exists(YUNET):
    urllib.request.urlretrieve(ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx", YUNET)
if not os.path.exists(SFACE):
    urllib.request.urlretrieve(ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx", SFACE)

detector = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.6, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE, "")

feature_columns = ([f"emb_{i:03d}" for i in range(128)]
                   + [f"hsv_{c}_{b}" for c in "hsv" for b in range(8)]
                   + [f"lbp_{b}" for b in range(10)]
                   + ["brightness", "contrast"])


def load_image(path, max_side=800):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def detect_face(img):
    # returns the most confident face: x, y, w, h, 5 landmark points, score (or None)
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None:
        return None
    return faces[np.argmax(faces[:, -1])]


def get_features(face):
    emb = recognizer.feature(face).flatten().astype(float)
    emb = emb / np.linalg.norm(emb)

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    colour = []
    for channel, top in enumerate([180, 256, 256]):
        hist = cv2.calcHist([hsv], [channel], None, [8], [0, top]).flatten()
        colour.extend(hist / hist.sum())

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    texture, _ = np.histogram(lbp, bins=10, range=(0, 10))
    texture = texture / texture.sum()

    tone = [gray.mean() / 255, gray.std() / 255]
    return np.concatenate([emb, colour, texture, tone])


def features_from_path(path):
    """Photo path -> 164 features, or None if no face is found."""
    img = load_image(path)
    face = detect_face(img)
    if face is None:
        return None
    return get_features(recognizer.alignCrop(img, face))
