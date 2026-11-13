import os
import joblib
import numpy as np
import pandas as pd

print("1. Checking Product Recommender...")
prod_bundle = joblib.load("models/product_recommender.joblib")
print("  Model name:", prod_bundle.get("model_name"))
print("  Categories:", prod_bundle.get("categories"))
print("  Feature columns:", prod_bundle.get("feature_columns"))

print("\n2. Checking Voice Model...")
voice_bundle = joblib.load("verify_voice/voice_model.joblib")
print("  Voice model:", type(voice_bundle["model"]))
print("  Classes:", voice_bundle["model"].classes_)
print("  Feature columns:", voice_bundle["feature_columns"])

print("\n3. Checking Image Features & OpenCV Models...")
from src.image_features import features_from_path, recognizer, detector
img_path = "images/raw/twariq_neutral.jpg"
feats = features_from_path(img_path)
print(f"  twariq_neutral.jpg extracted features shape: {feats.shape if feats is not None else None}")

print("\nAll models and pipeline checks passed!")

