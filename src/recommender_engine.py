"""Product Recommender Engine for the Streamlit Application.
Uses the trained pipeline from models/product_recommender.joblib.
"""
import os
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "product_recommender.joblib")

CATEGORIES = ["Books", "Clothing", "Electronics", "Groceries", "Sports"]

_recommender_bundle = None

# Default representative profile templates
MEMBER_DEFAULT_PROFILES = {
    "twariq": {
        "customer_id": "CUST-1042",
        "engagement_mean": 82.5,
        "interest_mean": 4.6,
        "platform": "LinkedIn",
        "sentiment": "Positive",
        "prior_purchases": 6,
        "prior_avg_amount": 320.0,
        "prior_avg_rating": 4.5,
        "prior_share_Books": 0.15,
        "prior_share_Clothing": 0.10,
        "prior_share_Electronics": 0.50,
        "prior_share_Groceries": 0.05,
        "prior_share_Sports": 0.20,
    },
    "audric": {
        "customer_id": "CUST-1088",
        "engagement_mean": 68.0,
        "interest_mean": 3.8,
        "platform": "Instagram",
        "sentiment": "Positive",
        "prior_purchases": 4,
        "prior_avg_amount": 185.0,
        "prior_avg_rating": 4.1,
        "prior_share_Books": 0.10,
        "prior_share_Clothing": 0.40,
        "prior_share_Electronics": 0.20,
        "prior_share_Groceries": 0.10,
        "prior_share_Sports": 0.20,
    },
    "ketia": {
        "customer_id": "CUST-1065",
        "engagement_mean": 74.0,
        "interest_mean": 4.2,
        "platform": "TikTok",
        "sentiment": "Positive",
        "prior_purchases": 5,
        "prior_avg_amount": 240.0,
        "prior_avg_rating": 4.3,
        "prior_share_Books": 0.30,
        "prior_share_Clothing": 0.30,
        "prior_share_Electronics": 0.10,
        "prior_share_Groceries": 0.20,
        "prior_share_Sports": 0.10,
    },
}

CATEGORY_METADATA = {
    "Electronics": {
        "icon": "⚡",
        "tagline": "High-Performance Tech & Gadgets",
        "color": "#3B82F6",
        "badge": "Electronics",
        "desc": "Smart devices, modern computing gear, audio equipment, and productivity peripherals.",
    },
    "Sports": {
        "icon": "🏃",
        "tagline": "Fitness & Outdoor Gear",
        "color": "#10B981",
        "badge": "Sports & Active",
        "desc": "Athletic apparel, cardio training gear, sports equipment, and outdoor accessories.",
    },
    "Clothing": {
        "icon": "👕",
        "tagline": "Apparel & Modern Fashion",
        "color": "#EC4899",
        "badge": "Clothing",
        "desc": "Trendsetting seasonal wear, everyday comfort apparel, and designer accessories.",
    },
    "Books": {
        "icon": "📚",
        "tagline": "Knowledge, Fiction & Learning",
        "color": "#8B5CF6",
        "badge": "Books & Literature",
        "desc": "Top-rated non-fiction, industry guides, bestsellers, and digital reads.",
    },
    "Groceries": {
        "icon": "🛒",
        "tagline": "Organic Essentials & Pantry",
        "color": "#F59E0B",
        "badge": "Groceries",
        "desc": "Fresh organic staples, specialty beverages, artisanal goods, and daily essentials.",
    },
}


def get_recommender_bundle():
    global _recommender_bundle
    if _recommender_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Product recommender model not found at {MODEL_PATH}")
        _recommender_bundle = joblib.load(MODEL_PATH)
    return _recommender_bundle


def get_default_profile(member_name=None):
    if member_name and member_name.lower() in MEMBER_DEFAULT_PROFILES:
        return MEMBER_DEFAULT_PROFILES[member_name.lower()].copy()
    return MEMBER_DEFAULT_PROFILES["twariq"].copy()


def recommend_product(profile_dict, top_k=5):
    """
    Generate product recommendations using the trained pipeline.

    Args:
        profile_dict: dictionary containing customer social and transactional features.
        top_k: number of top category recommendations to return.

    Returns:
        dict with top category, confidence, probabilities dict, and category metadata.
    """
    bundle = get_recommender_bundle()
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    classes = bundle.get("classes", list(pipeline.classes_))

    # Prepare input dataframe with exact feature columns
    row = {col: profile_dict.get(col, 0.0) for col in feature_columns}
    # Ensure types are correct
    row["engagement_mean"] = float(row.get("engagement_mean", 50.0))
    row["interest_mean"] = float(row.get("interest_mean", 3.0))
    row["platform"] = str(row.get("platform", "Instagram"))
    row["sentiment"] = str(row.get("sentiment", "Positive"))

    input_df = pd.DataFrame([row])[feature_columns]

    probabilities = pipeline.predict_proba(input_df)[0]
    order = np.argsort(probabilities)[::-1]

    top_category = classes[order[0]]
    top_confidence = float(probabilities[order[0]])

    all_probabilities = {classes[i]: float(probabilities[i]) for i in order}
    top_categories = {classes[i]: float(probabilities[i]) for i in order[:top_k]}

    metadata = CATEGORY_METADATA.get(top_category, {
        "icon": "📦",
        "tagline": f"Premium {top_category}",
        "color": "#6366F1",
        "badge": top_category,
        "desc": f"Recommended selection in {top_category}.",
    })

    return {
        "recommended_category": top_category,
        "confidence": top_confidence,
        "top_categories": top_categories,
        "all_probabilities": all_probabilities,
        "metadata": metadata,
        "input_profile": row,
    }

