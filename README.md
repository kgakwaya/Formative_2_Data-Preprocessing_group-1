# Formative 2: Multimodal Data Preprocessing

User identity and product recommendation system: a face check, then a voice check, then a product prediction.

This folder holds **Task 1 (data merge)**, **Task 2 (image collection and processing)**, and the **Streamlit interactive demo** (Task 6).

## Contents

| Path | What it is |
|---|---|
| `task1_data_merge.ipynb` | EDA, cleaning, join logic and post-merge checks, feature engineering |
| `task2_image_processing.ipynb` | face photos: loading, face detection and alignment, augmentation, feature extraction, feature checks |
| `Gakwaya_Ineza_ketia_formative2_task3_voiceprint.ipynb` | voice data: audio features, voiceprint model training & evaluation |
| `Formative_2_Data_Merge_v5.ipynb` | product recommendation model: feature engineering, model comparison, final pipeline |
| `app.py` | **Streamlit demo app** — the full 3-model interactive journey |
| `src/face_engine.py` | facial recognition engine (YuNet + SFace, used by the app) |
| `src/voice_engine.py` | voiceprint verification engine (librosa MFCCs, used by the app) |
| `src/recommender_engine.py` | product recommendation engine (scikit-learn pipeline, used by the app) |
| `src/image_features.py` | the face pipeline (detect → align → describe), shared with the CLI app |
| `data/raw/` | `customer_social_profiles.csv`, `customer_transactions.csv` (as provided) |
| `data/processed/merged_customer_dataset.csv` | merged, engineered dataset for the product model: 117 rows × 29 columns |
| `data/processed/product_label_map.json` | product category ↔ integer label |
| `images/raw/` | 3 photos per member, named `<member>_<expression>.jpg` (neutral, smiling, surprised) |
| `images/augmented/` | 6 augmentations per photo, named `<member>_<expression>_<augmentation>.jpg` |
| `images/faces/` | the aligned 112×112 face crops the features are computed from |
| `features/image_features.csv` | 63 rows × 170 columns: metadata, 128-d face embedding, colour and texture histograms |
| `audio/` | voice recordings (`.flac`) for each member + an unknown speaker |
| `verify_voice/voice_model.joblib` | trained voiceprint model |
| `models/product_recommender.joblib` | trained product recommendation pipeline |
| `figures/` | every figure from both notebooks, for the report |
| `docs/system_flow.png` | the system flow diagram |

## Run the Notebooks

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
jupyter notebook
```

Run `task1_data_merge.ipynb`, then `task2_image_processing.ipynb`, top to bottom. On first use, Task 2 downloads two pretrained OpenCV models (YuNet face detector, SFace face embedder; about 39 MB) into `models/`. That folder is git-ignored.

Both notebooks were run end to end with Python 3.13, OpenCV 5.0, pandas 3.1 and scikit-learn 1.9.

## Run the Streamlit Demo App

The Streamlit app (`app.py`) implements the full **3-model interactive journey**:

```
Start → Facial Recognition → Product Recommendation → Voice Validation → Display Predicted Product
```

### Prerequisites

Make sure you have already:
1. Run the notebooks above (or at least have the trained model files in place).
2. The following model files must exist:
   - `models/face_detection_yunet_2023mar.onnx` (downloaded automatically by the notebook)
   - `models/face_recognition_sface_2021dec.onnx` (downloaded automatically by the notebook)
   - `models/product_recommender.joblib` (created by `Formative_2_Data_Merge_v5.ipynb`)
   - `verify_voice/voice_model.joblib` (created by the voiceprint notebook)
   - `features/image_features.csv` (created by `task2_image_processing.ipynb`)

### Install & Launch

```bash
# Activate your virtual environment
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install all dependencies (includes streamlit, librosa, etc.)
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

This opens the app at **http://localhost:8501** in your browser.

### How to Use

1. **Start** — Click **Begin Journey** to enter the flow.
2. **Facial Recognition** — Choose a demo preset photo (Audric / Ketia / Twariq), use your webcam, or upload an image. The model runs YuNet face detection + SFace embedding matching. If the face is not recognized → **Access Denied**.
3. **Product Recommendation + Voice Validation** — Once the face passes, the product model runs automatically in the background. Then select or record a voice sample. The voiceprint model checks that the voice matches the authenticated face. If it fails → **Access Denied**.
4. **Display Predicted Product** — If both biometric gates pass, the predicted product category is revealed with confidence scores, category distributions, the customer profile used, and a **What-If simulator** to experiment with different inputs.

## For the teammates on Tasks 3–6

**Product model (Task 4).** Target: `product_label`. Inputs: the 24 feature columns of `merged_customer_dataset.csv`; drop `transaction_id`, `customer_id` and `purchase_date`. The classes are balanced, so beat the 24% majority-class baseline (Sports). Use stratified cross-validation; there are only 117 rows.

**Face model (Task 4).** Label: `member`. Inputs: the `emb_*` columns, optionally plus `hsv_*`, `lbp_*`, `brightness` and `contrast`. Split by source photo, not by row, so copies of one photo never sit on both sides of the split.

**CLI demo (Task 6).** Process a login photo the same way the training rows were made:

```python
from src.image_features import features_from_path
x = features_from_path("login.jpg")   # None means no face found, so deny access
```

Compare the login face against every member and take the best match first, then reject it if that match is below a threshold (OpenCV's same-person threshold for SFace is 0.363 cosine similarity). Don't use the threshold alone: `ketia_surprised` and `audric_surprised` score 0.37 against each other, because both photos show the same exaggerated expression. Best-match-first gets the member right for all 54 augmented copies.

## Note on the written interpretations

The text under each figure quotes numbers from the current data and photos. If a photo or dataset changes, re-run the notebook and update those sentences.
