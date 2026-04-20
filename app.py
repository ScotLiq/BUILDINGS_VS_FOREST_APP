import streamlit as st
import joblib
import pandas as pd
from PIL import Image
from feature_extractor import extract_all_features

st.set_page_config(
    page_title="Buildings vs Forest Classifier",
    page_icon="🌲",
    layout="centered"
)

st.title("🏢 🌲 Buildings vs Forest Classifier")
st.markdown("Upload an image to classify it as **Buildings** or **Forest**.")

SUPPORTED_EXT = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

@st.cache_resource
def load_model():
    artifacts = joblib.load("buildings_vs_forest_pipeline.joblib")
    return artifacts['pipeline'], artifacts['label_encoder']

pipeline, le = load_model()

uploaded_files = st.file_uploader(
    "Choose an image",
    type=list(SUPPORTED_EXT),
    help="Supported: JPG, JPEG, PNG, BMP, TIF, TIFF, WEBP",
    key=f"uploader_{st.session_state.uploader_key}",
    accept_multiple_files=True
)

# ── filter valid vs bad files ─────────────────────────────────────────────────
valid_file   = None
bad_files    = []
loaded_image = None

for f in (uploaded_files or []):
    ext = f.name.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXT:
        bad_files.append(f.name)
        continue
    try:
        img_test = Image.open(f).convert("RGB")
        if valid_file is None:
            valid_file   = f
            loaded_image = img_test
    except Exception:
        bad_files.append(f.name)

# ── reset everything if any bad file is detected ─────────────────────────────
if bad_files:
    st.error(
        f"❌ Unsupported file(s) detected: **{', '.join(bad_files)}**\n\n"
        f"Please upload only: **{', '.join(sorted(SUPPORTED_EXT)).upper()}**"
    )
    st.session_state.uploader_key += 1
    st.rerun()

# ── classify the valid file ───────────────────────────────────────────────────
if valid_file is not None:
    st.image(loaded_image, caption=f"Classifying: {valid_file.name}", use_container_width=True)

    with st.spinner("Classifying image..."):
        try:
            valid_file.seek(0)
            feats_dict = extract_all_features(valid_file)
            input_df   = pd.DataFrame([feats_dict])
            input_df   = input_df.reindex(columns=pipeline.feature_names_in_, fill_value=0.0)

            pred  = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]

            predicted_class = le.inverse_transform([pred])[0]
            confidence      = proba[pred] * 100

            veg_ratio    = feats_dict.get('vegetation_pixel_ratio', 0.0)
            edge_density = feats_dict.get('edge_density',           0.0)

            # Fix: sky-heavy forests misread as buildings
            if predicted_class == 'buildings' and veg_ratio > 0.28:
                predicted_class = 'forest'
                confidence      = max(confidence, 72.0)

            # ── strict gate ───────────────────────────────────────────────────
            CONFIDENCE_THRESHOLD = 50.0
            MIN_VEG_FOR_FOREST   = 0.10
            MIN_EDGE_FOR_BLDG    = 0.10

            is_plausible     = True
            rejection_reason = ""

            if confidence < CONFIDENCE_THRESHOLD:
                is_plausible     = False
                rejection_reason = f"model confidence too low ({confidence:.1f}% < {CONFIDENCE_THRESHOLD}%)"
            elif predicted_class == "forest" and veg_ratio < MIN_VEG_FOR_FOREST:
                is_plausible     = False
                rejection_reason = f"predicted forest but vegetation ratio is very low ({veg_ratio:.2f} < {MIN_VEG_FOR_FOREST})"
            elif predicted_class == "buildings" and edge_density < MIN_EDGE_FOR_BLDG:
                is_plausible     = False
                rejection_reason = f"predicted buildings but edge density is very low ({edge_density:.3f} < {MIN_EDGE_FOR_BLDG})"
            
            if not is_plausible:
                st.warning("🤔 **Prediction: UNKNOWN / UNRECOGNIZED**")
                st.caption(f"Rejected because: {rejection_reason}.")
                st.info(
                    "This image doesn't appear to contain a recognizable "
                    "**forest** or **buildings** scene. "
                    "Please upload a clearer aerial/landscape photo."
                )
            else:
                if predicted_class == "forest":
                    st.success("🌲 **Prediction: FOREST**")
                else:
                    st.error("🏢 **Prediction: BUILDINGS**")

                st.metric(label="Confidence", value=f"{confidence:.1f}%")
                label = "Forest" if predicted_class == "forest" else "Buildings"
                st.progress(confidence / 100, text=f"{label} Confidence: {confidence:.1f}%")

        except Exception as e:
            st.error(f"Error during classification: {str(e)}")

st.caption("Uses your custom hand-crafted image features")