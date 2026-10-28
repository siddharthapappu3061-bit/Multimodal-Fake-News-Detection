from pathlib import Path
import json
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Multimodal News Analysis", page_icon="📰", layout="wide")
BUNDLE = Path(__file__).resolve().parent / "artifacts"
st.title("Multimodal News Analysis")
st.caption("Academic Fakeddit prototype • Predictions reflect learned dataset patterns, not independent fact verification.")
page = st.sidebar.radio("Explore", ["Fake News Detection", "Model Comparison", "Dataset Analysis", "Model Architecture"])


@st.cache_resource
def load_predictor(bundle):
    from inference import Predictor
    return Predictor(bundle)


if page == "Fake News Detection":
    title = st.text_area("News headline or social media post")
    uploaded = st.file_uploader("Accompanying image", type=["jpg", "jpeg", "png", "webp"])
    if st.button("Analyze", type="primary"):
        if not title.strip() or uploaded is None:
            st.warning("Enter text and upload an image to compare all three models.")
        elif not (BUNDLE / "config.json").exists():
            st.error("Run colab_training.ipynb, download windows_demo.zip, and extract its artifacts folder here first.")
        else:
            try:
                with st.spinner("Running the three models on CPU…"):
                    image = Image.open(uploaded).convert("RGB")
                    predictions = load_predictor(str(BUNDLE)).predict(title, image)
                for column, (kind, probabilities) in zip(st.columns(3), predictions.items()):
                    label = ["Fake", "Real"][int(probabilities[1] > probabilities[0])]
                    column.metric(kind.title(), label, f"{max(probabilities):.1%} model probability")
                st.bar_chart(pd.DataFrame(predictions, index=["Fake", "Real"]).T)
                st.image(image, width=320)
            except Exception as exc:
                st.error(f"Could not load/analyze the exported bundle: {exc}")
elif page == "Model Comparison":
    path = BUNDLE / "results" / "metrics.json"
    if path.exists():
        result = json.loads(path.read_text())
        st.caption("Same 500 test pairs for all models. Precision, recall, F1 and ROC-AUC use Fake as the positive class.")
        st.dataframe(pd.DataFrame(result["models"]).T.drop(columns="confusion_matrix"))
        st.write("Fusion strictly outperforms both unimodal models:", result["fusion_outperforms_both"])
        for kind in result["models"]:
            st.image(str(BUNDLE / "results" / f"{kind}.png"))
        st.dataframe(pd.read_csv(BUNDLE / "results" / "predictions.csv").head(20))
    else:
        st.info("Evaluation results appear after Colab training and export.")
elif page == "Dataset Analysis":
    report_path = BUNDLE / "results" / "dataset_validation.json"
    if not report_path.exists():
        report_path = Path(__file__).resolve().parent / "results" / "dataset_validation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        st.dataframe(pd.DataFrame({s: {"samples": d["count"], "Fake": d["classes"].get("0", 0), "Real": d["classes"].get("1", 0)} for s, d in report["splits"].items()}).T)
        st.write("Existing clean_title and paired local images are used without resampling or relabeling.")
        st.json(report)
    data_root = Path(__file__).resolve().parent / "data"
    if (data_root / "test_processed.csv").exists():
        from dataset import resolve_image
        samples = pd.read_csv(data_root / "test_processed.csv", dtype={"id": str})
        selected = st.selectbox("Test sample", range(len(samples)))
        row = samples.iloc[selected]
        st.write(row.clean_title)
        st.write("Dataset label:", {0: "Fake", 1: "Real"}[int(row["2_way_label"])])
        st.image(str(resolve_image(data_root, row.local_image_path)), width=300)
else:
    st.graphviz_chart('digraph { rankdir=LR; Text -> DistilBERT -> "768 features"; Image -> ResNet50 -> "2048 features"; "768 features" -> "Text head"; "2048 features" -> "Image head"; "768 features" -> "256 text projection" -> Concatenate; "2048 features" -> "256 image projection" -> Concatenate; Concatenate -> "512 → 128 → 2 fusion head"; }')
    st.write("DistilBERT uses attention-mask-aware mean pooling (128-token limit). ResNet50 uses IMAGENET1K_V2 transforms and global pooled features. Both pretrained encoders stay frozen.")
    st.write("Unimodal heads: input → 256 → ReLU → Dropout(0.3) → 2. Fusion: separate 256-dimensional projections → concatenation → 128 → ReLU → Dropout(0.3) → 2.")
    st.write("Feature extraction and head training run on a Colab GPU. This app loads the exported encoders and heads on CPU without network downloads.")
