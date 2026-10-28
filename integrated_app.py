"""Five-model UI, staged and tested before the existing entrypoints are updated."""
from io import BytesIO
from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st
from PIL import Image

from integration_support import MODEL_NAMES, DEFAULT_MODEL, bundle_paths, asset_signature, read_json, load_evaluation
from integration_runtime import load_predictor

PROJECT = Path(__file__).resolve().parent


def main():
    st.set_page_config(page_title="Multimodal News Analysis", page_icon="📰", layout="wide")
    baseline, clip = bundle_paths(PROJECT)
    st.title("Multimodal News Analysis")
    st.caption("Academic Fakeddit prototype • Predictions reflect learned dataset patterns, not independent fact verification.")
    page = st.sidebar.radio("Explore", ["Fake News Detection", "Model Comparison", "Dataset Analysis", "Model Architecture"])

    if page == "Fake News Detection":
        primary = st.selectbox("Classification model", list(MODEL_NAMES), index=list(MODEL_NAMES).index(DEFAULT_MODEL),
                               format_func=MODEL_NAMES.get, key="primary_model")
        st.caption("Default: DistilBERT + CLIP Vision • 83.4% accuracy on the 500-example test split. All five models remain available for comparison.")
        title = st.text_area("News headline or social media post", key="headline")
        uploaded = st.file_uploader("Accompanying image", type=["jpg", "jpeg", "png", "webp"])
        compare = st.checkbox("Compare all five models", value=True)
        raw = uploaded.getvalue() if uploaded is not None else None
        input_key = hashlib.sha256(title.strip().encode() + (raw or b"")).hexdigest()
        if st.button("Analyze", type="primary"):
            if not title.strip() or raw is None:
                st.warning("Enter text and upload an image to analyze it with the trained models.")
            else:
                try:
                    signature = asset_signature(baseline, clip)
                    with st.spinner("Running the five models on CPU…"):
                        with Image.open(BytesIO(raw)) as opened:
                            # Keep metadata for the encoders' existing EXIF/RGB handling.
                            image = opened.copy()
                        scores = load_predictor(str(baseline), str(clip), signature).predict(title, image)
                    st.session_state["analysis"] = {"input_key": input_key, "scores": scores, "image": raw}
                except Exception as exc:
                    st.session_state.pop("analysis", None)
                    st.error(f"Could not load/analyze the exported models: {exc}")
        result = st.session_state.get("analysis")
        if result and result["input_key"] == input_key:
            scores = result["scores"]
            probabilities = scores[primary]
            label = ["Fake", "Real"][int(probabilities[1] > probabilities[0])]
            st.subheader(MODEL_NAMES[primary])
            st.metric("Prediction", label, f"{max(probabilities):.1%} model probability")
            st.caption(f"Fake: {probabilities[0]:.2%} · Real: {probabilities[1]:.2%}")
            if compare:
                # Retain the original three-column baseline row; add the CLIP row below.
                for kinds in (("text", "image", "fusion"), ("clip_image", "clip_fusion")):
                    for column, kind in zip(st.columns(len(kinds)), kinds):
                        p = scores[kind]
                        column.metric(MODEL_NAMES[kind], ["Fake", "Real"][int(p[1] > p[0])], f"{max(p):.1%} model probability")
                table = pd.DataFrame({MODEL_NAMES[k]: p for k, p in scores.items()}, index=["Fake", "Real"]).T
                st.bar_chart(table)
                st.dataframe(table.style.format("{:.2%}"))
            st.image(result["image"], width=320)
        elif result:
            st.info("The input has changed. Click Analyze to update the predictions.")

    elif page == "Model Comparison":
        try:
            result = load_evaluation(baseline, clip)
            st.caption("Same 500 test pairs for all five models. Precision, recall, F1 and ROC-AUC use Fake as the positive class; macro-F1 averages Fake and Real.")
            table = pd.DataFrame({MODEL_NAMES[k]: v for k, v in result["models"].items()}).T
            numeric = table.drop(columns="confusion_matrix").astype(float)
            st.dataframe(numeric.style.format("{:.2%}"))
            st.bar_chart(numeric[["accuracy", "macro_f1"]])
            st.write("DistilBERT + CLIP Vision has the highest test accuracy in this run (83.4%). Its validation macro-F1 also exceeded the original fusion model.")
            for row in (("text", "image", "fusion"), ("clip_image", "clip_fusion")):
                for col, kind in zip(st.columns(len(row)), row):
                    col.markdown(f"**{MODEL_NAMES[kind]}**")
                    col.dataframe(pd.DataFrame(result["models"][kind]["confusion_matrix"],
                        index=["Actual Fake", "Actual Real"], columns=["Predicted Fake", "Predicted Real"]))
            with st.expander("Original training curves and evaluation charts"):
                for kind in ("text", "image", "fusion"):
                    path = baseline / "results" / f"{kind}.png"
                    if path.is_file():
                        st.image(str(path))
            st.dataframe(pd.read_csv(clip / "results/test_predictions.csv", dtype={"id": str}).head(20))
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Could not read verified evaluation results: {exc}")

    elif page == "Dataset Analysis":
        report_path = baseline / "results/dataset_validation.json"
        if report_path.is_file():
            report = read_json(report_path)
            st.dataframe(pd.DataFrame({s: {"samples": d["count"], "Fake": d["classes"].get("0", 0), "Real": d["classes"].get("1", 0)} for s, d in report["splits"].items()}).T)
            st.write("Existing clean_title and paired local images are used without resampling or relabeling.")
            st.json(report)
        data_root = PROJECT / "data"
        if (data_root / "test_processed.csv").is_file():
            from dataset import resolve_image
            samples = pd.read_csv(data_root / "test_processed.csv", dtype={"id": str})
            selected = st.selectbox("Test sample", range(len(samples)))
            row = samples.iloc[selected]
            st.write(row.clean_title)
            st.write("Dataset label:", {0: "Fake", 1: "Real"}[int(row["2_way_label"])])
            st.image(str(resolve_image(data_root, row.local_image_path)), width=300)
    else:
        st.graphviz_chart('digraph { rankdir=LR; Text -> DistilBERT -> "768 text features"; Image -> ResNet50 -> "2048 image features"; Image -> "CLIP Vision" -> "512 CLIP features"; "768 text features" -> "Text head"; "2048 image features" -> "ResNet50 head"; "512 CLIP features" -> "CLIP image head"; "768 text features" -> "ResNet50 fusion"; "2048 image features" -> "ResNet50 fusion"; "768 text features" -> "CLIP fusion (default)"; "512 CLIP features" -> "CLIP fusion (default)"; }')
        st.write("A single frozen DistilBERT encoder supplies 768-dimensional attention-mask-aware mean features (128-token limit) to the text head and both fusion heads.")
        st.write("ResNet50 retains its IMAGENET1K_V2 transforms and 2048-dimensional features. Frozen CLIP ViT-B/32 uses its exported image processor and L2-normalized 512-dimensional projected features.")
        st.write("Each image-only head projects its input to 256, then ReLU → Dropout(0.3) → 2. Each fusion head projects text and image separately to 256, concatenates them, then uses 512 → 128 → ReLU → Dropout(0.3) → 2.")
        st.write("All five trained heads and three frozen encoders are cached on CPU. Training remains in Colab; this application performs inference only.")


if __name__ == "__main__":
    main()
