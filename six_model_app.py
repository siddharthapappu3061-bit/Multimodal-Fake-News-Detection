"""Six-model UI preserving the original theme, navigation and comparison layout."""
from io import BytesIO
from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st
from PIL import Image

from integration_support import MODEL_NAMES, DEFAULT_MODEL, bundle_paths, asset_signature, read_json, load_evaluation
from gated_integration import SIX_MODEL_NAMES, gated_bundle_path, gated_signature, default_model
from gated_runtime import load_six_predictor, load_six_reports
from reliability import assess, load_policy, displayed_decision, policy_description

PROJECT = Path(__file__).resolve().parent


def main():
    st.set_page_config(page_title="Multimodal News Analysis", page_icon="📰", layout="wide")
    baseline, clip = bundle_paths(PROJECT)
    st.title("Multimodal News Analysis")
    gated = gated_bundle_path(PROJECT)
    try:
        signature = gated_signature(baseline, clip, gated)
        reports = load_six_reports(str(baseline), str(clip), str(gated), signature)
        primary_default = default_model(gated, reports)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not verify the gated export: {exc}")
        st.info("The preserved five-model app is available with: streamlit run integrated_app.py")
        st.stop()
    st.caption("Academic Fakeddit prototype • Predictions reflect learned dataset patterns, not independent fact verification.")
    page = st.sidebar.radio("Explore", ["Fake News Detection", "Model Comparison", "Dataset Analysis", "Model Architecture"])

    if page == "Fake News Detection":
        policy = load_policy(PROJECT / 'experiments/reliability_combined/policy.json', gated)
        primary = st.selectbox("Classification model", list(SIX_MODEL_NAMES), index=list(SIX_MODEL_NAMES).index(primary_default),
                               format_func=SIX_MODEL_NAMES.get, key="primary_model")
        st.caption(f"Default: {SIX_MODEL_NAMES[primary_default]} • Selected using validation performance and local CPU verification. All six models remain available.")
        title = st.text_area("News headline or social media post", key="headline")
        uploaded = st.file_uploader("Accompanying image", type=["jpg", "jpeg", "png", "webp"])
        compare = st.checkbox("Compare all six models", value=True)
        selective = st.checkbox("Use selective classification for gated fusion", value=False,
                                disabled=not policy['available'], key='selective_classification',
                                help='Optional validation-checked policy. It changes only the displayed gated decision; all model probabilities remain unchanged.')
        if policy['available']:
            st.caption(f"Optional policy accepts when {policy_description(policy)}. Abstained inputs display Requires Verification. This trades coverage for accepted accuracy; it does not guarantee correctness.")
        else:
            st.caption("No compatible policy passed the validation checks. Informational warnings remain available.")
        raw = uploaded.getvalue() if uploaded is not None else None
        input_key = hashlib.sha256(title.strip().encode() + (raw or b"")).hexdigest()
        if st.button("Analyze", type="primary"):
            if not title.strip() or raw is None:
                st.warning("Enter text and upload an image to analyze it with the trained models.")
            else:
                try:
                    with st.spinner("Running the six models on CPU…"):
                        with Image.open(BytesIO(raw)) as opened:
                            # Keep metadata for the encoders' existing EXIF/RGB handling.
                            image = opened.copy()
                        diagnostic = load_six_predictor(str(baseline), str(clip), str(gated), signature).analyze(title, image)
                        scores = diagnostic["scores"]
                    st.session_state["six_analysis"] = {"input_key": input_key, "signature": signature, "scores": scores, "image": raw, "diagnostic": diagnostic}
                except Exception as exc:
                    st.session_state.pop("six_analysis", None)
                    st.error(f"Could not load/analyze the exported models: {exc}")
        result = st.session_state.get("six_analysis")
        if result and result["input_key"] == input_key and result.get("signature") == signature:
            scores = result["scores"]
            probabilities = scores[primary]
            label = ["Fake", "Real"][int(probabilities[1] > probabilities[0])]
            st.subheader(SIX_MODEL_NAMES[primary])
            decision = displayed_decision(scores, policy, selective, primary)
            st.metric("Decision", decision, f"{max(probabilities):.1%} model probability")
            st.caption(f"Underlying model prediction: {label}. Model probability is not factual certainty.")
            if decision == 'Requires Verification':
                st.info("The optional policy abstained. Review this input independently; the underlying Fake/Real prediction and probabilities have not changed.")
            st.caption(f"Fake: {probabilities[0]:.2%} · Real: {probabilities[1]:.2%}")
            if compare:
                # Retain the original three-column baseline row; add the CLIP row below.
                for kinds in (("text", "image", "fusion"), ("clip_image", "clip_fusion", "gated_fusion")):
                    for column, kind in zip(st.columns(len(kinds)), kinds):
                        p = scores[kind]
                        column.metric(SIX_MODEL_NAMES[kind], ["Fake", "Real"][int(p[1] > p[0])], f"{max(p):.1%} model probability")
                table = pd.DataFrame({SIX_MODEL_NAMES[k]: p for k, p in scores.items()}, index=["Fake", "Real"]).T
                st.bar_chart(table)
                st.dataframe(table.style.format("{:.2%}"))
            diagnostic = result["diagnostic"]
            with st.container(border=True):
                st.subheader("Gated fusion reliability")
                reliability = assess(scores['text'][0], scores['clip_image'][0], scores['gated_fusion'][0])
                st.write(f"Gated prediction: **{reliability['predicted_class']}** · model probability: **{reliability['model_probability']:.2%}**")
                st.write(f"Text model: **{reliability['text_prediction']}** (Fake {scores['text'][0]:.2%}) · CLIP Vision: **{reliability['image_prediction']}** (Fake {scores['clip_image'][0]:.2%})")
                st.caption('Unimodal predictions agree.' if reliability['unimodal_agreement'] else 'Unimodal predictions disagree.')
                st.metric("Disagreement score", f"{diagnostic['disagreement']:.3f}")
                st.caption(f"Absolute difference between text-only and CLIP image-only Fake probabilities (0–1). Validation-selected threshold: {diagnostic['threshold']:.3f}.")
                if diagnostic['warning']:
                    st.info(diagnostic['warning'])
                else:
                    st.caption("No disagreement warning is triggered for this input.")
                st.caption("Disagreement is an exploratory signal, not proof of misinformation or a reliable error detector by itself. Warnings never alter model predictions; optional abstention changes only the displayed decision.")
                st.subheader("Gated fusion coefficients")
                text_column, image_column = st.columns(2)
                text_column.metric("Text gate", f"{diagnostic['g_text']:.3f}")
                image_column.metric("Image gate", f"{diagnostic['g_image']:.3f}")
                st.bar_chart(pd.DataFrame({'Coefficient': [diagnostic['g_text'], diagnostic['g_image']]}, index=['Text','Image']))
                st.caption("These learned coefficients sum to 1. They are not definitive feature-importance percentages or guarantees of which modality determined the prediction.")
            st.image(result["image"], width=320)
        elif result:
            st.info("The input has changed. Click Analyze to update the predictions.")

    elif page == "Model Comparison":
        try:
            result = reports["dashboard"]
            st.caption("Same 500 test pairs for all six models. Precision, recall, F1 and ROC-AUC use Fake as the positive class; macro-F1 averages Fake and Real.")
            table = pd.DataFrame({SIX_MODEL_NAMES[k]: v for k, v in result["models"].items()}).T
            numeric = table.drop(columns="confusion_matrix").astype(float)
            st.dataframe(numeric.style.format("{:.2%}"))
            st.bar_chart(numeric[["accuracy", "macro_f1"]])
            comparison = pd.DataFrame([
                {"Model": SIX_MODEL_NAMES[k], "Validation macro-F1": reports['validate'][name]['macro_f1'],
                 "Test accuracy": reports['test'][name]['accuracy'], "Test macro-F1": reports['test'][name]['macro_f1'],
                 "Test ROC-AUC": reports['test'][name]['roc_auc']}
                for k, name in [('clip_fusion','concatenation'),('gated_fusion','gated')]
            ]).set_index('Model')
            st.subheader("Fusion validation and test comparison")
            st.dataframe(comparison.style.format("{:.2%}"))
            st.caption("Default promotion uses validation macro-F1 and verified CPU parity; test results are reported without model tuning.")
            reliability_path = PROJECT / 'experiments/reliability_combined/report.json'
            if reliability_path.is_file():
                with st.expander("Selective classification: accuracy and coverage"):
                    reliability_report = read_json(reliability_path)
                    st.caption("Confidence, disagreement and combined policies use validation-only threshold selection. The reused validation audit is not fresh independent evidence; test results do not select policies. Accepted Fake recall excludes abstained Fake cases; overall Fake recall counts abstentions as missed automatic detections.")
                    rows = []
                    for split in ['validation']:
                        for phase in ['before', 'after']:
                            m = reliability_report[split + '_' + phase]
                            rows.append({'Split': split, 'Policy': phase, **{k:m[k] for k in ['accepted_accuracy','coverage','abstention_rate','abstained_error_rate','accepted_fake_recall','overall_fake_recall']}})
                    st.dataframe(pd.DataFrame(rows))
                    st.write('Frozen candidate comparisons (these are not all enabled policies):')
                    candidates = pd.DataFrame(reliability_report['comparisons'])
                    st.dataframe(candidates[['split','strategy','threshold','coverage','accepted_accuracy','abstention_rate','abstained_error_rate','accepted_fake_recall']].assign(threshold=candidates.threshold.map(str)))
                    st.caption(reliability_report['policy']['reason'])
            for row in (("text", "image", "fusion"), ("clip_image", "clip_fusion", "gated_fusion")):
                for col, kind in zip(st.columns(len(row)), row):
                    col.markdown(f"**{SIX_MODEL_NAMES[kind]}**")
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
        st.graphviz_chart('digraph { rankdir=LR; Text -> DistilBERT -> "768 text features"; Image -> ResNet50 -> "2048 image features"; Image -> "CLIP Vision" -> "512 CLIP features"; "768 text features" -> "Text head"; "2048 image features" -> "ResNet50 head"; "512 CLIP features" -> "CLIP image head"; "768 text features" -> "ResNet50 fusion"; "2048 image features" -> "ResNet50 fusion"; "768 text features" -> "CLIP fusion"; "512 CLIP features" -> "CLIP fusion"; "768 text features" -> "Gated fusion"; "512 CLIP features" -> "Gated fusion"; }')
        st.write("A single frozen DistilBERT encoder supplies 768-dimensional attention-mask-aware mean features (128-token limit) to the text head and all three fusion heads.")
        st.write("ResNet50 retains its IMAGENET1K_V2 transforms and 2048-dimensional features. Frozen CLIP ViT-B/32 uses its exported image processor and L2-normalized 512-dimensional projected features.")
        st.write("Each image-only head projects its input to 256, then ReLU → Dropout(0.3) → 2. Each concatenation fusion head projects text and image separately to 256, concatenates them, then uses 512 → 128 → ReLU → Dropout(0.3) → 2.")
        st.write("Gated fusion projects both modalities to 256 with LayerNorm and ReLU, learns two softmax gates, then classifies the concatenated weighted features. Gate coefficients are not definitive feature importance.")
        st.write("All six trained heads and three frozen encoders are cached on CPU. Training remains in Colab; this application performs inference only.")


if __name__ == "__main__":
    main()
