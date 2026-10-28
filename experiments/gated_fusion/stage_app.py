"""Create a separate six-model application from the preserved five-model UI."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
source=(ROOT/'integrated_app.py').read_text(encoding='utf-8')
source=source.replace('"""Five-model UI, staged and tested before the existing entrypoints are updated."""','"""Six-model UI preserving the original theme, navigation and comparison layout."""')
source=source.replace('from integration_runtime import load_predictor', '''from gated_integration import SIX_MODEL_NAMES, gated_bundle_path, gated_signature, default_model
from gated_runtime import load_six_predictor, load_six_reports''')
source=source.replace('    st.title("Multimodal News Analysis")','''    st.title("Multimodal News Analysis")
    gated = gated_bundle_path(PROJECT)
    try:
        signature = gated_signature(baseline, clip, gated)
        reports = load_six_reports(str(baseline), str(clip), str(gated), signature)
        primary_default = default_model(gated, reports)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not verify the gated export: {exc}")
        st.info("The preserved five-model app is available with: streamlit run integrated_app.py")
        st.stop()''')
source=source.replace('list(MODEL_NAMES)', 'list(SIX_MODEL_NAMES)').replace('.index(DEFAULT_MODEL)', '.index(primary_default)').replace('format_func=MODEL_NAMES.get','format_func=SIX_MODEL_NAMES.get')
source=source.replace('        st.caption("Default: DistilBERT + CLIP Vision • 83.4% accuracy on the 500-example test split. All five models remain available for comparison.")', '''        st.caption(f"Default: {SIX_MODEL_NAMES[primary_default]} • Selected using validation performance and local CPU verification. All six models remain available.")''')
source=source.replace('Compare all five models','Compare all six models').replace('Running the five models on CPU','Running the six models on CPU')
source=source.replace('                    signature = asset_signature(baseline, clip)\n','')
source=source.replace('                        scores = load_predictor(str(baseline), str(clip), signature).predict(title, image)',
'''                        diagnostic = load_six_predictor(str(baseline), str(clip), str(gated), signature).analyze(title, image)
                        scores = diagnostic["scores"]''')
source=source.replace('st.session_state["analysis"]', 'st.session_state["six_analysis"]').replace('pop("analysis",', 'pop("six_analysis",').replace('get("analysis")','get("six_analysis")')
source=source.replace('{"input_key": input_key, "scores": scores, "image": raw}', '{"input_key": input_key, "signature": signature, "scores": scores, "image": raw, "diagnostic": diagnostic}')
source=source.replace('if result and result["input_key"] == input_key:', 'if result and result["input_key"] == input_key and result.get("signature") == signature:')
source=source.replace('MODEL_NAMES[','SIX_MODEL_NAMES[').replace('SIX_SIX_MODEL_NAMES','SIX_MODEL_NAMES')
source=source.replace('("clip_image", "clip_fusion")', '("clip_image", "clip_fusion", "gated_fusion")')
source=source.replace('            st.image(result["image"], width=320)', '''            diagnostic = result["diagnostic"]
            with st.container(border=True):
                st.subheader("Text–image disagreement")
                st.metric("Disagreement score", f"{diagnostic['disagreement']:.3f}")
                st.caption(f"Absolute difference between text-only and CLIP image-only Fake probabilities (0–1). Validation-selected threshold: {diagnostic['threshold']:.3f}.")
                if diagnostic['warning']:
                    st.info(diagnostic['warning'])
                else:
                    st.caption("No disagreement warning is triggered for this input.")
                st.caption("This is an exploratory reliability signal, not proof of misinformation. It never changes the predicted label.")
                st.subheader("Gated fusion coefficients")
                text_column, image_column = st.columns(2)
                text_column.metric("Text gate", f"{diagnostic['g_text']:.3f}")
                image_column.metric("Image gate", f"{diagnostic['g_image']:.3f}")
                st.bar_chart(pd.DataFrame({'Coefficient': [diagnostic['g_text'], diagnostic['g_image']]}, index=['Text','Image']))
                st.caption("These learned coefficients sum to 1. They are not definitive feature-importance percentages or guarantees of which modality determined the prediction.")
            st.image(result["image"], width=320)''')
source=source.replace('result = load_evaluation(baseline, clip)', 'result = reports["dashboard"]')
source=source.replace('all five models','all six models')
source=source.replace('            st.write("DistilBERT + CLIP Vision has the highest test accuracy in this run (83.4%). Its validation macro-F1 also exceeded the original fusion model.")', '''            comparison = pd.DataFrame([
                {"Model": SIX_MODEL_NAMES[k], "Validation macro-F1": reports['validate'][name]['macro_f1'],
                 "Test accuracy": reports['test'][name]['accuracy'], "Test macro-F1": reports['test'][name]['macro_f1'],
                 "Test ROC-AUC": reports['test'][name]['roc_auc']}
                for k, name in [('clip_fusion','concatenation'),('gated_fusion','gated')]
            ]).set_index('Model')
            st.subheader("Fusion validation and test comparison")
            st.dataframe(comparison.style.format("{:.2%}"))
            st.caption("Default promotion uses validation macro-F1 and verified CPU parity; test results are reported without model tuning.")''')
source=source.replace('"CLIP fusion (default)"','"CLIP fusion"')
source=source.replace('; }\')', '; "768 text features" -> "Gated fusion"; "512 CLIP features" -> "Gated fusion"; }\')')
source=source.replace('text head and both fusion heads','text head and all three fusion heads')
source=source.replace('Each fusion head projects', 'Each concatenation fusion head projects')
source=source.replace('        st.write("All five trained heads', '''        st.write("Gated fusion projects both modalities to 256 with LayerNorm and ReLU, learns two softmax gates, then classifies the concatenated weighted features. Gate coefficients are not definitive feature importance.")
        st.write("All six trained heads''')
(ROOT/'six_model_app.py').write_text(source, encoding='utf-8')
compile(source,'six_model_app.py','exec')
print('Staged six_model_app.py; original entrypoints unchanged.')
