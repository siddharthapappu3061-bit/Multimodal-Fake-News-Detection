"""Shared Streamlit resource cache; model imports/loading are deferred until Analyze."""
import streamlit as st


@st.cache_resource(max_entries=1, show_spinner=False)
def load_predictor(baseline_bundle, clip_bundle, artifact_signature):
    # The signature deliberately participates in the key to invalidate replaced bundles.
    from integrated_inference import FiveModelPredictor
    return FiveModelPredictor(baseline_bundle, clip_bundle)
