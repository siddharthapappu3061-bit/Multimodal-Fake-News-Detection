"""Bounded Streamlit caching for the six-model application."""
import streamlit as st


@st.cache_resource(max_entries=1, show_spinner=False)
def load_six_predictor(base, clip, gated, signature):
    from gated_integration import SixModelPredictor
    return SixModelPredictor(base,clip,gated)


@st.cache_data(max_entries=2, show_spinner=False)
def load_six_reports(base,clip,gated,signature):
    from gated_integration import verified_reports
    return verified_reports(base,clip,gated)
