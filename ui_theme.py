"""Dark presentation components; no inference or model changes."""
from html import escape
import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
    :root {color-scheme:dark;}
    .stApp {background:#0b1018;color:#e5edf7;}
    [data-testid="stHeader"] {background:rgba(11,16,24,.94);}
    [data-testid="stSidebar"] {background:#101824;border-right:1px solid #263244;}
    .block-container {max-width:1320px;padding-top:2.8rem;padding-bottom:4rem;}
    h1,h2,h3 {letter-spacing:-.035em;color:#edf4ff;}
    h1 {font-size:2.6rem!important;font-weight:700!important;}
    p,li,label {line-height:1.65;}
    [data-testid="stCaptionContainer"] {color:#a8b7cb;}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#29374b!important;border-radius:16px!important;}
    [data-testid="stTextArea"] textarea {background:#101b2a;color:#eff5ff;border:1px solid #34445c;border-radius:12px;}
    [data-testid="stFileUploaderDropzone"] {background:#101b2a;border:1px dashed #435672;border-radius:12px;}
    [data-baseweb="select"]>div {background:#152133;color:#eff5ff;}
    [data-testid="stExpander"] {background:#101824;border:1px solid #29374b;border-radius:12px;}
    [data-testid="stMetric"] {background:#111c2b;padding:18px;border:1px solid #29374b;border-radius:12px;}
    .stButton>button[kind="primary"] {background:#9dc5ff;color:#091426;border:none;border-radius:10px;font-weight:700;min-height:48px;}
    .stButton>button[kind="primary"]:hover {background:#bed9ff;color:#091426;}
    button:focus-visible,textarea:focus-visible {outline:2px solid #b3d2ff!important;outline-offset:3px;}
    .eyebrow {font-size:.73rem;letter-spacing:.16em;color:#9dc5ff;font-weight:700;margin-bottom:10px;}
    .hero-copy {color:#a8b7cb;font-size:1.05rem;max-width:780px;margin-bottom:28px;}
    .result-card {background:linear-gradient(125deg,#17263b,#111c2b);border:1px solid #3b5270;border-radius:18px;padding:28px;margin:18px 0;}
    .result-top {display:flex;justify-content:space-between;gap:20px;align-items:center;flex-wrap:wrap;}
    .result-label {font-size:2.6rem;font-weight:700;letter-spacing:-.04em;}
    .result-score {font-size:2rem;font-weight:650;}
    .muted {color:#a8b7cb;font-size:.88rem;}
    .prob-track {height:9px;background:#26374c;border-radius:20px;overflow:hidden;margin:20px 0 10px;}
    .prob-fill {height:100%;background:#a8caff;}
    .prob-labels {display:flex;justify-content:space-between;font-size:.85rem;color:#c5d3e5;}
    .empty-state {padding:26px;border:1px dashed #34445c;border-radius:14px;color:#a8b7cb;text-align:center;margin-top:20px;}
    @media(max-width:640px) {.block-container{padding-top:1.5rem;}h1{font-size:2rem!important;}.result-card{padding:20px;}}
    </style>""", unsafe_allow_html=True)

def result_card(name, decision, probabilities):
    fake, real = probabilities
    st.markdown(f"""
    <div class="result-card">
      <div class="eyebrow">MODEL ASSESSMENT</div>
      <div class="muted">{escape(name)}</div>
      <div class="result-top">
        <div class="result-label">{escape(decision)}</div>
        <div><div class="result-score">{max(fake,real):.1%}</div><div class="muted">Predicted-class model score</div></div>
      </div>
      <div class="prob-track"><div class="prob-fill" style="width:{fake*100:.4f}%"></div></div>
      <div class="prob-labels"><span>Fake {fake:.2%}</span><span>Real {real:.2%}</span></div>
    </div>""", unsafe_allow_html=True)
    st.caption("Model scores reflect learned Fakeddit patterns. They are not independently verified probabilities of truth.")
