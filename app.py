from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------------------------
# CONFIGURASI HALAMAN
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apple Disease Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# FORCE LIGHT MODE OVERRIDE (MEMATIKAN DARK MODE SISTEM)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color-scheme: light !important; /* Paksa browser merender Light Mode */
    }

    /* Force Root Background & Text Color */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    
    /* Force Sidebar Style Light Mode */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* Headings Kontras Tinggi */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    
    p, li, span, label, div {
        color: #334155 !important;
    }

    /* Card Styling */
    .pro-card {
        background-color: #FFFFFF !important;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
    }

    .pro-card-header {
        border-left: 4px solid #059669; /* Emerald Green */
        padding-left: 12px;
        margin-bottom: 16px;
    }

    .step-card {
        background-color: #F1F5F9 !important;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        height: 100%;
    }

    .step-number {
        display: inline-block;
        background-color: #059669;
        color: #FFFFFF !important;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        text-align: center;
        line-height: 26px;
        font-weight: 700;
        font-size: 13px;
    }

    /* ========================================== */
    /* FILE UPLOADER LIGHT MODE FORCE             */
    /* ========================================== */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 8px;
    }

    [data-testid="stFileUploader"] label {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Area Kotak Drag & Drop */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F8FAFC !important;
        border: 2px dashed #CBD5E1 !important;
    }

    /* Teks Instruksi di dalam Dropzone */
    [data-testid="stFileUploaderDropzoneInstructions"] div, 
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #475569 !important;
    }

    /* Tombol Upload (Browse files) */
    [data-testid="stFileUploader"] button {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* File Info jika file sudah terupload */
    [data-testid="stFileUploaderFileData"] * {
        color: #0F172A !important;
    }

    /* Expander Text Fix */
    .streamlit-expanderHeader {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        color: #334155 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)
