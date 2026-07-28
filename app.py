st.markdown(
    """
    <style>
    /* ... kode CSS lainnya ... */

    /* === WARNA TULISAN UPLOAD FILE === */
    /* Label atas */
    [data-testid="stFileUploader"] label {
        color: #0F172A !important; /* Warna teks label atas */
        font-weight: 600 !important;
    }

    /* Teks dalam area Dropzone */
    [data-testid="stFileUploaderDropzoneInstructions"] div, 
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #475569 !important; /* Warna teks "Drag & drop" & format file */
    }

    /* Tombol 'Browse files' */
    [data-testid="stFileUploader"] button {
        color: #FFFFFF !important; /* Warna teks tombol */
        background-color: #059669 !important; /* Warna background tombol */
        border: none !important;
    }

    /* Teks nama file yang berhasil diupload */
    [data-testid="stFileUploaderFileData"] {
        color: #0F172A !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)
