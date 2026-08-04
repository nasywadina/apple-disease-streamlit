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
# CUSTOM CSS (PERBAIKAN KONTRAS TEKS & OVERRIDE DARK/LIGHT MODE)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Force App Background */
    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Force Sidebar Style */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    }

    /* Headings Kontras Tinggi */
    h1 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    p, li, span, label {
        color: #334155 !important;
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

    .step-card h4 {
        color: #0F172A !important;
        margin-top: 6px;
        margin-bottom: 6px;
    }

    .step-card p {
        color: #475569 !important;
        font-size: 13px;
        margin: 0;
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

    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #059669 !important;
    }

    /* Expander Text Fix */
    .streamlit-expanderHeader {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# LOAD MODEL & CLASS INFO
# ------------------------------------------------------------------------------
@st.cache_resource
def load_apple_model():
    try:
        model = tf.keras.models.load_model("apple_disease_model.h5")
        return model
    except Exception:
        return None


model = load_apple_model()

CLASS_NAMES = [
    "Anthracnose",
    "Black Pox",
    "Black Rot",
    "Codling Moth",
    "Healthy",
    "Powdery Mildew",
]

CLASS_DETAILS = {
    "Anthracnose": {
        "title": "Antraknosa (Anthracnose)",
        "desc": "Disebabkan oleh jamur Colletotrichum. Mengakibatkan lesi bercak cokelat kehitaman yang cekung pada kulit buah apel.",
        "prevention": "Pangkas bagian tanaman terinfeksi, jaga sirkulasi udara di tajuk pohon, dan aplikasikan fungisida tembaga.",
    },
    "Black Pox": {
        "title": "Cacar Hitam (Black Pox)",
        "desc": "Disebabkan oleh Helminthosporium papulosum yang menimbulkan bintik-bintik hitam menonjol kasar pada permukaan buah.",
        "prevention": "Bersihkan kebun dari guguran buah tua dan semprotkan fungisida pelindung saat awal musim.",
    },
    "Black Rot": {
        "title": "Busuk Hitam (Black Rot)",
        "desc": "Infeksi oleh Botryosphaeria obtusa yang memicu pembusukan melingkar hingga buah mengering dan mengerut hitam.",
        "prevention": "Kumpulkan dan musnahkan buah yang mengering (mummy apple) serta hindari luka fisik pada buah.",
    },
    "Codling Moth": {
        "title": "Hama Ulat Apel (Codling Moth)",
        "desc": "Kerusakan akibat larva Cydia pomonella yang menggerogoti hingga membuat lorong ke bagian dalam dan biji buah.",
        "prevention": "Gunakan perangkap feromon, pasang pembungkus buah (fruit bagging), atau gunakan insektisida hayati.",
    },
    "Healthy": {
        "title": "Buah Apel Sehat (Healthy)",
        "desc": "Buah apel dalam kondisi segar, permukaan mulus, dan bebas dari tanda infeksi patogen atau hama.",
        "prevention": "Pertahankan pemeliharaan rutin, pemupukan berimbang, dan pengairan yang teratur.",
    },
    "Powdery Mildew": {
        "title": "Embun Tepung (Powdery Mildew)",
        "desc": "Disebabkan oleh jamur Podosphaera leucotricha yang melapisi permukaan kulit buah atau daun dengan lapisan seperti serbuk putih.",
        "prevention": "Gunakan varietas tahan jamur, pangkas tunas terinfeksi, dan aplikasikan fungisida sulfur.",
    },
}


def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.markdown("## 🍎 **Apple AI System**")
st.sidebar.caption("Klasifikasi Penyakit Buah Apel v1.0")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi Halaman:",
    ["🏠 Home", "🤖 Deteksi Penyakit"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size: 12px; color: #64748B;'>
        <b>Model:</b> MobileNetV2<br>
        <b>Batas Keyakinan:</b> 70.0%<br>
        <b>Status Sistem:</b> Aktif
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# HALAMAN 1: HOME
# ------------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("Sistem Klasifikasi Penyakit Buah Apel")
    st.write(
        "Aplikasi cerdas berbasis Deep Learning untuk mendeteksi kondisi kesehatan buah apel secara presisi."
    )
    st.markdown("---")

    # 1. Penjelasan Aplikasi
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="pro-card-header"><h3>📖 Penjelasan Aplikasi</h3></div>',
        unsafe_allow_html=True,
    )
    st.write(
        """
        Aplikasi ini dirancang untuk mengidentifikasi penyakit pada buah apel (*Malus domestica*) secara otomatis. 
        Menggunakan metode **Deep Learning** dengan arsitektur **MobileNetV2**, sistem dapat mengenali apakah buah apel dalam kondisi **Sehat** atau terinfeksi salah satu dari **5 jenis penyakit/hama** (Antraknosa, Cacar Hitam, Busuk Hitam, Ulat Apel, atau Embun Tepung).
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Tujuan Aplikasi
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="pro-card-header"><h3>🎯 Tujuan Aplikasi</h3></div>',
        unsafe_allow_html=True,
    )
    st.write(
        """
        - **Membantu Petani & Pengelola Kebun:** Mempercepat proses diagnosa penyakit di lapangan tanpa harus menunggu ahli.
        - **Mencegah Kerugian:** Memberikan saran penanganan awal yang tepat untuk mencegah penularan penyakit ke seluruh kebun.
        - **Penyaringan Gambar Cerdas:** Dilengkapi *Confidence Threshold* 60% untuk menolak gambar non-apel atau gambar yang terlalu buram.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Cara Penggunaan
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="pro-card-header"><h3>💡 Cara Penggunaan</h3></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">1</div>
                <h4>Pilih Menu Deteksi</h4>
                <p>Klik menu <b>🤖 Deteksi Penyakit</b> pada bilah navigasi di sebelah kiri.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">2</div>
                <h4>Unggah Gambar</h4>
                <p>Unggah foto buah apel pada panel kiri (format JPG atau PNG).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">3</div>
                <h4>Lihat Hasil</h4>
                <p>Sistem menampilkan hasil diagnosa, persentase probabilitas, dan saran solusi.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HALAMAN 2: DETEKSI PENYAKIT (TEACHABLE MACHINE LAYOUT)
# ------------------------------------------------------------------------------
elif page == "🤖 Deteksi Penyakit":
    st.title("Deteksi & Klasifikasi Penyakit (Real-time)")
    st.write(
        "Unggah gambar pada panel kiri untuk melihat hasil diagnosa dan probabilitas kelas pada panel kanan."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    # -------------------------- KOLOM KIRI: INPUT --------------------------
    with col1:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="pro-card-header"><h3>📷 Input Gambar</h3></div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Pilih file gambar buah apel (JPG, JPEG, PNG)...",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(
                image, caption="Gambar yang Diunggah", use_container_width=True
            )
        else:
            st.info(
                "Silakan unggah gambar buah apel di atas untuk memulai analisis."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- KOLOM KANAN: OUTPUT --------------------------
    with col2:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="pro-card-header"><h3>📊 Hasil Diagnosa</h3></div>',
            unsafe_allow_html=True,
        )

        if uploaded_file is not None:
            if model is None:
                st.error("Model `apple_disease_model.h5` tidak ditemukan!")
            else:
                with st.spinner("Menganalisis gambar..."):
                    processed_img = preprocess_image(image)
                    predictions = model.predict(processed_img)[0]
                    top_idx = int(np.argmax(predictions))

                    predicted_class = CLASS_NAMES[top_idx]
                    confidence = float(predictions[top_idx]) * 100

                CONFIDENCE_THRESHOLD = 60.0

                if confidence < CONFIDENCE_THRESHOLD:
                    # KONDISI TIDAK TERDETEKSI (< 60%)
                    st.error("❌ **Hasil Diagnosa: Tidak Terdeteksi**")
                    st.warning(
                        f"Keyakinan tertinggi hanya **{confidence:.2f}%** (di bawah syarat 60.0%). "
                        "Gambar tidak dikenali sebagai buah apel yang valid."
                    )

                    st.markdown("---")
                    st.markdown(
                        "<h4 style='color: #0F172A;'>Probabilitas Seluruh Kelas:</h4>",
                        unsafe_allow_html=True,
                    )
                    for c_name in CLASS_NAMES:
                        st.write(f"**{c_name}**: 0.0%")
                        st.progress(0.0)

                else:
                    # KONDISI TERDETEKSI (>= 60%)
                    st.success(f"🏆 **Hasil Diagnosa: {predicted_class}**")
                    st.info(f"🎯 **Confidence:** {confidence:.2f}%")

                    st.markdown("---")
                    st.markdown(
                        "<h4 style='color: #0F172A;'>Probabilitas Seluruh Kelas:</h4>",
                        unsafe_allow_html=True,
                    )

                    for i, c_name in enumerate(CLASS_NAMES):
                        prob = float(predictions[i])
                        prob_percent = prob * 100
                        st.write(f"**{c_name}**: {prob_percent:.1f}%")
                        st.progress(prob)

                    st.markdown("---")
                    detail = CLASS_DETAILS.get(predicted_class, {})
                    if detail:
                        with st.expander(
                            "📌 Deskripsi Penyakit & Solusi Penanganan",
                            expanded=True,
                        ):
                            st.markdown(f"**{detail.get('title', '')}**")
                            st.write(detail.get("desc", ""))
                            st.markdown("**Langkah Penanganan:**")
                            st.write(detail.get("prevention", ""))
        else:
            st.write(
                "Hasil diagnosa dan bar probabilitas akan muncul di sini setelah Anda mengunggah gambar."
            )

        st.markdown("</div>", unsafe_allow_html=True)
