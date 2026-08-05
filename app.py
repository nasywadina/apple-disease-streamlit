from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apple Disease Classifier AI",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed",  # Menyembunyikan sidebar agar layar lebih fokus
)

# ------------------------------------------------------------------------------
# 2. CUSTOM CSS UNTUK TAMPILAN MODERN & MENARIK
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
    }

    /* Main Container Styling */
    .main-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }

    /* Hero Banner Header */
    .hero-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    
    .hero-header h1 {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        margin-bottom: 8px;
    }

    .hero-header p {
        color: #64748B !important;
        font-size: 1.05rem;
    }

    /* Section Titles */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 16px;
    }

    /* Custom Result Card */
    .result-badge-success {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 16px 20px;
        color: #065F46;
        margin-bottom: 20px;
    }

    .result-badge-danger {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 16px 20px;
        color: #991B1B;
        margin-bottom: 20px;
    }

    .result-badge-warning {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 16px 20px;
        color: #92400E;
        margin-bottom: 20px;
    }

    /* Progress bar custom styling */
    .stProgress > div > div > div > div {
        background-color: #059669 !important;
        border-radius: 10px;
    }

    /* Hide default Streamlit padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 3. LOAD MODEL & DATA REFERENSI
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
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ------------------------------------------------------------------------------
# 4. HEADER UTAMA (HERO BANNER)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <h1>🍎 Apple Health AI</h1>
        <p>Deteksi Otomatis & Analisis Kesehatan Buah Apel Secara Real-Time</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Panduan Ringkas & Interaktif
with st.expander(
    "💡 **Cara Menggunakan Aplikasi (Klik untuk Buka)**", expanded=False
):
    st.write(
        """
    1. **Pilih Metode Input:** Gunakan kamera *real-time* atau unggah gambar buah apel dari galeri Anda.
    2. **Analisis Otomatis:** Sistem AI akan memproses gambar dan menghitung tingkat kesehatan apel secara presisi.
    3. **Lihat Solusi:** Jika terdeteksi penyakit, lihat petunjuk penanganan yang direkomendasikan ahli.
    """
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. STEP 1: INPUT GAMBAR (UPLOAD & REALTIME CAMERA)
# ------------------------------------------------------------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📸 <span>Pilih atau Ambil Foto Buah Apel</span></div>',
    unsafe_allow_html=True,
)

# Pilihan Tab untuk Pengalaman Pengguna yang Lebih Menarik
tab_upload, tab_camera = st.tabs(
    ["📁 Upload Gambar (Galeri)", "📷 Real-time Kamera (Webcam)"]
)

image_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Unggah gambar apel di sini (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="file_uploader",
    )
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)

with tab_camera:
    camera_file = st.camera_input(
        "Arahkan kamera ke buah apel & ambil foto", key="camera_input"
    )
    if camera_file is not None:
        image_source = Image.open(camera_file)

# Tampilan Preview Gambar yang Diunggah/Diambil
if image_source is not None:
    st.markdown("---")
    st.markdown("##### 🖼️ Gambar yang Dipilih:")
    st.image(
        image_source, use_container_width=True, caption="Objek Buah Apel"
    )

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. STEP 2: HASIL DIAGNOSA & ANALISIS PROBABILITAS
# ------------------------------------------------------------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📊 <span>Hasil Analisis AI</span></div>',
    unsafe_allow_html=True,
)

if image_source is not None:
    if model is None:
        st.error(
            "⚠️ Model AI `apple_disease_model.h5` tidak ditemukan. Pastikan file model sudah berada di folder proyek."
        )
    else:
        with st.spinner("⚡ AI sedang menganalisis tingkat kesehatan apel..."):
            processed_img = preprocess_image(image_source)
            predictions = model(processed_img, training=False).numpy()[0]
            top_idx = int(np.argmax(predictions))

            predicted_class = CLASS_NAMES[top_idx]
            confidence = float(predictions[top_idx]) * 100

        CONFIDENCE_THRESHOLD = 70.0

        if confidence < CONFIDENCE_THRESHOLD:
            # 1. KONDISI TIDAK TERDETEKSI (< 70%)
            st.markdown(
                f"""
                <div class="result-badge-warning">
                    <h3 style="margin:0; color:#92400E;">❓ Tidak Terdeteksi Sebagai Apel Valid</h3>
                    <p style="margin:5px 0 0 0;">Tingkat keyakinan tertinggi hanya <b>{confidence:.1f}%</b> (Di bawah standar minimal 70.0%). Pastikan objek adalah buah apel dengan pencahayaan yang jelas.</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("#### 📈 Probabilitas Diagnosa:")
            for c_name in CLASS_NAMES:
                col_name, col_val = st.columns([3, 1])
                with col_name:
                    st.write(f"**{c_name}**")
                with col_val:
                    st.write("0.0%")
                st.progress(0.0)

        else:
            # 2. KONDISI TERDETEKSI (>= 70%)
            if predicted_class == "Healthy":
                st.markdown(
                    f"""
                    <div class="result-badge-success">
                        <h3 style="margin:0; color:#065F46;">✅ Buah Apel Sehat (Healthy)</h3>
                        <p style="margin:5px 0 0 0;">Tingkat Akurasi / Keyakinan AI: <b>{confidence:.2f}%</b></p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-badge-danger">
                        <h3 style="margin:0; color:#991B1B;">⚠️ Terinfeksi: {predicted_class}</h3>
                        <p style="margin:5px 0 0 0;">Tingkat Akurasi / Keyakinan AI: <b>{confidence:.2f}%</b></p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            # Visualisasi Probabilitas Semua Kelas
            st.markdown("#### 📈 Rincian Probabilitas Semua Kelas:")

            for i, c_name in enumerate(CLASS_NAMES):
                prob = float(predictions[i])
                prob_percent = prob * 100

                col_txt, col_pct = st.columns([3, 1])
                with col_txt:
                    st.write(f"**{c_name}**")
                with col_pct:
                    st.write(f"**{prob_percent:.1f}%**")

                st.progress(prob)

            # Informasi Penyakit & Penanganan
            detail = CLASS_DETAILS.get(predicted_class, {})
            if detail:
                st.markdown("---")
                st.markdown("### 📌 Penjelasan & Langkah Penanganan")
                st.info(
                    f"**{detail.get('title', '')}**\n\n{detail.get('desc', '')}"
                )

                st.success(
                    f"**💡 Solusi & Pencegahan:**\n\n{detail.get('prevention', '')}"
                )
else:
    st.write(
        " Silakan unggah gambar atau gunakan kamera di atas. Hasil diagnosa dan saran penanganan akan langsung ditampilkan di bagian ini."
    )

st.markdown("</div>", unsafe_allow_html=True)
