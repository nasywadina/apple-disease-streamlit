from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apple Disease Classifier (SPA)",
    page_icon="🍎",
    layout="centered",  # Menggunakan layout centered agar tampilan rapi saat di-scroll
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 2. CUSTOM CSS (STYLING MODERN & ELEGAN)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

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

    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .pro-card-header {
        border-left: 4px solid #059669;
        padding-left: 12px;
        margin-bottom: 20px;
    }

    .stProgress > div > div > div > div {
        background-color: #059669 !important;
    }

    .streamlit-expanderHeader {
        color: #0F172A !important;
        font-weight: 600 !important;
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
# 4. SIDEBAR INFORMASI & PENGATURAN
# ------------------------------------------------------------------------------
st.sidebar.markdown("## 🍎 **Apple AI System**")
st.sidebar.caption("Klasifikasi Penyakit Buah Apel v2.0 (SPA)")
st.sidebar.markdown("---")

input_mode = st.sidebar.radio(
    "📷 **Pilih Metode Input Gambar:**",
    ["📁 Upload File (JPG/PNG)", "📸 Real-time Kamera (Webcam)"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size: 12px; color: #64748B;'>
        <b>Arsitektur:</b> MobileNetV2<br>
        <b>Model Type:</b> Transfer Learning<br>
        <b>Batas Keyakinan:</b> 70.0%<br>
        <b>Layout:</b> Single Page Scroll (SPA)
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 5. SINGLE PAGE APPLICATION (VERTIKAL / SCROLL LAYOUT)
# ------------------------------------------------------------------------------

# --- HEADER & DESKRIPSI UTAMA ---
st.title("🍎 Sistem Klasifikasi Penyakit Buah Apel")
st.write(
    "Deteksi kondisi kesehatan buah apel secara presisi menggunakan Deep Learning (MobileNetV2)."
)

with st.expander(
    "📖 **Panduan Sistem & Penjelasan Aplikasi (Klik untuk Buka/Tutup)**",
    expanded=False,
):
    st.markdown("#### 🎯 **Tujuan Aplikasi**")
    st.write(
        "- Membantu petani & pengelola kebun mendiagnosa penyakit apel secara mandiri & cepat.\n"
        "- Menyarankan langkah penanganan dini berbasis ahli patologi tanaman.\n"
        "- Menyeleksi gambar tak valid/buram melalui *Confidence Threshold* **70%**."
    )
    st.markdown("#### 💡 **Cara Penggunaan**")
    st.write(
        "1. Pilih metode input di **Sidebar** (Upload File atau Kamera Real-time).\n"
        "2. Unggah gambar atau tangkap foto buah apel pada bagian atas.\n"
        "3. *Scroll* ke bawah untuk melihat hasil diagnosa dan solusi penanganan."
    )

st.markdown("---")

# ==================== 1. BAGIAN ATAS: INPUT GAMBAR ====================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="pro-card-header"><h3>📷 Input Gambar</h3></div>',
    unsafe_allow_html=True,
)

image_source = None

if input_mode == "📁 Upload File (JPG/PNG)":
    uploaded_file = st.file_uploader(
        "Pilih foto buah apel (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"],
        key="file_uploader",
    )
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)
        st.image(
            image_source, caption="Gambar yang Diunggah", use_container_width=True
        )
    else:
        st.info("Silakan unggah gambar buah apel untuk memulai analisa.")

elif input_mode == "📸 Real-time Kamera (Webcam)":
    camera_file = st.camera_input(
        "Arahkan kamera ke buah apel & ambil foto:", key="camera_input"
    )
    if camera_file is not None:
        image_source = Image.open(camera_file)
        st.image(
            image_source,
            caption="Hasil Tangkapan Kamera Real-time",
            use_container_width=True,
        )
    else:
        st.info("Izinkan akses kamera pada browser Anda lalu ambil foto objek.")

st.markdown("</div>", unsafe_allow_html=True)

# ==================== 2. BAGIAN BAWAH: HASIL DIAGNOSA ====================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="pro-card-header"><h3>📊 Hasil Diagnosa</h3></div>',
    unsafe_allow_html=True,
)

if image_source is not None:
    if model is None:
        st.error(
            "Model `apple_disease_model.h5` tidak ditemukan pada direktori utama!"
        )
    else:
        with st.spinner("Menganalisis gambar..."):
            processed_img = preprocess_image(image_source)
            predictions = model(processed_img, training=False).numpy()[0]
            top_idx = int(np.argmax(predictions))

            predicted_class = CLASS_NAMES[top_idx]
            confidence = float(predictions[top_idx]) * 100

        CONFIDENCE_THRESHOLD = 70.0

        if confidence < CONFIDENCE_THRESHOLD:
            # KONDISI TIDAK TERDETEKSI (< 70%)
            st.error("❌ **Hasil Diagnosa: Tidak Terdeteksi**")
            st.warning(
                f"Keyakinan tertinggi hanya **{confidence:.2f}%** (di bawah syarat 70.0%). "
                "Gambar tidak dikenali sebagai buah apel yang valid atau kualitas terlalu buram."
            )

            st.markdown("---")
            st.markdown("#### Probabilitas Seluruh Kelas:")
            for c_name in CLASS_NAMES:
                st.write(f"**{c_name}**: 0.0%")
                st.progress(0.0)

        else:
            # KONDISI TERDETEKSI (>= 70%)
            st.success(f"🏆 **Hasil Diagnosa: {predicted_class}**")
            st.info(f"🎯 **Confidence:** {confidence:.2f}%")

            st.markdown("---")
            st.markdown("#### Probabilitas Seluruh Kelas:")

            for i, c_name in enumerate(CLASS_NAMES):
                prob = float(predictions[i])
                prob_percent = prob * 100
                st.write(f"**{c_name}**: {prob_percent:.1f}%")
                st.progress(prob)

            st.markdown("---")
            detail = CLASS_DETAILS.get(predicted_class, {})
            if detail:
                with st.expander(
                    "📌 **Deskripsi Penyakit & Solusi Penanganan**",
                    expanded=True,
                ):
                    st.markdown(f"**{detail.get('title', '')}**")
                    st.write(detail.get("desc", ""))
                    st.markdown("**Langkah Penanganan:**")
                    st.write(detail.get("prevention", ""))
else:
    st.write(
        "Hasil diagnosa, probabilitas kelas, dan solusi penanganan akan muncul di sini setelah Anda memasukkan gambar di atas."
    )

st.markdown("</div>", unsafe_allow_html=True)
