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
    initial_sidebar_state="collapsed",  # Tanpa sidebar agar fokus pada scroll halaman utama
)

# ------------------------------------------------------------------------------
# 2. CUSTOM CSS (VISUAL MODERN, BORDER CARD, & STYLING WARNA)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Styling Card Utama */
    .app-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Badge Diagnosa */
    .badge-success {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 16px;
        color: #065F46;
    }

    .badge-danger {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 16px;
        color: #991B1B;
    }

    .badge-warning {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 16px;
        color: #92400E;
    }

    /* Custom Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #059669 !important;
        border-radius: 8px;
    }

    /* Mengatur jarak top padding */
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
# 4. URUTAN 1: HEADER & PETUNJUK
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A;">🍎 Apple Health AI System</h1>
        <p style="color: #64748B; font-size: 1rem;">Sistem Deteksi Penyakit Buah Apel Berbasis Computer Vision (MobileNetV2)</p>
    </div>
""",
    unsafe_allow_html=True,
)

with st.expander(
    "📖 **Petunjuk Penggunaan Aplikasi (Klik untuk Buka)**", expanded=False
):
    st.write(
        """
        1. **Pilih Metode Input:** Gunakan opsi **Upload File** untuk memilih gambar dari galeri atau **Real-time Kamera** untuk mengambil foto langsung.
        2. **Gunakan Gambar yang Jelas:** Pastikan pencahayaan cukup dan objek buah apel berada di tengah frame.
        3. **Gulir (Scroll) ke Bawah:** Setelah gambar dimasukkan, hasil diagnosa, grafik probabilitas, dan solusi penanganan akan langsung ditampilkan di bagian bawah.
    """
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. URUTAN 2: INPUT GAMBAR (UPLOAD / CAMERA REALTIME)
# ------------------------------------------------------------------------------
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown("### 📷 **1. Input Gambar Buah Apel**")

tab_upload, tab_camera = st.tabs(
    ["📁 Upload Gambar (JPG/PNG)", "📸 Real-time Kamera (Webcam)"]
)

image_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Unggah foto buah apel di sini...",
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

if image_source is not None:
    st.markdown("---")
    st.markdown("##### 🖼️ **Preview Gambar:**")
    st.image(
        image_source, use_container_width=True, caption="Objek Terpilih"
    )

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. URUTAN 3 & 4: HASIL DIAGNOSA & DETAIL PROBABILITAS
# ------------------------------------------------------------------------------
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown("### 📊 **2. Hasil Diagnosa AI**")

if image_source is not None:
    if model is None:
        st.error(
            "⚠️ Model `apple_disease_model.h5` tidak ditemukan pada direktori utama."
        )
    else:
        with st.spinner("Analyzing image..."):
            processed_img = preprocess_image(image_source)
            predictions = model(processed_img, training=False).numpy()[0]
            top_idx = int(np.argmax(predictions))

            predicted_class = CLASS_NAMES[top_idx]
            confidence = float(predictions[top_idx]) * 100

        CONFIDENCE_THRESHOLD = 70.0

        # --- CONDITION 1: UNKNOWN / LOW CONFIDENCE (< 70%) ---
        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown(
                f"""
                <div class="badge-warning">
                    <h3 style="margin:0;">❓ Status: Tidak Terdeteksi</h3>
                    <p style="margin:5px 0 0 0;">Keyakinan terdeteksi hanya <b>{confidence:.1f}%</b> (Di bawah syarat minimal 70.0%). Pastikan foto fokus dan merupakan buah apel yang valid.</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>#### 📈 Probabilitas Seluruh Kelas:", unsafe_allow_html=True)
            for c_name in CLASS_NAMES:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"**{c_name}**")
                with c2:
                    st.write("0.0%")
                st.progress(0.0)

        # --- CONDITION 2: SUCCESS DETECTED (>= 70%) ---
        else:
            if predicted_class == "Healthy":
                st.markdown(
                    f"""
                    <div class="badge-success">
                        <h3 style="margin:0;">✅ Status: {predicted_class} (Sehat)</h3>
                        <p style="margin:5px 0 0 0;">Tingkat Akurasi / Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="badge-danger">
                        <h3 style="margin:0;">⚠️ Status Terinfeksi: {predicted_class}</h3>
                        <p style="margin:5px 0 0 0;">Tingkat Akurasi / Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>#### 📈 Persentase Probabilitas Semua Kelas:", unsafe_allow_html=True)
            for i, c_name in enumerate(CLASS_NAMES):
                prob = float(predictions[i])
                prob_percent = prob * 100

                col_label, col_val = st.columns([3, 1])
                with col_label:
                    st.write(f"**{c_name}**")
                with col_val:
                    st.write(f"**{prob_percent:.1f}%**")

                st.progress(prob)

            # ------------------------------------------------------------------
            # 7. URUTAN 5: PENJELASAN & PENANGANAN PENYAKIT (PALING BAWAH)
            # ------------------------------------------------------------------
            detail = CLASS_DETAILS.get(predicted_class, {})
            if detail:
                st.markdown("---")
                st.markdown("### 📌 **3. Detail Penyakit & Solusi Penanganan**")
                
                st.info(
                    f"**{detail.get('title', '')}**\n\n{detail.get('desc', '')}"
                )

                st.success(
                    f"**💡 Langkah Penanganan & Pencegahan:**\n\n{detail.get('prevention', '')}"
                )

else:
    st.write(
        " Silakan masukkan gambar di bagian atas. Hasil analisis lengkap akan ditampilkan secara otomatis pada bagian ini."
    )

st.markdown("</div>", unsafe_allow_html=True)
