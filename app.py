from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# Config Halaman Streamlit
st.set_page_config(
    page_title="Pendeteksi Penyakit Buah Apel", page_icon="🍎", layout="wide"
)

# Custom CSS agar tampilan lebih mirip Teachable Machine / Modern Dashboard
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title & Header
st.title("🍎 Teachable Machine: Klasifikasi Penyakit Buah Apel")
st.write(
    "Unggah gambar buah apel untuk mengidentifikasi kondisi/penyakit secara otomatis berbasis Deep Learning (MobileNetV2)."
)
st.markdown("---")

# Load Model
@st.cache_resource
def load_apple_model():
    try:
        model = tf.keras.models.load_model("apple_disease_model.h5")
        return model
    except Exception as e:
        return None

model = load_apple_model()

# Daftar Kelas
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
        "desc": "Disebabkan oleh jamur. Menyebabkan bercak cokelat kehitaman yang cekung pada kulit buah apel.",
        "prevention": "Pangkas bagian tanaman yang terinfeksi, jaga sirkulasi udara, dan semprotkan fungisida berbasis tembaga.",
    },
    "Black Pox": {
        "title": "Cacar Hitam (Black Pox)",
        "desc": "Infeksi jamur yang menimbulkan bintik-bintik hitam kecil sedikit menonjol pada permukaan buah.",
        "prevention": "Bersihkan sisa-sisa buah yang gugur di tanah dan gunakan fungisida pelindung saat awal musim.",
    },
    "Black Rot": {
        "title": "Busuk Hitam (Black Rot)",
        "desc": "Penyakit serius yang membuat buah membusuk cokelat melingkar hingga mengering hitam menyerupai mumi.",
        "prevention": "Buang buah yang terinfeksi dari area kebun dan lakukan penyemprotan fungisida secara berkala.",
    },
    "Codling Moth": {
        "title": "Hama Ulat Apel (Codling Moth)",
        "desc": "Kerusakan akibat larva ngat yang menggerogoti hingga ke dalam inti/biji buah apel.",
        "prevention": "Gunakan perangkap feromon, pasang pembungkus buah, atau aplikasikan insektisida hayati.",
    },
    "Healthy": {
        "title": "Buah Apel Sehat (Healthy)",
        "desc": "Buah apel dalam kondisi segar, bersih dari bercak jamur maupun kerusakan akibat hama.",
        "prevention": "Pertahankan perawatan rutin, pemupukan seimbang, dan pengairan yang teratur.",
    },
    "Powdery Mildew": {
        "title": "Embun Tepung (Powdery Mildew)",
        "desc": "Lapisan putih menyerupai tepung pada permukaan kulit buah atau daun akibat infeksi jamur.",
        "prevention": "Gunakan varietas tahan jamur dan aplikasikan fungisida sulfur secara teratur.",
    },
}

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================================================================
# TAMPILAN DUA KOLOM (ALA TEACHABLE MACHINE)
# ==============================================================================
col1, col2 = st.columns([1, 1], gap="large")

# -------------------------- KOLOM KIRI: INPUT --------------------------
with col1:
    st.subheader("📷 Input Gambar")
    uploaded_file = st.file_uploader(
        "Pilih file gambar (JPG/PNG)...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Input", use_container_width=True)
    else:
        st.info("Silakan unggah gambar buah apel pada tombol di atas.")

# -------------------------- KOLOM KANAN: OUTPUT --------------------------
with col2:
    st.subheader("📊 Output Prediksi (Output Class)")

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

            # THRESHOLDING (60%)
            CONFIDENCE_THRESHOLD = 60.0

            if confidence < CONFIDENCE_THRESHOLD:
                # Jika tidak terdeteksi / di bawah threshold
                st.error("❌ **Hasil Diagnosa: Tidak Terdeteksi**")
                st.warning(
                    f"Confidence tertinggi hanya **{confidence:.2f}%** (di bawah batas minimal 60%). "
                    "Gambar tidak dikenali sebagai objek buah apel yang valid."
                )

                st.markdown("---")
                st.write("**Probabilitas Seluruh Kelas:**")
                # Tampilkan progress bar 0% untuk semua kelas
                for c_name in CLASS_NAMES:
                    st.write(f"**{c_name}**: 0.0%")
                    st.progress(0.0)

            else:
                # Jika berhasil terdeteksi (>= 60%)
                st.success(f"🏆 **Hasil Diagnosa: {predicted_class}**")
                st.info(f"🎯 **Confidence:** {confidence:.2f}%")

                st.markdown("---")
                st.write("**Probabilitas Seluruh Kelas:**")

                # Tampilkan progress bar ala Teachable Machine
                for i, c_name in enumerate(CLASS_NAMES):
                    prob = float(predictions[i])
                    prob_percent = prob * 100

                    # Tampilkan nama kelas & persentase
                    st.write(f"**{c_name}**: {prob_percent:.1f}%")
                    st.progress(prob)

                # Detail & Penanganan Penyakit
                st.markdown("---")
                detail = CLASS_DETAILS.get(predicted_class, {})
                if detail:
                    with st.expander(
                        "📌 Deskripsi & Solusi Penanganan", expanded=True
                    ):
                        st.markdown(f"**{detail.get('title', '')}**")
                        st.write(detail.get("desc", ""))
                        st.markdown("**Cara Penanganan:**")
                        st.write(detail.get("prevention", ""))
    else:
        st.write("Hasil prediksi akan muncul di sini setelah gambar diunggah.")
