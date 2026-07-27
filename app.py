import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="AppleScan - Deteksi Penyakit Apel",
    page_icon="🍎",
    layout="centered",
)

st.title("🍎 AppleScan: Sistem Deteksi Penyakit Buah Apel")
st.write(
    "Unggah gambar buah apel untuk mengidentifikasi jenis penyakit secara otomatis."
)

# ==============================================================================
# LOAD MODEL & DATA DETAIL PENYAKIT
# ==============================================================================


@st.cache_resource
def load_apple_model():
    model_path = "apple_disease_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
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
        "title": "Anthracnose (Antraknosa)",
        "desc": "Infeksi jamur yang menyebabkan lesi/bercak melingkar cekung berwarna cokelat gelap hingga hitam.",
        "prevention": "Pangkas bagian terinfeksi, jaga sirkulasi udara, dan gunakan fungisida berbasis tembaga.",
    },
    "Black Pox": {
        "title": "Black Pox (Cacar Hitam)",
        "desc": "Bintik-bintik hitam kecil menonjol pada kulit buah akibat Helminthosporium papulosum.",
        "prevention": "Pangkas rutin mengurangi kelembapan dan gunakan fungisida pelindung.",
    },
    "Black Rot": {
        "title": "Black Rot (Busuk Hitam)",
        "desc": "Buah membusuk, berwarna hitam gelap, dan mengering menyerupai mumi.",
        "prevention": "Bersihkan sisa buah mumi, pangkas dahan mati, dan sanitasi lahan.",
    },
    "Codling Moth": {
        "title": "Codling Moth (Hama Ngengat Apel)",
        "desc": "Kerusakan akibat larva Cydia pomonella yang menggerek ke dalam buah hingga ke biji.",
        "prevention": "Pasang perangkap feromon, bungkus buah muda, dan gunakan insektisida hayati.",
    },
    "Healthy": {
        "title": "Healthy (Buah Sehat)",
        "desc": "Buah dalam kondisi segar tanpa noda dan bebas indikasi jamur maupun hama.",
        "prevention": "Pertahankan penyiraman teratur, pemupukan seimbang, dan pemantauan berkala.",
    },
    "Powdery Mildew": {
        "title": "Powdery Mildew (Embun Tepung)",
        "desc": "Bercak putih bertepung pada permukaan kulit dan daun akibat Podosphaera leucotricha.",
        "prevention": "Pangkas tunas terinfeksi di awal musim, gunakan fungisida sulfur.",
    },
}


def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# ==============================================================================
# INTERFACE UPLOAD & PREDIKSI
# ==============================================================================
uploaded_file = st.file_uploader(
    "Pilih gambar apel (JPG/PNG)...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    if model is None:
        st.error(
            "File model `apple_disease_model.h5` tidak ditemukan di direktori!"
        )
    else:
        with st.spinner("Menganalisis gambar..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)[0]
            top_idx = int(np.argmax(predictions))

            predicted_class = CLASS_NAMES[top_idx]
            confidence = float(predictions[top_idx]) * 100

        # ======================================================================
        # TAMBAHKAN THRESHOLDING DI SINI
        # ======================================================================
        CONFIDENCE_THRESHOLD = 70.0  # Batas minimal keyakinan (70%)

        if confidence < CONFIDENCE_THRESHOLD:
            # Jika tingkat keyakinan di bawah 60%
            st.warning("⚠️ **Gambar Tidak Terdeteksi / Kurang Jelas**")
            st.error(
                f"Tingkat keyakinan tertinggi hanya **{confidence:.2f}%** "
                f"(di bawah batas minimal {CONFIDENCE_THRESHOLD}%).\n\n"
                "Sistem tidak dapat mengidentifikasi penyakit secara pasti. "
                "Harap unggah gambar buah apel yang lebih jelas dan dekat."
            )
        else:
            # Jika lulus thresholding (Keyakinan >= 60%)
            st.success(f"**Hasil Diagnosa:** {predicted_class}")
            st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")

            detail = CLASS_DETAILS.get(predicted_class, {})
            if detail:
                with st.expander(
                    "📌 Detail Penyakit & Solusi Penanganan", expanded=True
                ):
                    st.markdown(f"**{detail.get('title', '')}**")
                    st.write(detail.get("desc", ""))
                    st.markdown("**Cara Pencegahan / Penanganan:**")
                    st.write(detail.get("prevention", ""))

        # Menampilkan grafik probabilitas untuk semua kelas (tetap ada)
        st.write("---")
        st.subheader("Probabilitas Seluruh Kelas")
        prob_dict = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }
        st.bar_chart(prob_dict)
        
        # Tampilkan Hasil
        st.success(f"**Hasil Diagnosa:** {predicted_class}")
        st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")

        detail = CLASS_DETAILS.get(predicted_class, {})
        if detail:
            with st.expander("📌 Detail Penyakit & Solusi Penanganan", expanded=True):
                st.markdown(f"**{detail.get('title', '')}**")
                st.write(detail.get("desc", ""))
                st.markdown("**Cara Pencegahan / Penanganan:**")
                st.write(detail.get("prevention", ""))

        st.write("---")
        st.subheader("Probabilitas Seluruh Kelas")
        prob_dict = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }
        st.bar_chart(prob_dict)
