from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------------------------
# CONFIGURASI HALAMAN
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apple Diagnostic AI - Sistem Deteksi Penyakit Apel",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# CUSTOM CSS (DESAIN PROFESIONAL & PALET WARNA TEAL-NAVY)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Background Utama */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header & Title Styling */
    h1 {
        color: #0F5257 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #0B2545 !important;
        font-weight: 600 !important;
    }
    
    /* Custom Card Style */
    .pro-card {
        background-color: #FFFFFF;
        padding: 28px;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.02);
        margin-bottom: 24px;
        border: 1px solid #E2E8F0;
    }

    .pro-card-header {
        border-left: 4px solid #0F5257;
        padding-left: 12px;
        margin-bottom: 18px;
    }
    
    .step-card {
        background-color: #F1F5F9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        height: 100%;
    }

    .step-number {
        display: inline-block;
        background-color: #0F5257;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 10px;
    }

    /* Custom Progress Bar Color */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0F5257 0%, #0B2545 100%);
        border-radius: 8px;
    }

    /* Sidebar Custom Style */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Streamlit Expander Style Fix */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #0F5257;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# LOAD MODEL & DATASET CLASS INFO
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
        "desc": "Penyakit akibat infeksi jamur Colletotrichum. Mengakibatkan lesi bercak cokelat kehitaman yang mencekung pada permukaan kulit buah apel.",
        "prevention": "Lakukan pemangkasan bagian terinfeksi, tingkatkan sirkulasi udara di tajuk pohon, dan aplikasikan fungisida berbasis tembaga.",
    },
    "Black Pox": {
        "title": "Cacar Hitam (Black Pox)",
        "desc": "Disebabkan oleh jamur Helminthosporium papulosum yang menimbulkan pustul/bintik hitam menonjol dan kasar pada buah.",
        "prevention": "Jaga sanitasi kebun dari guguran buah tua, serta semprotkan fungisida pelindung secara teratur pada awal musim pembentukan buah.",
    },
    "Black Rot": {
        "title": "Busuk Hitam (Black Rot)",
        "desc": "Infeksi serius oleh Botryosphaeria obtusa yang memicu pembusukan melingkar hingga buah mengering dan mengkerut menghitam.",
        "prevention": "Buang dan musnahkan buah yang mengering (mummy apple), hindari luka mekanis pada buah, dan aplikasikan fungisida preventif.",
    },
    "Codling Moth": {
        "title": "Hama Ulat Apel (Codling Moth)",
        "desc": "Kerusakan fisik akibat larva Cydia pomonella yang menggerogoti hingga membuat lorong menuju bagian dalam dan biji apel.",
        "prevention": "Gunakan perangkap feromon, pasang pembungkus buah (fruit bagging), atau gunakan insektisida hayati terdaftar.",
    },
    "Healthy": {
        "title": "Buah Apel Sehat (Healthy)",
        "desc": "Kondisi fisik buah apel optimal, permukaan mulus, dan bebas dari tanda-tanda infeksi patogen maupun serangan hama.",
        "prevention": "Pertahankan pemeliharaan rutin, pemupukan berimbang, dan manajemen pengairan tanah yang optimal.",
    },
    "Powdery Mildew": {
        "title": "Embun Tepung (Powdery Mildew)",
        "desc": "Disebabkan oleh jamur Podosphaera leucotricha yang melapisi permukaan kulit buah atau daun dengan lapisan seperti serbuk putih.",
        "prevention": "Gunakan varietas tanaman yang memiliki ketahanan tinggi dan lakukan pemangkasan tunas terinfeksi serta aplikasi fungisida sulfur.",
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
st.sidebar.markdown("### 🍎 **Apple Diagnostic System**")
st.sidebar.caption("Deep Learning Image Analysis v1.0")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi Utama",
    ["🏠 Home", "🤖 Teachable Machine Detector"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size: 12px; color: #64748B;'>
        <b>Arsitektur Model:</b> MobileNetV2<br>
        <b>Confidence Threshold:</b> 60.0%<br>
        <b>Framework:</b> TensorFlow & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# HALAMAN 1: HOME
# ------------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("Sistem Klasifikasi Penyakit Buah Apel")
    st.markdown(
        "<p style='color: #475569; font-size: 16px; margin-top: -10px;'>"
        "Platform identifikasi kondisi kesehatan buah berbasis kecerdasan buatan dan Convolutional Neural Network."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 1. Penjelasan Aplikasi
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<div class="pro-card-header"><h3>📖 Penjelasan Aplikasi</h3></div>', unsafe_allow_html=True)
    st.write(
        """
        Aplikasi ini dirancang untuk mendiagnosa kondisi kesehatan buah apel (*Malus domestica*) secara otomatis menggunakan teknologi **Deep Learning**. 
        Dengan menerapkan teknik **Transfer Learning** pada arsitektur **MobileNetV2**, sistem mampu menganalisis citra permukaan buah apel dan mengklasifikasikannya ke dalam **6 kategori** (1 kelas sehat dan 5 kelas infeksi penyakit/hama) secara presisi dan efisien.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Tujuan Aplikasi
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<div class="pro-card-header"><h3>🎯 Tujuan Pengembangan</h3></div>', unsafe_allow_html=True)
    st.write(
        """
        - **Akurasi Diagnosa Dini:** Membantu petani dan pengelola kebun mengidentifikasi jenis penyakit pada buah apel secara akurat guna mencegah persebaran infeksi yang lebih luas.
        - **Optimalisasi Penanganan:** Memberikan rujukan solusi dan penanganan yang tepat sasaran sesuai dengan jenis penyakit yang terdeteksi.
        - **Validasi Input (*False-Positive Prevention*):** Menolak prediksi pada gambar non-apel atau gambar berkualitas buruk menggunakan mekanisme *Confidence Thresholding* 60%.
        - **Aksesibilitas Tinggi:** Menyediakan sistem pakar berbasis web yang dapat diakses secara publik, cepat, dan responsif dari berbagai perangkat.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Cara Penggunaan
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<div class="pro-card-header"><h3>💡 Panduan Penggunaan</h3></div>', unsafe_allow_html=True)

    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">1</div>
                <h4 style="margin: 0; color: #0B2545;">Buka Detektor</h4>
                <p style="font-size: 13px; color: #475569; margin-top: 8px;">Pilih menu <b>Teachable Machine Detector</b> pada bilah navigasi di sebelah kiri.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_step2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">2</div>
                <h4 style="margin: 0; color: #0B2545;">Unggah Gambar</h4>
                <p style="font-size: 13px; color: #475569; margin-top: 8px;">Unggah foto buah apel dengan format JPG atau PNG pada panel pengujian.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_step3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-number">3</div>
                <h4 style="margin: 0; color: #0B2545;">Hasil & Solusi</h4>
                <p style="font-size: 13px; color: #475569; margin-top: 8px;">Sistem menampilkan hasil diagnosa, grafik probabilitas kelas, serta petunjuk penanganan.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HALAMAN 2: DETEKS PENYAKIT (TEACHABLE MACHINE LAYOUT)
# ------------------------------------------------------------------------------
elif page == "🤖 Teachable Machine Detector":
    st.title("Klasifikasi Penyakit Buah Apel")
    st.markdown(
        "<p style='color: #475569; font-size: 15px; margin-top: -10px;'>"
        "Modul diagnosa citra real-time dengan antarmuka dua panel."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Layout Side-by-Side (2 Kolom ala Teachable Machine)
    col1, col2 = st.columns([1, 1], gap="large")

    # -------------------------- KOLOM KIRI: INPUT --------------------------
    with col1:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown('<div class="pro-card-header"><h3>📷 Input Gambar</h3></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Unggah berkas gambar buah apel (JPG, JPEG, PNG)...",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Preview Gambar Input", use_container_width=True)
        else:
            st.info("Silakan unggah gambar buah apel untuk memulai proses klasifikasi.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- KOLOM KANAN: OUTPUT --------------------------
    with col2:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown('<div class="pro-card-header"><h3>📊 Hasil Diagnosa & Probabilitas</h3></div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            if model is None:
                st.error("File model `apple_disease_model.h5` tidak ditemukan pada direktori utama.")
            else:
                with st.spinner("Memproses & Menganalisis Gambar..."):
                    processed_img = preprocess_image(image)
                    predictions = model.predict(processed_img)[0]
                    top_idx = int(np.argmax(predictions))

                    predicted_class = CLASS_NAMES[top_idx]
                    confidence = float(predictions[top_idx]) * 100

                CONFIDENCE_THRESHOLD = 60.0

                if confidence < CONFIDENCE_THRESHOLD:
                    # KONDISI TIDAK TERDETEKSI (Confidence < 60%)
                    st.error("❌ **Hasil Diagnosa: Tidak Terdeteksi**")
                    st.warning(
                        f"Tingkat keyakinan tertinggi hanya **{confidence:.2f}%** (di bawah batas minimal 60.0%). "
                        "Objek dalam gambar tidak dikenali sebagai buah apel yang valid atau citra terlalu buram."
                    )

                    st.markdown("---")
                    st.markdown("<h4 style='font-size: 15px; color: #0B2545;'>Probabilitas Kelas:</h4>", unsafe_allow_html=True)
                    for c_name in CLASS_NAMES:
                        st.write(f"**{c_name}**: 0.0%")
                        st.progress(0.0)

                else:
                    # KONDISI TERDETEKSI VALID (Confidence >= 60%)
                    st.success(f"🏆 **Hasil Diagnosa: {predicted_class}**")
                    st.info(f"🎯 **Tingkat Keyakinan (Confidence):** {confidence:.2f}%")

                    st.markdown("---")
                    st.markdown("<h4 style='font-size: 15px; color: #0B2545;'>Probabilitas Kelas:</h4>", unsafe_allow_html=True)

                    # Display Progress Bars per Class
                    for i, c_name in enumerate(CLASS_NAMES):
                        prob = float(predictions[i])
                        prob_percent = prob * 100
                        st.write(f"**{c_name}**: {prob_percent:.1f}%")
                        st.progress(prob)

                    # Detail Deskripsi & Solusi Penanganan
                    st.markdown("---")
                    detail = CLASS_DETAILS.get(predicted_class, {})
                    if detail:
                        with st.expander("📌 Detail Penyakit & Tindakan Penanganan", expanded=True):
                            st.markdown(f"**{detail.get('title', '')}**")
                            st.write(detail.get("desc", ""))
                            st.markdown("**Solusi Penanganan:**")
                            st.write(detail.get("prevention", ""))
        else:
            st.write("Hasil prediksi dan distribusi probabilitas akan ditampilkan di sini secara otomatis setelah gambar diunggah.")

        st.markdown("</div>", unsafe_allow_html=True)
