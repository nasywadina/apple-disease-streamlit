from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------------------------
# CONFIGURASI HALAMAN
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Pendeteksi Penyakit Buah Apel",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# CUSTOM CSS (WARNA & TAMPILAN MODERN)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Background Utama */
    .main {
        background-color: #f4f6f8;
    }
    
    /* Header & Title Styling */
    h1, h2, h3 {
        color: #1b5e20;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Styling Kartu / Card Box */
    .custom-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border-left: 5px solid #2e7d32;
    }
    
    .info-card {
        background-color: #e8f5e9;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #c8e6c9;
        margin-bottom: 15px;
    }

    /* Progress bar custom color */
    .stProgress > div > div > div > div {
        background-color: #4caf50;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# LOAD MODEL & KONSTANTA
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
        "desc": "Disebabkan oleh infeksi jamur Colletotrichum. Gejala ditandai dengan munculnya bercak melingkar berwarna cokelat kehitaman yang cekung pada permukaan kulit buah apel.",
        "prevention": "Pangkas bagian tanaman terinfeksi, tingkatkan sirkulasi udara di sekitar tajuk pohon, dan lakukan penyemprotan fungisida berbasis tembaga secara berkala.",
    },
    "Black Pox": {
        "title": "Cacar Hitam (Black Pox)",
        "desc": "Infeksi jamur Helminthosporium papulosum yang memicu timbulnya bintik-bintik hitam kecil sedikit menonjol dan kasar pada permukaan buah.",
        "prevention": "Sanitasi kebun dengan membersihkan buah dan daun yang gugur di tanah, serta aplikasikan fungisida pelindung sejak fase awal pembentukan buah.",
    },
    "Black Rot": {
        "title": "Busuk Hitam (Black Rot)",
        "desc": "Disebabkan oleh Botryosphaeria obtusa. Menyebabkan buah membusuk dengan pola melingkar berwarna cokelat hingga akhirnya mengering, menghitam, dan mengerut menyerupai mumi.",
        "prevention": "Kumpulkan dan bakar buah yang terinfeksi mummy apple, cegah luka fisik pada buah, dan aplikasikan fungisida preventif.",
    },
    "Codling Moth": {
        "title": "Hama Ulat Apel (Codling Moth)",
        "desc": "Kerusakan akibat larva ngat Cydia pomonella yang menggerogoti dan membuat lorong ke dalam daging hingga inti/biji buah apel.",
        "prevention": "Gunakan perangkap feromon untuk memantau populasi ngat, bungkus buah apel muda (*fruit bagging*), atau aplikasikan biologis/insektisida terdaftar.",
    },
    "Healthy": {
        "title": "Buah Apel Sehat (Healthy)",
        "desc": "Buah apel berada dalam kondisi segar, tekstur permukaan mulus, dan bebas dari bercak infeksi jamur maupun kerusakan akibat hama.",
        "prevention": "Pertahankan pola pemeliharaan rutin, pemupukan berimbang, drainase tanah yang baik, serta pemantauan kebun secara periodik.",
    },
    "Powdery Mildew": {
        "title": "Embun Tepung (Powdery Mildew)",
        "desc": "Infeksi jamur Podosphaera leucotricha yang ditandai dengan bercak atau lapisan halus berwarna putih keputihan menyerupai bedak/tepung pada kulit buah atau daun.",
        "prevention": "Tanam varietas yang lebih tahan, pangkas tunas yang terinfeksi pada awal musim semi, dan gunakan fungisida berbahan aktif sulfur.",
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
st.sidebar.image("https://img.icons8.com/color/96/apple.png", width=70)
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home", "🤖 Deteksi Penyakit (Teachable Machine)"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Sistem Klasifikasi Penyakit Apel**\n"
    "Berbasis Transfer Learning MobileNetV2 & Streamlit Framework."
)

# ------------------------------------------------------------------------------
# HALAMAN 1: HOME
# ------------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🍎 Sistem Deteksi Penyakit Buah Apel Berbasis AI")
    st.write(
        "Selamat datang di platform klasifikasi citra penyakit buah apel otomatis berbasis pembelajaran dalam (*deep learning*)."
    )
    st.markdown("---")

    # 1. Penjelasan Aplikasi
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📖 1. Penjelasan Aplikasi")
    st.write(
        """
        Aplikasi ini merupakan sistem pendeteksi dan pengklasifikasi penyakit pada buah apel (*Malus domestica*) secara otomatis. 
        Sistem memanfaatkan teknologi kecerdasan buatan dengan metode **Deep Learning** menggunakan arsitektur **MobileNetV2** yang telah dioptimasi (*Transfer Learning*). 
        
        Melalui aplikasi web ini, pengguna cukup mengunggah foto buah apel, dan sistem akan menganalisis kondisi fisik buah secara *real-time* untuk mengidentifikasi apakah buah berada dalam kondisi **Sehat** atau terinfeksi salah satu dari **5 Jenis Penyakit/Hama Utama** (Antraknosa, Cacar Hitam, Busuk Hitam, Ulat Apel, atau Embun Tepung).
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Tujuan Aplikasi
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🎯 2. Tujuan Aplikasi")
    st.write(
        """
        Adapun tujuan utama dari pengembangan aplikasi ini adalah:
        - **Membantu Petani & Masyarakat:** Memudahkan proses identifikasi penyakit buah apel secara cepat dan akurat tanpa harus menunggu hadirnya penyuluh atau tenaga ahli di lapangan.
        - **Mencegah Kesalahan Diagnosis:** Meminimalisir kesalahan identifikasi gejala penyakit yang sering kali mirip secara kasat mata, sehingga pemberian pestisida/penanganan menjadi lebih tepat sasaran.
        - **Mencegah Prediksi Palsu (*False Positive*):** Dilengkapi dengan mekanisme *Confidence Thresholding* (60%) untuk menolak dan menyaring input gambar non-apel atau gambar berkualitas buruk.
        - **Menyediakan Layanan Terbuka:** Menyajikan media diagnosa digital berbasis web yang interaktif, informatif, serta dapat diakses secara publik dan gratis 24/7.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Cara Penggunaan
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("💡 3. Panduan & Cara Penggunaan")
    st.write(
        """
        Langkah-langkah untuk melakukan diagnosa penyakit buah apel menggunakan aplikasi ini sangat mudah:
        """
    )

    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.markdown(
            """
            <div class="info-card">
                <h4>Langkah 1: Masuk Halaman</h4>
                <p>Buka menu <b>🤖 Deteksi Penyakit (Teachable Machine)</b> pada bilah navigasi di sebelah kiri (sidebar).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_step2:
        st.markdown(
            """
            <div class="info-card">
                <h4>Langkah 2: Unggah Gambar</h4>
                <p>Klik tombol <b>Browse files</b> pada panel sebelah kiri untuk memilih gambar buah apel yang ingin diperiksa (Format JPG/PNG).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_step3:
        st.markdown(
            """
            <div class="info-card">
                <h4>Langkah 3: Lihat Hasil</h4>
                <p>Sistem akan menampilkan status diagnosa, tingkat keyakinan (confidence), bilah persentase probabilitas, serta solusi penanganannya pada panel kanan.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HALAMAN 2: DETEKSI PENYAKIT (TEACHABLE MACHINE LAYOUT)
# ------------------------------------------------------------------------------
elif page == "🤖 Deteksi Penyakit (Teachable Machine)":
    st.title("🤖 Klasifikasi Penyakit Buah Apel (Real-time)")
    st.write(
        "Unggah gambar buah apel pada panel kiri untuk melihat estimasi klasifikasi dan persentase probabilitas pada panel kanan."
    )
    st.markdown("---")

    # Layout Side-by-Side (2 Kolom ala Teachable Machine)
    col1, col2 = st.columns([1, 1], gap="large")

    # -------------------------- KOLOM KIRI: INPUT --------------------------
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📷 Input Gambar")
        uploaded_file = st.file_uploader(
            "Pilih file gambar buah apel (JPG, JPEG, PNG)...",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Input", use_container_width=True)
        else:
            st.info("📌 Silakan unggah gambar buah apel untuk memulai analisa.")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- KOLOM KANAN: OUTPUT --------------------------
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📊 Output Prediksi (Output Class)")

        if uploaded_file is not None:
            if model is None:
                st.error(
                    "❌ File model `apple_disease_model.h5` tidak ditemukan pada direktori utama."
                )
            else:
                with st.spinner("🔍 Menganalisis gambar..."):
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
                        "Gambar teridentifikasi bukan sebagai buah apel yang valid atau kualitas gambar terlalu buram."
                    )

                    st.markdown("---")
                    st.write("**Probabilitas Seluruh Kelas:**")
                    for c_name in CLASS_NAMES:
                        st.write(f"**{c_name}**: 0.0%")
                        st.progress(0.0)

                else:
                    # KONDISI TERDETEKSI VALID (Confidence >= 60%)
                    st.success(f"🏆 **Hasil Diagnosa: {predicted_class}**")
                    st.info(f"🎯 **Tingkat Keyakinan (Confidence):** {confidence:.2f}%")

                    st.markdown("---")
                    st.write("**Probabilitas Seluruh Kelas:**")

                    # Display Progress Bars per Class ala Teachable Machine
                    for i, c_name in enumerate(CLASS_NAMES):
                        prob = float(predictions[i])
                        prob_percent = prob * 100
                        st.write(f"**{c_name}**: {prob_percent:.1f}%")
                        st.progress(prob)

                    # Detail Deskripsi & Solusi Penanganan
                    st.markdown("---")
                    detail = CLASS_DETAILS.get(predicted_class, {})
                    if detail:
                        with st.expander(
                            "📌 Deskripsi & Solusi Penanganan", expanded=True
                        ):
                            st.markdown(f"**{detail.get('title', '')}**")
                            st.write(detail.get("desc", ""))
                            st.markdown("**Langkah Penanganan:**")
                            st.write(detail.get("prevention", ""))
        else:
            st.write(
                "Tampilan probabilitas dan diagnosa penyakit akan langsung muncul di sini setelah gambar diunggah."
            )

        st.markdown("</div>", unsafe_allow_html=True)
