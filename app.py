from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AppleScan AI — Klasifikasi Kondisi Buah Apel",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------------------
# 2. CUSTOM CSS — DESAIN "ORCHARD" (TEMA APEL: MERAH / HIJAU DAUN / KRIM)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700;800&display=swap');

    :root{
        --apple-red:#B3261E;
        --apple-red-dark:#8C1D17;
        --apple-red-soft:#F6D9D6;
        --leaf-green:#2F7A4B;
        --leaf-green-soft:#DCEEE1;
        --gold:#D79A2C;
        --gold-soft:#F7E7C9;
        --cream:#FFFBF5;
        --cream-alt:#FBF2E6;
        --charcoal:#26211C;
        --stone:#6E655C;
        --line:#EDE1D2;
    }

    html{scroll-behavior:smooth;}
    html, body, [class*="css"]{font-family:'Inter', sans-serif; color:var(--charcoal);}
    h1,h2,h3,.display{font-family:'Fraunces', serif;}

    /* Jaring pengaman: pastikan semua teks bawaan Streamlit tetap hitam pekat,
       tidak ikut memutih saat tema browser/Streamlit pengguna dalam mode gelap */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stVerticalBlockBorderWrapper"] label,
    .stTabs [data-baseweb="tab"] *{
        color:#000000 !important;
    }

    .stApp{background-color:var(--cream) !important;}
    #MainMenu, footer, header{visibility:hidden;}
    .block-container{padding-top:0 !important; padding-bottom:0 !important; padding-left:48px !important; padding-right:48px !important; max-width:100% !important;}

    /* ---------- NAVBAR ---------- */
    .navbar{
        position:fixed; top:0; left:0; right:0; z-index:99999;
        display:flex; align-items:center; justify-content:space-between;
        padding:16px 48px; background:rgba(255,251,245,0.96);
        backdrop-filter:blur(6px); border-bottom:1px solid var(--line);
    }
    .navbar-spacer{height:74px;}
    .nav-brand{display:flex; align-items:center; gap:10px; font-family:'Fraunces',serif;
        font-weight:600; font-size:1.25rem; color:var(--charcoal);}
    .nav-brand .mark{width:34px; height:34px; border-radius:9px; background:var(--apple-red);
        display:flex; align-items:center; justify-content:center; font-size:1.05rem;}
    .nav-links{display:flex; gap:34px; font-size:0.92rem; font-weight:500;}
    .nav-links a{color:var(--stone); text-decoration:none;}
    .nav-links a:hover{color:var(--apple-red);}
    .nav-cta{background:var(--apple-red); color:#fff !important; padding:9px 20px;
        border-radius:999px; font-weight:600; text-decoration:none; font-size:0.88rem;}
    .nav-cta:hover{background:var(--apple-red-dark); color:#fff !important;}
    @media (max-width:900px){.nav-links{display:none;}}

    /* ---------- SEED DIVIDER (signature motif) ---------- */
    .seed-divider{display:flex; justify-content:center; gap:10px; padding:6px 0 34px 0;}
    .seed-divider span{width:7px; height:11px; border-radius:60% 60% 60% 60% / 70% 70% 40% 40%;
        background:var(--gold); opacity:0.55; transform:rotate(20deg);}
    .seed-divider span:nth-child(2n){background:var(--leaf-green); opacity:0.45;}
    .seed-divider span:nth-child(3n){background:var(--apple-red); opacity:0.4;}

    /* ---------- SECTION WRAPPERS ---------- */
    .section{padding:78px 0;}
    .section-alt{background:var(--cream-alt); margin:0 -48px; padding-left:48px; padding-right:48px;}
    .eyebrow{display:inline-flex; align-items:center; gap:8px; color:var(--apple-red);
        font-weight:700; font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase;
        background:var(--apple-red-soft); padding:6px 14px; border-radius:999px; margin-bottom:18px;}
    .section-title{font-size:2.1rem; font-weight:600; color:var(--charcoal); margin:0 0 12px 0;}
    .section-sub{color:var(--stone); font-size:1.02rem; max-width:640px; line-height:1.6;}
    .center{text-align:center; margin-left:auto; margin-right:auto;}

    /* ---------- HERO ---------- */
    .hero-wrap{display:flex; align-items:center; gap:56px; padding:64px 0 20px 0; flex-wrap:wrap;}
    .hero-text{flex:1 1 420px; min-width:320px;}
    .hero-title{font-size:3.1rem; line-height:1.08; font-weight:600; color:var(--charcoal); margin:14px 0 18px 0;}
    .hero-title .accent{color:var(--apple-red);}
    .hero-desc{color:var(--stone); font-size:1.08rem; line-height:1.7; max-width:520px; margin-bottom:30px;}
    .hero-btns{display:flex; gap:14px; margin-bottom:38px; flex-wrap:wrap;}
    .btn-primary{background:var(--apple-red); color:#fff !important; padding:13px 26px; border-radius:10px;
        font-weight:700; text-decoration:none; font-size:0.95rem; box-shadow:0 8px 18px -6px rgba(179,38,30,0.45);}
    .btn-primary:hover{background:var(--apple-red-dark); color:#fff !important;}
    .btn-secondary{border:1.5px solid var(--line); color:var(--charcoal) !important; padding:12px 24px;
        border-radius:10px; font-weight:600; text-decoration:none; font-size:0.95rem; background:#fff;}
    .btn-secondary:hover{border-color:var(--apple-red); color:var(--apple-red) !important;}
    .hero-stats{display:flex; gap:38px; flex-wrap:wrap;}
    .stat-num{font-family:'Fraunces',serif; font-size:1.9rem; font-weight:600; color:var(--apple-red);}
    .stat-label{color:var(--stone); font-size:0.82rem;}

    /* ---------- HERO APPLE GRAPHIC (signature illustration) ---------- */
    .hero-visual{flex:1 1 320px; min-width:280px; display:flex; justify-content:center;}
    .apple-frame{width:340px; height:340px; border-radius:28px; background:linear-gradient(160deg,#FDEDE9 0%,#FBF2E6 55%,#E7F2EA 100%);
        display:flex; align-items:center; justify-content:center; border:1px solid var(--line);
        box-shadow:0 30px 50px -24px rgba(38,33,28,0.25);}

    /* ---------- CARDS ---------- */
    .card{background:#fff; border:1px solid var(--line); border-radius:16px; padding:26px;
        box-shadow:0 4px 14px -8px rgba(38,33,28,0.08);}

    /* Kotak native st.container(border=True) — dipakai untuk area yang berisi widget
       interaktif (upload/kamera/hasil), disamakan gayanya dengan .card */
    div[data-testid="stVerticalBlockBorderWrapper"]{
        border-radius:16px !important; border-color:var(--line) !important;
        box-shadow:0 4px 14px -8px rgba(38,33,28,0.08);
    }

    /* ---------- FILE UPLOADER (disamakan dengan tema Orchard) ---------- */
    div[data-testid="stFileUploaderDropzone"]{
        background:var(--cream-alt) !important;
        border:1.5px dashed var(--line) !important;
        border-radius:14px !important;
        transition:border-color .15s ease, background .15s ease;
    }
    div[data-testid="stFileUploaderDropzone"]:hover{
        border-color:var(--apple-red) !important;
        background:var(--apple-red-soft) !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small,
    div[data-testid="stFileUploaderDropzoneInstructions"] div{
        color:var(--charcoal) !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] svg{
        fill:var(--apple-red) !important;
    }
    div[data-testid="stFileUploaderDropzone"] button{
        background:#fff !important;
        color:var(--charcoal) !important;
        border:1.5px solid var(--line) !important;
        border-radius:8px !important;
        font-weight:600 !important;
        box-shadow:none !important;
    }
    div[data-testid="stFileUploaderDropzone"] button:hover{
        border-color:var(--apple-red) !important;
        color:var(--apple-red) !important;
    }
    div[data-testid="stFileUploaderDropzone"] button p{
        color:inherit !important;
    }
    /* kartu file yang sudah terunggah (nama file, ukuran, tombol hapus) */
    div[data-testid="stFileUploaderFile"]{
        background:#fff !important;
        border:1px solid var(--line) !important;
        border-radius:10px !important;
        color:var(--charcoal) !important;
    }
    div[data-testid="stFileUploaderFile"] *{
        color:var(--charcoal) !important;
    }
    div[data-testid="stFileUploaderFile"] small{
        color:var(--stone) !important;
    }

    .grid{display:grid; gap:22px;}
    .grid-2{grid-template-columns:1.1fr 1fr;}
    .grid-3{grid-template-columns:repeat(3,1fr);}
    .grid-4{grid-template-columns:repeat(4,1fr);}
    @media (max-width:900px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr;}}

    .mission-check{display:flex; align-items:flex-start; gap:10px; margin-top:12px; color:var(--charcoal); font-size:0.95rem;}
    .mission-check .dot{color:var(--leaf-green); font-weight:700;}

    .class-card{border-left:4px solid var(--stone);}
    .class-card.healthy{border-left-color:var(--leaf-green);}
    .class-card.pest{border-left-color:var(--gold);}
    .class-card.disease{border-left-color:var(--apple-red);}
    .class-dot{width:12px; height:12px; border-radius:50%; display:inline-block; margin-bottom:10px;}
    .class-card.healthy .class-dot{background:var(--leaf-green);}
    .class-card.pest .class-dot{background:var(--gold);}
    .class-card.disease .class-dot{background:var(--apple-red);}
    .class-name{font-weight:700; font-size:1rem; margin-bottom:4px; color:#000000 !important;}
    .class-desc{color:var(--stone); font-size:0.86rem; line-height:1.5;}

    .tech-icon{width:52px; height:52px; border-radius:14px; display:flex; align-items:center;
        justify-content:center; font-size:1.4rem; margin-bottom:16px; color:#fff;}
    .tech-title{font-weight:700; font-size:1.05rem; margin-bottom:6px; color:#000000 !important;}
    .tech-desc{color:var(--stone); font-size:0.88rem; line-height:1.5; margin-bottom:12px;}
    .tag{display:inline-block; font-size:0.72rem; font-weight:700; padding:4px 11px;
        border-radius:999px; letter-spacing:0.03em;}

    .arch-step{text-align:center;}
    .arch-icon{width:56px; height:56px; border-radius:14px; margin:0 auto 12px auto; display:flex;
        align-items:center; justify-content:center; font-size:1.5rem; color:#fff;}
    .arch-title{font-weight:700; font-size:0.95rem; color:#000000 !important;}
    .arch-sub{color:var(--stone); font-size:0.8rem; margin-top:2px;}
    .arch-arrow{text-align:center; font-size:1.3rem; color:var(--line); align-self:center;}

    .feature-card{text-align:left;}
    .feature-icon{width:48px; height:48px; border-radius:12px; display:flex; align-items:center;
        justify-content:center; font-size:1.25rem; margin-bottom:14px; color:#fff;}

    .flow-step{text-align:center; position:relative;}
    .flow-num{width:52px; height:52px; border-radius:50%; background:var(--apple-red); color:#fff;
        font-family:'Fraunces',serif; font-weight:600; font-size:1.3rem; display:flex; align-items:center;
        justify-content:center; margin:0 auto 14px auto;}
    .flow-title{font-weight:700; font-size:0.95rem; margin-bottom:4px; color:#000000 !important;}
    .flow-sub{color:var(--stone); font-size:0.82rem; line-height:1.4;}

    /* ---------- DIAGNOSIS BADGES (fungsi klasifikasi) ---------- */
    .badge-success{background:var(--leaf-green-soft); border:2px solid var(--leaf-green); border-radius:14px; padding:18px 20px; color:#1E4D30;}
    .badge-danger{background:var(--apple-red-soft); border:2px solid var(--apple-red); border-radius:14px; padding:18px 20px; color:#6E1712;}
    .badge-warning{background:var(--gold-soft); border:2px solid var(--gold); border-radius:14px; padding:18px 20px; color:#7A5613;}
    .badge-success h3, .badge-success p{color:#1E4D30 !important;}
    .badge-danger h3, .badge-danger p{color:#6E1712 !important;}
    .badge-warning h3, .badge-warning p{color:#7A5613 !important;}
    .stProgress > div > div > div > div{background-color:var(--apple-red) !important; border-radius:8px;}

    /* ---------- FOOTER ---------- */
    .footer{padding:36px 0; text-align:center; color:var(--stone); font-size:0.85rem; border-top:1px solid var(--line);}
    </style>
""",
    unsafe_allow_html=True,
)


def seed_divider():
    st.markdown(
        '<div class="seed-divider">' + "".join(["<span></span>"] * 9) + "</div>",
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

# kategori tampilan: 'disease' (jamur), 'pest' (hama), 'healthy'
CLASS_CATEGORY = {
    "Anthracnose": "disease",
    "Black Pox": "disease",
    "Black Rot": "disease",
    "Codling Moth": "pest",
    "Healthy": "healthy",
    "Powdery Mildew": "disease",
}

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
# 4. NAVBAR
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="navbar">
        <div class="nav-brand"><span class="mark">🍎</span> AppleScan AI</div>
        <div class="nav-links">
            <a href="#home">Beranda</a>
            <a href="#tentang">Tentang</a>
            <a href="#teknologi">Teknologi</a>
            <a href="#fitur">Fitur</a>
        </div>
        <a class="nav-cta" href="#coba">Coba Sekarang</a>
    </div>
    <div class="navbar-spacer"></div>
    <div id="home"></div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 5. HERO
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-text">
            <div class="eyebrow">🌿 Klasifikasi Berbasis AI</div>
            <div class="hero-title">Deteksi Dini<br><span class="accent">Kondisi Buah Apel</span><br>dengan AI</div>
            <div class="hero-desc">
                Sistem klasifikasi cerdas berbasis Convolutional Neural Network dengan
                Transfer Learning MobileNetV2 untuk mengidentifikasi kondisi buah apel
                secara akurat, langsung dari peramban web Anda.
            </div>
            <div class="hero-btns">
                <a class="btn-primary" href="#coba">🔍 Mulai Diagnosa</a>
                <a class="btn-secondary" href="#tentang">Pelajari Lebih</a>
            </div>
            <div class="hero-stats">
                <div><div class="stat-num">86.67%</div><div class="stat-label">Akurasi Model</div></div>
                <div><div class="stat-num">6</div><div class="stat-label">Kelas Deteksi</div></div>
                <div><div class="stat-num">MobileNetV2</div><div class="stat-label">Arsitektur</div></div>
            </div>
        </div>
        <div class="hero-visual">
            <div class="apple-frame">
                <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                    <path d="M100 62 C60 40 20 70 22 118 C24 158 58 182 88 182 C96 182 100 178 100 178 C100 178 104 182 112 182 C142 182 176 158 178 118 C180 70 140 40 100 62 Z" fill="#B3261E" opacity="0.92"/>
                    <path d="M100 62 C100 62 96 42 74 32" stroke="#8C1D17" stroke-width="6" fill="none" stroke-linecap="round"/>
                    <path d="M100 44 C100 44 122 22 148 32 C148 32 138 58 112 58 Z" fill="#2F7A4B"/>
                    <ellipse cx="70" cy="98" rx="18" ry="26" fill="#ffffff" opacity="0.16"/>
                </svg>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
seed_divider()

# ------------------------------------------------------------------------------
# 6. TENTANG SISTEM
# ------------------------------------------------------------------------------
st.markdown('<div id="tentang" style="padding-top:78px;"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="center" style="max-width:680px;">
        <div class="eyebrow" style="margin-left:auto;margin-right:auto;">Tentang Sistem</div>
        <div class="section-title">Tentang AppleScan AI</div>
        <div class="section-sub center">
            Sistem klasifikasi otomatis yang memanfaatkan teknologi Deep Learning untuk
            mengidentifikasi kondisi buah apel dengan presisi tinggi.
        </div>
    </div>
    <br>
    <div class="card">
        <div style="font-size:1.6rem;margin-bottom:10px;"></div>
        <div class="tech-title">Misi Kami</div>
        <div class="tech-desc" style="margin-bottom:6px;">
            Membantu konsumen mengenali kondisi buah apel secara objektif
            dan efisien sebelum membeli atau mengonsumsinya.
        </div>
        <div class="mission-check"><span class="dot">✓</span> Akurasi model teruji 86.67%</div>
        <div class="mission-check"><span class="dot">✓</span> Hasil analisis cepat</div>
        <div class="mission-check"><span class="dot">✓</span> Berbasis web, tanpa instalasi tambahan</div>
    </div>
    <br>
    <div class="tech-title" style="margin-bottom:2px;">6 Kategori Klasifikasi</div>
    """,
    unsafe_allow_html=True,
)

class_cols = st.columns(3)
class_layout = [
    ("Healthy", "healthy"),
    ("Anthracnose", "disease"),
    ("Black Pox", "disease"),
    ("Black Rot", "disease"),
    ("Codling Moth", "pest"),
    ("Powdery Mildew", "disease"),
]
for i, (cname, cat) in enumerate(class_layout):
    with class_cols[i % 3]:
        short = {
            "Healthy": "Buah sehat, bebas infeksi.",
            "Anthracnose": "Lesi cekung cokelat kehitaman.",
            "Black Pox": "Bintik hitam kasar menonjol.",
            "Black Rot": "Busuk melingkar hingga mengerut.",
            "Codling Moth": "Lorong bekas gerekan larva.",
            "Powdery Mildew": "Lapisan serbuk putih pada kulit.",
        }[cname]
        st.markdown(
            f"""
            <div class="card class-card {cat}" style="margin-bottom:16px;">
                <div class="class-dot"></div>
                <div class="class-name">{cname}</div>
                <div class="class-desc">{short}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div style="padding-bottom:78px;"></div>', unsafe_allow_html=True)
seed_divider()

# ------------------------------------------------------------------------------
# 7. TECHNOLOGY STACK
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="section section-alt" id="teknologi">
        <div class="center" style="max-width:680px;">
            <div class="eyebrow" style="margin-left:auto;margin-right:auto;">Technology Stack</div>
            <div class="section-title">Teknologi Canggih</div>
            <div class="section-sub center">Dibangun dengan teknologi terkini dalam bidang Machine Learning dan Web Development.</div>
        </div>
        <br>
        <div class="grid grid-3">
            <div class="card">
                <div class="tech-icon" style="background:#B3261E;">🧠</div>
                <div class="tech-title">TensorFlow</div>
                <div class="tech-desc">Framework Deep Learning untuk membangun dan melatih model Convolutional Neural Network (CNN).</div>
                <span class="tag" style="background:var(--apple-red-soft);color:var(--apple-red-dark);">Deep Learning</span>
            </div>
            <div class="card">
                <div class="tech-icon" style="background:#26211C;">🎈</div>
                <div class="tech-title">Streamlit</div>
                <div class="tech-desc">Framework web Python yang ringan dan interaktif untuk membangun antarmuka aplikasi klasifikasi.</div>
                <span class="tag" style="background:#EAE6E0;color:var(--charcoal);">Frontend & Backend</span>
            </div>
            <div class="card">
                <div class="tech-icon" style="background:#2F7A4B;">🧩</div>
                <div class="tech-title">MobileNetV2</div>
                <div class="tech-desc">Arsitektur CNN ringan berbasis Transfer Learning dengan multiple layers untuk ekstraksi fitur visual.</div>
                <span class="tag" style="background:var(--leaf-green-soft);color:var(--leaf-green);">AI Model</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 8. ARSITEKTUR MODEL
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="section">
        <div class="card">
            <div class="section-title center" style="font-size:1.5rem;">Arsitektur Model</div>
            <br>
            <div style="display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr; align-items:center; gap:6px;">
                <div class="arch-step">
                    <div class="arch-icon" style="background:#B3261E;">🖼️</div>
                    <div class="arch-title">Input Layer</div>
                    <div class="arch-sub">224×224 RGB</div>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <div class="arch-icon" style="background:#8C1D17;">🧩</div>
                    <div class="arch-title">MobileNetV2 Base</div>
                    <div class="arch-sub">Feature Extraction</div>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <div class="arch-icon" style="background:#2F7A4B;">📉</div>
                    <div class="arch-title">Global Pooling</div>
                    <div class="arch-sub">Dimension Reduction</div>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <div class="arch-icon" style="background:#D79A2C;">🎯</div>
                    <div class="arch-title">Dense Layers</div>
                    <div class="arch-sub">6 Classes Output</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
seed_divider()

# ------------------------------------------------------------------------------
# 9. FITUR UNGGULAN
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="section section-alt" id="fitur">
        <div class="center" style="max-width:680px;">
            <div class="eyebrow" style="margin-left:auto;margin-right:auto;">Features</div>
            <div class="section-title">Fitur Unggulan</div>
            <div class="section-sub center">Berbagai fitur yang membuat sistem ini praktis digunakan untuk klasifikasi kondisi buah apel.</div>
        </div>
        <br>
        <div class="grid grid-4">
            <div class="card feature-card">
                <div class="feature-icon" style="background:#B3261E;">⚡</div>
                <div class="tech-title" style="font-size:0.98rem;">Proses Cepat</div>
                <div class="tech-desc">Hasil klasifikasi tampil segera setelah gambar diunggah atau difoto.</div>
            </div>
            <div class="card feature-card">
                <div class="feature-icon" style="background:#2F7A4B;">📈</div>
                <div class="tech-title" style="font-size:0.98rem;">Akurasi Tinggi</div>
                <div class="tech-desc">Model teruji dengan akurasi 86.67% pada data pengujian.</div>
            </div>
            <div class="card feature-card">
                <div class="feature-icon" style="background:#D79A2C;">📱</div>
                <div class="tech-title" style="font-size:0.98rem;">Responsive</div>
                <div class="tech-desc">Dapat diakses dari desktop maupun perangkat mobile.</div>
            </div>
            <div class="card feature-card">
                <div class="feature-icon" style="background:#26211C;">🔒</div>
                <div class="tech-title" style="font-size:0.98rem;">Privasi Terjaga</div>
                <div class="tech-desc">Gambar diproses untuk analisis dan tidak disimpan permanen di server.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 10. CARA KERJA
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="section">
        <div class="section-title center">Cara Kerja</div>
        <br>
        <div class="grid grid-4">
            <div class="flow-step">
                <div class="flow-num">1</div>
                <div class="flow-title">Upload Gambar</div>
                <div class="flow-sub">Pilih atau unggah foto buah apel yang ingin dianalisis.</div>
            </div>
            <div class="flow-step">
                <div class="flow-num">2</div>
                <div class="flow-title">AI Analysis</div>
                <div class="flow-sub">Model CNN menganalisis fitur visual pada gambar.</div>
            </div>
            <div class="flow-step">
                <div class="flow-num">3</div>
                <div class="flow-title">Classification</div>
                <div class="flow-sub">Sistem mengklasifikasikan kondisi buah ke 6 kategori.</div>
            </div>
            <div class="flow-step">
                <div class="flow-num">4</div>
                <div class="flow-title">Get Results</div>
                <div class="flow-sub">Terima hasil diagnosa lengkap dengan langkah penanganan.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
seed_divider()

# ------------------------------------------------------------------------------
# 11. COBA SEKARANG — FUNGSI KLASIFIKASI ASLI (UPLOAD / KAMERA)
# ------------------------------------------------------------------------------
st.markdown('<div id="coba" style="padding-top:78px;"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="center" style="max-width:680px;">
        <div class="eyebrow" style="margin-left:auto;margin-right:auto;">Try It Now</div>
        <div class="section-title">Coba AppleScan AI</div>
        <div class="section-sub center">Unggah gambar buah apel Anda dan biarkan AI menganalisis kondisinya dengan cepat.</div>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

# --- 11a. Input Gambar ---
with st.container(border=True):
    st.markdown("#### Input Gambar Buah Apel")

    tab_upload, tab_camera = st.tabs(
        ["Upload Gambar (JPG/PNG)", "Kamera (Ambil Foto)"]
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
        st.markdown("##### Preview Gambar")
        prev_col1, prev_col2, prev_col3 = st.columns([2, 1, 2])
        with prev_col2:
            st.image(image_source, use_container_width=True, caption="Objek Terpilih")

st.markdown("<br>", unsafe_allow_html=True)

# --- 11b. Hasil Diagnosa ---
with st.container(border=True):
    st.markdown("#### 2. Hasil Diagnosa AI")

    if image_source is not None:
        if model is None:
            st.error("Model `apple_disease_model.h5` tidak ditemukan pada direktori utama.")
        else:
            with st.spinner("Menganalisis gambar..."):
                processed_img = preprocess_image(image_source)
                predictions = model(processed_img, training=False).numpy()[0]
                top_idx = int(np.argmax(predictions))

                predicted_class = CLASS_NAMES[top_idx]
                confidence = float(predictions[top_idx]) * 100

            CONFIDENCE_THRESHOLD = 70.0

            # --- KONDISI 1: TIDAK TERDETEKSI (< 70%) ---
            if confidence < CONFIDENCE_THRESHOLD:
                st.markdown(
                    f"""
                    <div class="badge-warning">
                        <h3 style="margin:0;">❓ Status: Tidak Terdeteksi</h3>
                        <p style="margin:5px 0 0 0;">Keyakinan terdeteksi hanya <b>{confidence:.1f}%</b> (di bawah syarat minimal 70.0%). Pastikan foto fokus dan merupakan buah apel yang valid.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Probabilitas Seluruh Kelas:**")
                for c_name in CLASS_NAMES:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(f"**{c_name}**")
                    with c2:
                        st.write("0.0%")
                    st.progress(0.0)

            # --- KONDISI 2: TERDETEKSI (>= 70%) ---
            else:
                if predicted_class == "Healthy":
                    st.markdown(
                        f"""
                        <div class="badge-success">
                            <h3 style="margin:0;">Status: {predicted_class} (Sehat)</h3>
                            <p style="margin:5px 0 0 0;">Tingkat Akurasi / Confidence: <b>{confidence:.2f}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="badge-danger">
                            <h3 style="margin:0;">Status Terinfeksi: {predicted_class}</h3>
                            <p style="margin:5px 0 0 0;">Tingkat Akurasi / Confidence: <b>{confidence:.2f}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📈 Persentase Probabilitas Semua Kelas:**")
                for i, c_name in enumerate(CLASS_NAMES):
                    prob = float(predictions[i])
                    prob_percent = prob * 100

                    col_label, col_val = st.columns([3, 1])
                    with col_label:
                        st.write(f"**{c_name}**")
                    with col_val:
                        st.write(f"**{prob_percent:.1f}%**")

                    st.progress(prob)

                # --- Detail Penyakit ---
                detail = CLASS_DETAILS.get(predicted_class, {})
                if detail:
                    st.markdown("---")
                    st.markdown("#### Detail Penyakit & Solusi Penanganan")
                    st.info(f"**{detail.get('title', '')}**\n\n{detail.get('desc', '')}")
                    st.success(f"**Langkah Penanganan & Pencegahan:**\n\n{detail.get('prevention', '')}")

    else:
        st.write("Silakan masukkan gambar di bagian atas. Hasil analisis lengkap akan ditampilkan secara otomatis pada bagian ini.")

st.markdown('<div style="padding-bottom:78px;"></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 12. FOOTER
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        🍎 AppleScan AI — Skripsi Klasifikasi Kondisi Buah Apel berbasis CNN &amp; Transfer Learning MobileNetV2
    </div>
    """,
    unsafe_allow_html=True,
)
