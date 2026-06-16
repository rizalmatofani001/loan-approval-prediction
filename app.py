import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# IDENTITAS PROYEK
# =====================================================
PROJECT_TITLE = "Dashboard Prediksi Persetujuan Pinjaman"
PROJECT_SUBTITLE = "Implementasi Random Forest Classifier pada Loan Approval Prediction Dataset"
TEAM_MEMBERS = [
    "M. Rizal Matofani (2043231001)",
    "Stefy Aurelia Ginting (2043231101)"
]
AFFILIATION = "Departemen Statistika Bisnis, Fakultas Vokasi, Institut Teknologi Sepuluh Nopember"
COURSE_LECTURER = "Noviyanti Santoso, S.Si., M.Si., Ph.D."

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Loan Approval Prediction Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PATH FILE
# =====================================================
MODEL_PATH = Path("models/random_forest_loan_model.pkl")
METADATA_PATH = Path("models/metadata.json")
METRICS_PATH = Path("models/metrics.csv")
FEATURE_IMPORTANCE_PATH = Path("models/feature_importance.csv")

# =====================================================
# CSS TAMPILAN
# =====================================================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #e9f4ff 0%, #f7fbff 35%, #eef4fb 100%);
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        background:
            linear-gradient(135deg, rgba(11, 61, 99, 0.95), rgba(21, 101, 192, 0.94), rgba(47, 128, 237, 0.94)),
            url("https://images.unsplash.com/photo-1563013544-824ae1b704d3?q=80&w=1600&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        padding: 32px;
        border-radius: 24px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 32px rgba(9, 45, 87, 0.26);
        border: 1px solid rgba(255, 255, 255, 0.22);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 850;
        margin-bottom: 8px;
        letter-spacing: 0.2px;
    }

    .hero-subtitle {
        font-size: 17px;
        line-height: 1.7;
        opacity: 0.98;
        max-width: 1150px;
    }

    .identity-box {
        margin-top: 20px;
        padding: 16px 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        font-size: 14px;
        line-height: 1.65;
        backdrop-filter: blur(3px);
    }

    .section-card {
        background: rgba(255, 255, 255, 0.96);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #e1ebf5;
        box-shadow: 0 8px 22px rgba(20, 52, 89, 0.07);
        margin-bottom: 22px;
    }

    .mini-card {
        background: linear-gradient(180deg, #ffffff, #f8fbff);
        padding: 22px;
        border-radius: 20px;
        border: 1px solid #e1ebf5;
        box-shadow: 0 8px 22px rgba(20, 52, 89, 0.07);
        min-height: 165px;
        position: relative;
        overflow: hidden;
    }

    .mini-card:after {
        content: "";
        position: absolute;
        right: -30px;
        top: -30px;
        width: 90px;
        height: 90px;
        background: rgba(47, 128, 237, 0.10);
        border-radius: 50%;
    }

    .mini-icon {
        font-size: 40px;
        margin-bottom: 8px;
    }

    .mini-title {
        font-size: 18px;
        font-weight: 850;
        color: #0b3d63;
        margin-bottom: 8px;
    }

    .mini-text {
        font-size: 14px;
        color: #4d5d6c;
        line-height: 1.6;
    }

    .highlight-box-blue {
        background: linear-gradient(135deg, #eef6ff, #f8fbff);
        border-left: 7px solid #1565c0;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 16px;
        color: #183b56;
    }

    .highlight-box-green {
        background: linear-gradient(135deg, #ecfdf3, #f6fef9);
        border-left: 7px solid #12b76a;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 16px;
        color: #075e45;
    }

    .highlight-box-red {
        background: linear-gradient(135deg, #fff1f3, #fff8f9);
        border-left: 7px solid #e11d48;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 16px;
        color: #881337;
    }

    .prediction-approve {
        background: linear-gradient(135deg, #ecfdf3, #d1fadf);
        color: #05603a;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #a6f4c5;
        font-size: 20px;
        font-weight: 850;
        box-shadow: 0 8px 22px rgba(18, 183, 106, 0.12);
    }

    .prediction-reject {
        background: linear-gradient(135deg, #fff1f3, #ffe4e8);
        color: #9f1239;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #fecdd3;
        font-size: 20px;
        font-weight: 850;
        box-shadow: 0 8px 22px rgba(225, 29, 72, 0.12);
    }

    .footer-box {
        background: rgba(255,255,255,0.96);
        padding: 18px 22px;
        border-radius: 18px;
        border: 1px solid #e1ebf5;
        font-size: 13px;
        color: #5b6574;
        line-height: 1.65;
        margin-top: 12px;
        box-shadow: 0 8px 22px rgba(20, 52, 89, 0.05);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff, #f7fbff);
        border: 1px solid #e1ebf5;
        padding: 16px 18px;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(20, 52, 89, 0.05);
    }

    div[data-testid="stMetricValue"] {
        color: #0b3d63;
        font-weight: 850;
    }

    div[data-testid="stMetricLabel"] {
        color: #536475;
        font-weight: 650;
    }

    .styled-table-container {
        background: #ffffff;
        padding: 16px;
        border-radius: 18px;
        border: 1px solid #dfeaf5;
        box-shadow: 0 8px 22px rgba(20, 52, 89, 0.06);
        margin: 12px 0 18px 0;
        overflow-x: auto;
    }

    .table-title {
        font-size: 18px;
        font-weight: 850;
        color: #0b3d63;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    table.styled-table {
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        font-size: 14px;
        border-radius: 16px;
        overflow: hidden;
    }

    table.styled-table thead tr {
        background: linear-gradient(135deg, #0b3d63, #1565c0);
        color: #ffffff;
        text-align: left;
        font-weight: 800;
    }

    table.styled-table th,
    table.styled-table td {
        padding: 13px 16px;
        border-bottom: 1px solid #edf2f7;
        vertical-align: middle;
    }

    table.styled-table tbody tr:nth-child(even) {
        background-color: #f8fbff;
    }

    table.styled-table tbody tr:nth-child(odd) {
        background-color: #ffffff;
    }

    table.styled-table tbody tr:hover {
        background-color: #eef6ff;
        transition: 0.2s;
    }

    .pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
    }

    .pill-blue {
        background: #e9f3ff;
        color: #175cd3;
        border: 1px solid #b2ddff;
    }

    .pill-green {
        background: #ecfdf3;
        color: #027a48;
        border: 1px solid #abefc6;
    }

    .pill-red {
        background: #fff1f3;
        color: #c01048;
        border: 1px solid #fecdd3;
    }

    .feature-card {
        background: linear-gradient(180deg, #ffffff, #f8fbff);
        border: 1px solid #e1ebf5;
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(20, 52, 89, 0.05);
    }

    .feature-name {
        color: #0b3d63;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .progress-bg {
        width: 100%;
        background: #e8f1fb;
        border-radius: 999px;
        height: 13px;
        overflow: hidden;
    }

    .progress-fill {
        height: 13px;
        background: linear-gradient(90deg, #1565c0, #2f80ed, #56ccf2);
        border-radius: 999px;
    }

    .progress-text {
        font-size: 12px;
        color: #536475;
        margin-top: 6px;
        font-weight: 700;
    }

    .metric-explain-card {
        background: linear-gradient(180deg, #ffffff, #f8fbff);
        border-radius: 18px;
        border: 1px solid #e1ebf5;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(20, 52, 89, 0.05);
        min-height: 140px;
    }

    .metric-explain-title {
        color: #0b3d63;
        font-size: 17px;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .metric-explain-value {
        color: #1565c0;
        font-size: 28px;
        font-weight: 900;
        margin-bottom: 6px;
    }

    .metric-explain-text {
        color: #536475;
        font-size: 13px;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD FUNCTION
# =====================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None

@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return None

def clean_data(data):
    data = data.copy()
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype(str).str.strip()
    return data

def load_csv_from_upload_or_path(uploaded_file, path_text):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if path_text:
        path = Path(path_text)
        if path.exists():
            return pd.read_csv(path)
        else:
            st.warning("Jalur file tidak ditemukan. Silakan cek kembali lokasi file CSV.")
            return None

    return None

def html_table(df, title="Tabel", icon="📋", max_rows=None):
    df_show = df.copy()
    if max_rows is not None:
        df_show = df_show.head(max_rows)

    html = df_show.to_html(
        index=False,
        classes="styled-table",
        border=0,
        escape=False
    )
    st.markdown(
        f"""
        <div class="styled-table-container">
            <div class="table-title">{icon} {title}</div>
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )

def show_hero():
    st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">💳 {PROJECT_TITLE}</div>
        <div class="hero-subtitle">
            {PROJECT_SUBTITLE}. Dashboard ini dirancang untuk membantu analisis awal
            terhadap kelayakan pengajuan pinjaman berdasarkan data pemohon, seperti pendapatan,
            jumlah pinjaman, tenor, jumlah tanggungan, status pekerjaan, pendidikan, dan skor kredit.
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_footer():
    st.markdown(f"""
    <div class="footer-box">
        <b>{PROJECT_TITLE}</b><br>
        Disusun oleh {TEAM_MEMBERS[0]} dan {TEAM_MEMBERS[1]}.<br>
        Dosen pengampu: {COURSE_LECTURER}.<br>
        {AFFILIATION}.
    </div>
    """, unsafe_allow_html=True)

def format_dataset_preview(df):
    preview = df.head(20).copy()
    for col in preview.columns:
        if preview[col].dtype == "object":
            preview[col] = preview[col].astype(str).str.strip()

    if "loan_status" in preview.columns:
        preview["loan_status"] = preview["loan_status"].astype(str).str.strip().map(
            lambda x: '<span class="pill pill-green">Approved</span>' if x.lower() == "approved"
            else '<span class="pill pill-red">Rejected</span>' if x.lower() == "rejected"
            else x
        )
    return preview

def variable_cards(selected_features):
    labels = {
        "no_of_dependents": ("👨‍👩‍👧", "Jumlah Tanggungan", "Jumlah anggota keluarga atau pihak yang menjadi tanggungan pemohon."),
        "education": ("🎓", "Pendidikan", "Kategori tingkat pendidikan pemohon, yaitu Graduate atau Not Graduate."),
        "self_employed": ("💼", "Status Pekerjaan", "Status apakah pemohon bekerja sendiri/wiraswasta atau bukan."),
        "income_annum": ("💵", "Pendapatan Tahunan", "Total pendapatan tahunan pemohon."),
        "loan_amount": ("🏦", "Jumlah Pinjaman", "Total nominal pinjaman yang diajukan."),
        "loan_term": ("⏳", "Tenor Pinjaman", "Jangka waktu pengembalian pinjaman."),
        "cibil_score": ("⭐", "CIBIL Score", "Skor kredit yang menggambarkan kelayakan kredit pemohon.")
    }

    for i in range(0, len(selected_features), 3):
        cols = st.columns(3)
        for col, feat in zip(cols, selected_features[i:i+3]):
            icon, title, desc = labels.get(feat, ("📌", feat, "Variabel input model."))
            with col:
                st.markdown(f"""
                <div class="mini-card">
                    <div class="mini-icon">{icon}</div>
                    <div class="mini-title">{title}</div>
                    <div class="mini-text">
                        <span class="pill pill-blue">{feat}</span><br><br>
                        {desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)

def metric_cards(metrics_df):
    desc = {
        "Accuracy": "Proporsi prediksi benar dari seluruh data uji.",
        "Precision": "Ketepatan model saat memprediksi pinjaman Approved.",
        "Recall": "Kemampuan model menangkap data yang benar-benar Approved.",
        "F1-Score": "Keseimbangan antara precision dan recall.",
        "ROC-AUC": "Kemampuan model membedakan kelas Approved dan Rejected."
    }

    icons = {
        "Accuracy": "🎯",
        "Precision": "🔎",
        "Recall": "📡",
        "F1-Score": "⚖️",
        "ROC-AUC": "📈"
    }

    for i in range(0, len(metrics_df), 5):
        cols = st.columns(5)
        for col, (_, row) in zip(cols, metrics_df.iloc[i:i+5].iterrows()):
            metric = row["Metric"]
            score = float(row["Score"])
            with col:
                st.markdown(f"""
                <div class="metric-explain-card">
                    <div class="metric-explain-title">{icons.get(metric, "📌")} {metric}</div>
                    <div class="metric-explain-value">{score:.4f}</div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width:{min(score*100, 100):.1f}%"></div>
                    </div>
                    <div class="metric-explain-text">{desc.get(metric, "")}</div>
                </div>
                """, unsafe_allow_html=True)

def feature_importance_cards(fi):
    fi_show = fi.copy().head(7)
    max_val = fi_show["Importance"].max() if len(fi_show) else 1

    for _, row in fi_show.iterrows():
        width = (row["Importance"] / max_val) * 100 if max_val else 0
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-name">🌟 {row["Feature"]}</div>
            <div class="progress-bg">
                <div class="progress-fill" style="width:{width:.1f}%"></div>
            </div>
            <div class="progress-text">Importance: {row["Importance"]:.6f}</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# LOAD MODEL DAN METADATA
# =====================================================
model = load_model()
metadata = load_metadata()
metrics_df_saved = load_metrics()
feature_importance_saved = load_feature_importance()

show_hero()

if model is None or metadata is None:
    st.error(
        "Model atau metadata belum ditemukan. Pastikan folder `models` berisi file "
        "`random_forest_loan_model.pkl`, `metadata.json`, `metrics.csv`, dan `feature_importance.csv`. "
        "Jika belum ada, jalankan `python train_model.py` terlebih dahulu."
    )
    st.stop()

selected_features = metadata["selected_features"]
target_col = metadata["target_col"]

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🏦 Navigasi")
    page = st.radio(
        "Pilih halaman:",
        [
            "Beranda",
            "Eksplorasi Dataset",
            "Evaluasi Model",
            "Feature Importance",
            "Prediksi Pinjaman"
        ]
    )

    st.markdown("---")
    st.markdown("## 👥 Identitas Kelompok")
    st.write("**M. Rizal Matofani**")
    st.caption("2043231001")
    st.write("**Stefy Aurelia Ginting**")
    st.caption("2043231101")
    st.markdown("**Dosen Pengampu**")
    st.caption(COURSE_LECTURER)
    st.markdown("**Departemen**")
    st.caption(AFFILIATION)

    st.markdown("---")
    st.markdown("## 📂 Dataset")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    path_text = st.text_input(
        "Atau masukkan jalur file CSV",
        value="loan_approval_dataset.csv"
    )

    st.markdown("---")


df = load_csv_from_upload_or_path(uploaded_file, path_text)
if df is not None:
    df = clean_data(df)

# =====================================================
# BERANDA
# =====================================================
if page == "Beranda":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-icon">💰</div>
            <div class="mini-title">Fokus Permasalahan</div>
            <div class="mini-text">
                Memprediksi status pengajuan pinjaman berdasarkan profil dan informasi finansial pemohon.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-icon">🌳</div>
            <div class="mini-title">Metode Analisis</div>
            <div class="mini-text">
                Model yang digunakan adalah Random Forest Classifier, yaitu algoritma ensemble berbasis banyak pohon keputusan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-icon">📊</div>
            <div class="mini-title">Output Dashboard</div>
            <div class="mini-text">
                Dashboard menghasilkan prediksi Approved atau Rejected, probabilitas prediksi, evaluasi model, dan fitur paling berpengaruh.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📌 Tujuan Dashboard")
    st.write("""
    Dashboard ini disusun untuk mendukung proyek akhir mata kuliah Pembelajaran Mesin.
    Tujuannya adalah mengimplementasikan algoritma *Random Forest* dalam kasus nyata, yaitu
    prediksi persetujuan pinjaman. Selain pemodelan, dashboard ini juga menyajikan eksplorasi
    dataset, evaluasi performa model, interpretasi fitur, dan simulasi prediksi berbasis input pengguna.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧾 Variabel yang Digunakan")
    st.write("Variabel dibuat dalam bentuk kartu agar lebih mudah dibaca saat dashboard dipresentasikan.")
    variable_cards(selected_features)
    st.markdown("</div>", unsafe_allow_html=True)

    if df is not None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📈 Ringkasan Dataset")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Jumlah Baris", f"{df.shape[0]:,}")
        with c2:
            st.metric("Jumlah Kolom", f"{df.shape[1]:,}")
        with c3:
            if target_col in df.columns:
                approved_count = (df[target_col].astype(str).str.strip().str.lower() == "approved").sum()
                st.metric("Approved", f"{approved_count:,}")
            else:
                st.metric("Approved", "-")
        with c4:
            if target_col in df.columns:
                rejected_count = (df[target_col].astype(str).str.strip().str.lower() == "rejected").sum()
                st.metric("Rejected", f"{rejected_count:,}")
            else:
                st.metric("Rejected", "-")
        st.markdown("</div>", unsafe_allow_html=True)

    show_footer()

# =====================================================
# EKSPLORASI DATASET
# =====================================================
elif page == "Eksplorasi Dataset":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📂 Eksplorasi Dataset")

    if df is None:
        st.warning("Silakan upload dataset atau masukkan jalur file CSV terlebih dahulu.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Jumlah Baris", f"{df.shape[0]:,}")
        with c2:
            st.metric("Jumlah Kolom", f"{df.shape[1]:,}")
        with c3:
            st.metric("Jumlah Duplikasi", f"{df.duplicated().sum():,}")

        preview = format_dataset_preview(df)
        html_table(preview, title="Preview 20 Data Pertama", icon="🧾")

        null_df = df.isnull().sum().reset_index()
        null_df.columns = ["Kolom", "Jumlah Nilai Kosong"]
        null_df["Status"] = null_df["Jumlah Nilai Kosong"].apply(
            lambda x: '<span class="pill pill-green">Lengkap</span>' if x == 0 else '<span class="pill pill-red">Perlu Cek</span>'
        )
        html_table(null_df, title="Pemeriksaan Nilai Kosong", icon="🧹")
    st.markdown("</div>", unsafe_allow_html=True)

    if df is not None:
        required_cols = selected_features + [target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan pada dataset: {missing_cols}")
        else:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📊 Visualisasi Data")

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                df[target_col].value_counts().plot(kind="bar", ax=ax)
                ax.set_title("Distribusi Status Pinjaman")
                ax.set_xlabel("Status Pinjaman")
                ax.set_ylabel("Jumlah Data")
                ax.tick_params(axis="x", rotation=0)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                df["cibil_score"].plot(kind="hist", bins=25, ax=ax)
                ax.set_title("Distribusi CIBIL Score")
                ax.set_xlabel("CIBIL Score")
                ax.set_ylabel("Frekuensi")
                st.pyplot(fig)

            col3, col4 = st.columns(2)

            with col3:
                fig, ax = plt.subplots()
                df.boxplot(column="income_annum", by=target_col, ax=ax)
                ax.set_title("Pendapatan Tahunan Berdasarkan Status Pinjaman")
                ax.set_xlabel("Status Pinjaman")
                ax.set_ylabel("Pendapatan Tahunan")
                plt.suptitle("")
                st.pyplot(fig)

            with col4:
                fig, ax = plt.subplots()
                df.boxplot(column="loan_amount", by=target_col, ax=ax)
                ax.set_title("Jumlah Pinjaman Berdasarkan Status Pinjaman")
                ax.set_xlabel("Status Pinjaman")
                ax.set_ylabel("Jumlah Pinjaman")
                plt.suptitle("")
                st.pyplot(fig)

            desc_df = df[selected_features].describe().reset_index().rename(columns={"index": "Statistik"})
            html_table(desc_df.round(2), title="Statistik Deskriptif Variabel Model", icon="📌")

            st.write("Heatmap korelasi variabel numerik:")
            numeric_cols = [
                "no_of_dependents",
                "income_annum",
                "loan_amount",
                "loan_term",
                "cibil_score"
            ]

            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(corr)
            fig.colorbar(im)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            ax.set_title("Korelasi Variabel Numerik")
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    show_footer()

# =====================================================
# EVALUASI MODEL
# =====================================================
elif page == "Evaluasi Model":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📉 Evaluasi Model Random Forest")

    if metrics_df_saved is not None:
        st.write("Hasil evaluasi model pada data uji disajikan dalam bentuk kartu metrik agar lebih mudah dibaca.")
        metric_cards(metrics_df_saved)

        metrics_table = metrics_df_saved.copy()
        metrics_table["Score"] = metrics_table["Score"].map(lambda x: f"{x:.4f}")
        metrics_table["Keterangan"] = [
            "Ketepatan prediksi keseluruhan",
            "Ketepatan prediksi kelas Approved",
            "Kemampuan menangkap kelas Approved",
            "Keseimbangan precision dan recall",
            "Kemampuan membedakan dua kelas"
        ]
        html_table(metrics_table, title="Ringkasan Evaluasi Model", icon="📊")

        st.markdown("""
        <div class="highlight-box-blue">
        <b>Interpretasi:</b> Nilai evaluasi yang tinggi menunjukkan bahwa model Random Forest
        mampu membedakan kelas pinjaman yang disetujui dan ditolak dengan baik. Namun, evaluasi
        tetap perlu dibaca secara kritis karena performa model sangat bergantung pada kualitas dataset.
        </div>
        """, unsafe_allow_html=True)

    if df is None:
        st.info("Upload dataset atau masukkan jalur CSV untuk menghitung evaluasi ulang.")
    else:
        required_cols = selected_features + [target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan pada dataset: {missing_cols}")
        else:
            data_eval = df[required_cols].copy()
            data_eval[target_col] = data_eval[target_col].astype(str).str.strip().map({
                "Approved": 1,
                "Rejected": 0,
                "approved": 1,
                "rejected": 0
            })
            data_eval = data_eval.dropna()

            X_eval = data_eval[selected_features]
            y_eval = data_eval[target_col]

            y_pred = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1]

            eval_metrics = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                "Score": [
                    accuracy_score(y_eval, y_pred),
                    precision_score(y_eval, y_pred),
                    recall_score(y_eval, y_pred),
                    f1_score(y_eval, y_pred),
                    roc_auc_score(y_eval, y_proba)
                ]
            })

            eval_display = eval_metrics.copy()
            eval_display["Score"] = eval_display["Score"].map(lambda x: f"{x:.4f}")
            html_table(eval_display, title="Evaluasi Ulang Berdasarkan Dataset yang Dimuat", icon="🔁")

            cm = confusion_matrix(y_eval, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(cm, display_labels=["Rejected", "Approved"])
            disp.plot(ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)
    show_footer()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
elif page == "Feature Importance":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🌟 Feature Importance")

    if feature_importance_saved is not None:
        st.write("Feature importance menunjukkan kontribusi relatif setiap variabel dalam proses prediksi model.")

        feature_importance_cards(feature_importance_saved)

        fi_table = feature_importance_saved.copy()
        fi_table["Importance"] = fi_table["Importance"].map(lambda x: f"{x:.6f}")
        html_table(fi_table, title="Tabel Feature Importance", icon="🌟")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(feature_importance_saved["Feature"], feature_importance_saved["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance Random Forest")
        st.pyplot(fig)

        top_feature = feature_importance_saved.iloc[0]["Feature"]
        st.success(
            f"Fitur paling dominan dalam model adalah **{top_feature}**. "
            "Artinya, fitur ini memiliki kontribusi terbesar dalam membantu model membedakan "
            "status pinjaman Approved dan Rejected."
        )

        st.markdown("""
        <div class="highlight-box-blue">
        Feature importance tidak selalu berarti hubungan sebab-akibat. Nilai ini hanya menunjukkan
        seberapa besar fitur digunakan model dalam proses pemisahan kelas berdasarkan pola pada data.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("File feature importance belum ditemukan. Jalankan training model terlebih dahulu.")

    st.markdown("</div>", unsafe_allow_html=True)
    show_footer()

# =====================================================
# PREDIKSI PINJAMAN
# =====================================================
elif page == "Prediksi Pinjaman":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧮 Simulasi Prediksi Pinjaman")
    st.write("""
    Masukkan data pemohon pada form berikut. Model akan memprediksi apakah pengajuan pinjaman
    berpotensi **disetujui (Approved)** atau **ditolak (Rejected)**.
    """)

    with st.form("loan_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            no_of_dependents = st.number_input(
                "Jumlah Tanggungan",
                min_value=0,
                max_value=20,
                value=2,
                step=1
            )

            education = st.selectbox(
                "Pendidikan",
                ["Graduate", "Not Graduate"]
            )

            self_employed = st.selectbox(
                "Status Pekerjaan",
                ["No", "Yes"],
                help="Yes = bekerja sendiri/wiraswasta, No = bukan wiraswasta"
            )

            income_annum = st.number_input(
                "Pendapatan Tahunan",
                min_value=0,
                value=6000000,
                step=100000
            )

        with col2:
            loan_amount = st.number_input(
                "Jumlah Pinjaman",
                min_value=0,
                value=15000000,
                step=100000
            )

            loan_term = st.number_input(
                "Tenor Pinjaman",
                min_value=1,
                max_value=40,
                value=10,
                step=1
            )

            cibil_score = st.number_input(
                "CIBIL Score",
                min_value=300,
                max_value=900,
                value=750,
                step=1
            )

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        input_data = pd.DataFrame([{
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        label = "Approved" if prediction == 1 else "Rejected"

        st.markdown("### 📋 Hasil Prediksi")

        if prediction == 1:
            st.markdown(
                f'<div class="prediction-approve">✅ Hasil Prediksi: {label}<br>'
                f'Pengajuan pinjaman berpotensi <b>disetujui</b>.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-reject">❌ Hasil Prediksi: {label}<br>'
                f'Pengajuan pinjaman berpotensi <b>ditolak</b>.</div>',
                unsafe_allow_html=True
            )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probabilitas Rejected", f"{float(probability[0]):.4f}")
        with c2:
            st.metric("Probabilitas Approved", f"{float(probability[1]):.4f}")

        input_display = input_data.copy()
        input_display = input_display.rename(columns={
            "no_of_dependents": "Jumlah Tanggungan",
            "education": "Pendidikan",
            "self_employed": "Status Pekerjaan",
            "income_annum": "Pendapatan Tahunan",
            "loan_amount": "Jumlah Pinjaman",
            "loan_term": "Tenor Pinjaman",
            "cibil_score": "CIBIL Score"
        })
        html_table(input_display, title="Data Input Pemohon", icon="🧾")

        st.markdown("### 📝 Interpretasi")
        if prediction == 1:
            st.write("""
            Berdasarkan kombinasi data yang dimasukkan, model menilai bahwa profil pemohon
            memiliki kecenderungan yang baik untuk memperoleh persetujuan pinjaman.
            Faktor seperti skor kredit, pendapatan, besar pinjaman, dan tenor dapat memengaruhi hasil prediksi.
            """)
        else:
            st.write("""
            Berdasarkan kombinasi data yang dimasukkan, model memprediksi bahwa pengajuan pinjaman
            memiliki kecenderungan untuk ditolak. Hal ini dapat dipengaruhi oleh skor kredit, jumlah pinjaman,
            tenor, pendapatan, atau kombinasi antarvariabel lain dalam data.
            """)



    st.markdown("</div>", unsafe_allow_html=True)
    show_footer()

# =====================================================
# TENTANG PROYEK
# =====================================================
elif page == "Tentang Proyek":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📘 Tentang Proyek")

    project_info = pd.DataFrame({
        "Komponen": [
            "Judul Proyek",
            "Subjudul",
            "Anggota 1",
            "Anggota 2",
            "Institusi",
            "Dosen Pengampu",
            "Dataset",
            "Algoritma",
            "Target Prediksi"
        ],
        "Keterangan": [
            PROJECT_TITLE,
            PROJECT_SUBTITLE,
            TEAM_MEMBERS[0],
            TEAM_MEMBERS[1],
            AFFILIATION,
            COURSE_LECTURER,
            "Loan Approval Prediction Dataset",
            "Random Forest Classifier",
            "loan_status, yaitu Approved dan Rejected"
        ]
    })
    html_table(project_info, title="Informasi Proyek", icon="📘")

    st.markdown("### 🎯 Manfaat Dashboard")
    st.write("""
    - Menampilkan eksplorasi data pinjaman secara interaktif.
    - Menyajikan performa model klasifikasi secara ringkas.
    - Menunjukkan fitur yang paling berpengaruh terhadap prediksi.
    - Memudahkan simulasi prediksi persetujuan pinjaman berdasarkan input pengguna.
    """)

    st.markdown("### ⚠️ Keterbatasan")
    st.write("""
    - Model hanya belajar dari dataset yang digunakan.
    - Prediksi tidak dapat dianggap sebagai keputusan kredit final.
    - Faktor eksternal seperti kebijakan bank, riwayat transaksi detail, dan dokumen administratif tidak sepenuhnya tercakup dalam dataset.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
    show_footer()
