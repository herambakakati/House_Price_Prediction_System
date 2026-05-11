import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide"
)

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "lightgbm_model.pkl"
DATA_PATH = BASE_DIR / "House Price.csv"

# ==========================================================
# LIVE ASSETS
# ==========================================================
BANNER_URL = "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?q=80&w=1600&auto=format&fit=crop"
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/609/609803.png"

# ==========================================================
# EXACT TRAINED MODEL FEATURES
# DO NOT CHANGE
# ==========================================================
MODEL_FEATURES = [
    "UNDER_CONSTRUCTION",
    "RERA",
    "BHK_NO.",
    "RESALE",
    "LATITUDE",
    "LONGITUDE",
    "LOG_SQFT",
    "POSTED_BY_Dealer",
    "POSTED_BY_Owner"
]

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["LATITUDE", "LONGITUDE", "ADDRESS"])

    parts = df["ADDRESS"].astype(str).str.split(",")

    df["LOCALITY"] = parts.apply(
        lambda x: x[0].strip().title() if len(x) > 0 else "Unknown"
    )

    df["CITY"] = parts.apply(
        lambda x: x[-1].strip().title() if len(x) > 1 else "Unknown"
    )

    return df

model = load_model()
df = load_data()

# ==========================================================
# CITY / LOCALITY MAPPING
# ==========================================================
city_locality_map = (
    df.groupby("CITY")["LOCALITY"]
    .apply(lambda x: sorted(x.unique()))
    .to_dict()
)

coord_lookup = (
    df.groupby(["CITY", "LOCALITY"])[["LATITUDE", "LONGITUDE"]]
    .median()
    .reset_index()
)

# ==========================================================
# PREMIUM CSS
# ==========================================================
st.markdown(f"""
<style>

/* MAIN APP BACKGROUND */
.stApp {{
    background:
        linear-gradient(
            135deg,
            #141e30 0%,
            #243b55 40%,
            #3b82f6 75%,
            #8b5cf6 100%
        ) !important;
}}

/* REMOVE STREAMLIT DEFAULT SPACING */
.block-container {{
    padding-top: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}}

/* HEADER BANNER */
.banner {{
    background-image: url("{BANNER_URL}");
    width: 100%;
    height: 320px;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border-radius: 0 0 30px 30px;
    overflow: hidden;
    box-shadow: 0 25px 60px rgba(0,0,0,0.45);
    margin-top: 0;
}}

/* HEADER OVERLAY */
.overlay {{
    height: 100%;
    background:
        linear-gradient(
            90deg,
            rgba(2,6,23,0.82),
            rgba(15,23,42,0.42)
        );
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 20px;
}}

/* LOGO */
.logo {{
    width: 95px;
    height: 95px;
    background: rgba(255,255,255,0.95);
    border-radius: 50%;
    padding: 14px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.35);
    margin-bottom: 18px;
}}

/* TITLE */
.title {{
    font-size: 46px;
    font-weight: 900;
    background: linear-gradient(
        90deg,
        #ffffff,
        #7dd3fc,
        #c084fc
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}

/* SUBTITLE */
.subtitle {{
    font-size: 18px;
    color: #e2e8f0;
    margin-top: 12px;
    font-weight: 500;
}}

/* CARDS */
.card {{
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.04)
        );
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 24px;
    color: white;
    text-align: center;
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}}

/* RESULT BOX */
.result {{
    background:
        linear-gradient(
            135deg,
            #1e3a5f 0%,
            #2563eb 50%,
            #4f46e5 80%,
            #7c3aed 100%
        );
    padding: 20px;
    border-radius: 24px;
    text-align: center;
    color: white;
    margin-top: 25px;
    box-shadow: 0 25px 55px rgba(0,0,0,0.35);
}}

/* FOOTER */
.footer {{
    margin-top: 35px;
    padding: 18px;
    text-align: center;
    color: #cbd5e1;
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}}

/* LABELS */
label {{
    color: #f8fafc !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}}

/* PREMIUM DROPDOWN */
div[data-baseweb="select"] > div {{
    background:
        linear-gradient(
            135deg,
            #1e3a5f,
            #2b4d73
        ) !important;
    border: 1.5px solid rgba(125,211,252,0.22) !important;
    border-radius: 18px !important;
    min-height: 62px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}}

/* DROPDOWN TEXT */
div[data-baseweb="select"] span {{
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}}

/* NUMBER INPUT */
div[data-testid="stNumberInput"] input {{
    background:
        linear-gradient(
            135deg,
            #1e3a5f,
            #2b4d73
        ) !important;
    color: white !important;
    border: 1.5px solid rgba(192,132,252,0.22) !important;
    border-radius: 18px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}}

/* NUMBER BUTTONS */
div[data-testid="stNumberInput"] button {{
    background:
        linear-gradient(
            135deg,
            #4338ca,
            #7c3aed
        ) !important;
    color: white !important;
    border: none !important;
}}

/* PREMIUM BUTTON */
div.stButton > button {{
    width: 100%;
    height: 72px;
    border-radius: 20px;
    border: none;
    font-size: 22px;
    font-weight: 800;
    color: white;

    background:
        linear-gradient(
            135deg,
            #243b55 0%,
            #3b82f6 45%,
            #6366f1 75%,
            #8b5cf6 100%
        );

    box-shadow:
        0 18px 40px rgba(59,130,246,0.35),
        0 0 22px rgba(139,92,246,0.18);

    transition: all 0.3s ease;
}}

div.stButton > button:hover {{
    transform: translateY(-3px);
    box-shadow:
        0 24px 50px rgba(59,130,246,0.45),
        0 0 28px rgba(139,92,246,0.25);
}}

/* SPACING FIX */
div.element-container {{
    margin-bottom: 10px !important;
}}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(f"""
<div class="banner">
    <div class="overlay">
        <img src="{LOGO_URL}" class="logo">
        <div>
            <div class="title">
                House Price Prediction System
            </div>
            <div class="subtitle">
                Premium AI-Powered Real Estate Valuation Platform
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# PROPERTY CONFIG
# ==========================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("## Property Details")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "Select City",
        sorted(city_locality_map.keys())
    )

    locality = st.selectbox(
        "Select Locality",
        city_locality_map[city]
    )

    posted_by = st.selectbox(
        "Property Listed By",
        ["Dealer", "Owner", "Builder"]
    )

    bhk = st.selectbox(
        "Bedrooms (BHK)",
        list(range(1, 11))
    )

with col2:
    construction = st.selectbox(
        "Construction Status",
        ["Ready to Move", "Under Construction"]
    )

    rera = st.selectbox(
        "RERA Approval",
        ["Yes", "No"]
    )

    resale = st.selectbox(
        "Property Category",
        ["New Property", "Resale"]
    )

    sqft = st.number_input(
        "Property Area (Sq Ft)",
        min_value=100,
        max_value=20000,
        value=1200
    )

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# LOCATION LOOKUP
# ==========================================================
match = coord_lookup[
    (coord_lookup["CITY"] == city) &
    (coord_lookup["LOCALITY"] == locality)
]

latitude = float(match.iloc[0]["LATITUDE"])
longitude = float(match.iloc[0]["LONGITUDE"])

# ==========================================================
# SUMMARY
# ==========================================================
st.markdown("## Property Insights Dashboard")

a, b, c = st.columns(3)

with a:
    st.markdown(f"""
    <div class="card">
        <h3>📍 Location Intelligence</h3>
        <p><b>{city}</b></p>
        <p>{locality}</p>
    </div>
    """, unsafe_allow_html=True)

with b:
    st.markdown(f"""
    <div class="card">
        <h3>🏠 Property Specifications</h3>
        <p><b>{bhk} BHK</b></p>
        <p>{sqft:,} Sq Ft</p>
    </div>
    """, unsafe_allow_html=True)

with c:
    st.markdown(f"""
    <div class="card">
        <h3>🌍 Geo Coordinates</h3>
        <p>Latitude: {latitude:.6f}</p>
        <p>Longitude: {longitude:.6f}</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# PREDICTION
# EXACT LOGIC UNCHANGED
# ==========================================================
if st.button("🔮 Get Property Valuation", use_container_width=True):

    X = pd.DataFrame(
        [[0] * len(MODEL_FEATURES)],
        columns=MODEL_FEATURES
    )

    X["UNDER_CONSTRUCTION"] = 1 if construction == "Under Construction" else 0
    X["RERA"] = 1 if rera == "Yes" else 0
    X["BHK_NO."] = bhk
    X["RESALE"] = 1 if resale == "Resale" else 0
    X["LATITUDE"] = latitude
    X["LONGITUDE"] = longitude
    X["LOG_SQFT"] = np.log1p(sqft)
    X["POSTED_BY_Dealer"] = 1 if posted_by == "Dealer" else 0
    X["POSTED_BY_Owner"] = 1 if posted_by == "Owner" else 0

    pred = np.expm1(model.predict(X)[0])

    if pred >= 100:
        price = f"₹ {pred / 100:.2f} Crore"
    else:
        price = f"₹ {pred:.2f} Lakhs"

    st.markdown(f"""
    <div class="result">
        <h2>Property Valuation Result</h2>
        <h1>{price}</h1>
        <p>
            Smart property valuation based on your provided details
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("""
<div class="footer">
    Developed by Heramba Kakati | AI-Powered House Price Prediction System
</div>
""", unsafe_allow_html=True)