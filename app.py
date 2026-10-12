import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="Breast Cancer Data Mining",
    page_icon="🎀",
    layout="wide"
)

# Background image
image_path = Path("assets/breast_cancer_bg.png")

if image_path.exists():
    image_base64 = base64.b64encode(
        image_path.read_bytes()
    ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255, 245, 250, 0.40),
                    rgba(255, 245, 250, 0.40)
                ),
                url("data:image/png;base64,{image_base64}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            padding-top: 3rem;
        }}
        /* Pink Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #f8c8dc !important;
    }}

    [data-testid="stSidebar"] * {{
        color: #5c0a2d !important;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #831843 !important;
        font-weight: 800 !important;
    }}

        /* Main title */
        h1 {{
            color: #9d174d !important;
            font-weight: 800 !important;
        }}

        /* Sub headings */
        h2, h3 {{
            color: #831843 !important;
            font-weight: 700 !important;
        }}

        /* Normal text */
        p {{
            color: #334155 !important;
            font-size: 17px !important;
        }}

        /* Info box */
        .stAlert {{
            background-color: rgba(255,255,255,0.92) !important;
            border-radius: 15px !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Main title
st.title("🎀 Breast Cancer Wisconsin Prediction")

st.subheader(
    "Data Mining & Machine Learning Analysis Dashboard"
)

st.write(
    "A data mining application for analyzing, "
    "classifying and visualizing the Breast Cancer "
    "Wisconsin dataset."
)

st.info(
    "👈 Use the sidebar to navigate through the project."
)
