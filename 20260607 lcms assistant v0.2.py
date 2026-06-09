import re
import streamlit as st
from PIL import Image, ImageEnhance
import easyocr
import numpy as np
from molmass import Formula

st.title("LCMS Assistant v0.2")

@st.cache_resource
def load_reader():
    return easyocr.Reader(["en"])

reader = load_reader()

tab1, tab2, tab3 = st.tabs([
    "Mass Spectra OCR",
    "Formula Calculator",
    "Mass Calculator"
])

with tab1:
    st.header("Load Your Mass Spectra Here")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"]
    )

    numbers = []

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((image.width * 2, image.height * 2))

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        st.image(image, caption="Processed image")

        result = reader.readtext(np.array(image), detail=0)

        st.subheader("Raw OCR Text")
        st.write(result)

        all_text = " ".join(result)
        numbers = re.findall(r"\d+\.\d+|\d+", all_text)

        st.subheader("Detected Numbers")
        st.write(numbers)

    mz_values = []

    for n in numbers:
        try:
            mz_values.append(float(n))
        except:
            pass

    target_diffs = [5, 17, 22, 36, 38]
    matches = []

    for i in range(len(mz_values)):
        for j in range(i + 1, len(mz_values)):
            diff = abs(mz_values[j] - mz_values[i])

            for target in target_diffs:
                if round(diff) == target:
                    matches.append(
                        (mz_values[i], mz_values[j], round(diff, 4), target)
                    )

    st.subheader("Potential Relationships")

    if matches:
        for m1, m2, diff, target in matches:
            st.write(
                f"{m1:.4f} ↔ {m2:.4f} | Δ={diff:.4f} | Match={target}"
            )
    else:
        st.write("No matches found")


with tab2:
    st.header("Formula → Adduct Calculator")

    formula = st.text_input("Molecular Formula", "C8H10N4O2")

    if formula:
        try:
            mw = Formula(formula).isotope.mass

            st.success(f"Monoisotopic Mass = {mw:.5f}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Positive Mode")
                st.write(f"[M+H]+ = {mw + 1.00783:.5f}")
                st.write(f"[M+NH4]+ = {mw + 18.03437:.5f}")
                st.write(f"[M+Na]+ = {mw + 22.98922:.5f}")
                st.write(f"[M+K]+ = {mw + 38.96316:.5f}")

            with col2:
                st.subheader("Negative Mode")
                st.write(f"[M-H]- = {mw - 1.00783:.5f}")
                st.write(f"[M+Cl]- = {mw + 34.96885:.5f}")
                st.write(f"[M+Ac-H]- = {mw + 59.01385:.5f}")

        except Exception:
            st.error("Invalid formula")


with tab3:
    st.header("Neutral Mass → Adduct Calculator")

    mass = st.number_input(
        "Enter Neutral Mass",
        min_value=0.0,
        value=194.08038,
        format="%.5f"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Positive Mode")
        st.write(f"[M+H]+ = {mass + 1.00783:.5f}")
        st.write(f"[M+NH4]+ = {mass + 18.03437:.5f}")
        st.write(f"[M+Na]+ = {mass + 22.98922:.5f}")
        st.write(f"[M+K]+ = {mass + 38.96316:.5f}")

    with col2:
        st.subheader("Negative Mode")
        st.write(f"[M-H]- = {mass - 1.00783:.5f}")
        st.write(f"[M+Cl]- = {mass + 34.96885:.5f}")
        st.write(f"[M+Ac-H]- = {mass + 59.01385:.5f}")

st.caption("LCMS Assistant v0.2 | Developed by Bowen Wang")