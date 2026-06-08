import streamlit as st
from molmass import Formula
from PIL import Image
import pytesseract

st.title("MS Calculator")



st.title("Number Reader from Image")

uploaded_file = st.file_uploader(
    "Upload or drag an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image")

    text = pytesseract.image_to_string(image)

    st.subheader("Recognized text")
    st.write(text)


formula = st.text_input("Molecular Formula", "C8H10N4O2")

if formula:
    try:
        mw = Formula(formula).isotope.mass

        st.write(f"Monoisotopic Mass = {mw:.5f}")
        st.subheader("Positive Mode")

        st.write(f"[M+H]+ = {mw + 1.00783:.5f}")
        st.write(f"[M+NH4]+ = {mw + 18.03437:.5f}")
        st.write(f"[M+Na]+ = {mw + 22.98922:.5f}")
        st.write(f"[M+K]+ = {mw + 38.96316:.5f}")

        st.subheader("Negative Mode")

        st.write(f"[M-H]- = {mw - 1.00783:.5f}")
        st.write(f"[M+Cl]- = {mw + 34.96885:.5f}")
        st.write(f"[M+FA-H]- = {mw + 44.99820:.5f}")
        st.write(f"[M+Ac-H]- = {mw + 59.01385:.5f}")

    except Exception as e:
        st.error("Invalid formula")


st.divider()

st.caption(
    "LCMS Assistant v0.1 | Developed by Bowen Wang"
)
