import streamlit as st
from molmass import Formula
from PIL import Image
import easyocr

st.title("MS Calculator")

st.title("Number Reader from Image")
reader = easyocr.Reader(['en'])

result = reader.readtext(np.array(image))

for item in result:
    st.write(item[1])

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
