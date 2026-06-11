import re
import streamlit as st
from PIL import Image, ImageEnhance
import easyocr
import numpy as np
from molmass import Formula
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import Crippen, Lipinski

st.title("LCMS Assistant v0.4")




@st.cache_resource
def load_reader():
    return easyocr.Reader(["en"])

reader = load_reader()


def uv_estimator(smiles):
    try:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return {
                "error": "Unable to read this SMILES string. Please check the SMILES syntax. RELAX."
            }

        mw = Descriptors.MolWt(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)

        aromatic_rings = Descriptors.NumAromaticRings(mol)
        aromatic_atoms = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetIsAromatic()
        )

        double_bonds = sum(
            1 for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.DOUBLE
        )

        hetero_atoms = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetSymbol() not in ["C", "H"]
        )

        carbonyl = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[OX1]"))
        nitro = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3+](=O)[O-]"))
        azo = mol.HasSubstructMatch(Chem.MolFromSmarts("N=N"))
        sulfur = any(atom.GetSymbol() == "S" for atom in mol.GetAtoms())
        halogen = any(
            atom.GetSymbol() in ["F", "Cl", "Br", "I"]
            for atom in mol.GetAtoms()
        )

        pi_score = 0
        pi_score += aromatic_rings * 3
        pi_score += double_bonds * 1.5

        if aromatic_atoms >= 10:
            pi_score += 4
        if aromatic_atoms >= 16:
            pi_score += 4
        if aromatic_atoms >= 22:
            pi_score += 4
        if carbonyl:
            pi_score += 2
        if nitro:
            pi_score += 3
        if azo:
            pi_score += 4
        if sulfur:
            pi_score += 1
        if halogen:
            pi_score += 1
        if hetero_atoms >= 4:
            pi_score += 2
        if mw >= 500:
            pi_score += 1
        if mw >= 800:
            pi_score += 1

        if pi_score >= 22:
            uv210 = "Very strong"
            uv220 = "Very strong"
            uv254 = "Very strong"
            uv280 = "Strong"
            lmax = "280–450+ nm"
            confidence = "Medium-high"
            note = "Large extended π-system / strong chromophore likely."

        elif pi_score >= 14:
            uv210 = "Very strong"
            uv220 = "Strong"
            uv254 = "Strong"
            uv280 = "Moderate–Strong"
            lmax = "250–380 nm"
            confidence = "Medium"
            note = "Aromatic or conjugated system likely gives strong UV response."

        elif pi_score >= 8:
            uv210 = "Strong"
            uv220 = "Moderate–Strong"
            uv254 = "Moderate"
            uv280 = "Weak–Moderate"
            lmax = "220–300 nm"
            confidence = "Medium"
            note = "Some chromophore present; UV response likely depends on concentration."

        elif pi_score >= 4:
            uv210 = "Moderate"
            uv220 = "Weak–Moderate"
            uv254 = "Weak"
            uv280 = "Poor"
            lmax = "200–240 nm"
            confidence = "Low"
            note = "Weak chromophore; mainly low-wavelength UV."

        else:
            uv210 = "Poor–Weak"
            uv220 = "Poor"
            uv254 = "Poor"
            uv280 = "Poor"
            lmax = "<200–220 nm"
            confidence = "Low"
            note = "No obvious UV chromophore detected."

        return {
            "Formula": formula,
            "MW": round(mw, 2),
            "Aromatic rings": aromatic_rings,
            "Aromatic atoms": aromatic_atoms,
            "Double bonds": double_bonds,
            "Hetero atoms": hetero_atoms,
            "Carbonyl": "Yes" if carbonyl else "No",
            "Nitro": "Yes" if nitro else "No",
            "Azo": "Yes" if azo else "No",
            "Sulfur": "Yes" if sulfur else "No",
            "Halogen": "Yes" if halogen else "No",
            "Pi score": round(pi_score, 1),
            "Conjugated system": "Yes" if pi_score >= 8 else "No/Weak",
            "210 nm": uv210,
            "220 nm": uv220,
            "254 nm": uv254,
            "280 nm": uv280,
            "Estimated λmax": lmax,
            "Confidence": confidence,
            "Note": note
        }

    except Exception as e:
        return {
            "error": f"Could not process this SMILES. Error: {str(e)}"
        }


def smiles_lcms_ion_predictor(smiles, mobile_phase):
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return {"error": "Invalid SMILES string. Please check the structure."}

        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)

        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        n_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "N")
        o_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")

        has_basic_n = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]"))
        has_carboxylic_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OX2H1]"))
        has_sulfonic_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)[OX2H1]"))
        has_phenol = mol.HasSubstructMatch(Chem.MolFromSmarts("c[OX2H1]"))
        has_ether = mol.HasSubstructMatch(Chem.MolFromSmarts("[OD2]([#6])[#6]"))
        has_carbonyl = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[OX1]"))

        mh_score = 1
        mnh4_score = 1
        mna_score = 1
        mh_neg_score = 1

        # [M+H]+
        if has_basic_n:
            mh_score += 3
        if n_atoms >= 1:
            mh_score += 1
        if hba >= 2:
            mh_score += 1

        # [M+NH4]+
        if o_atoms >= 2:
            mnh4_score += 2
        if has_ether:
            mnh4_score += 2
        if has_carbonyl:
            mnh4_score += 1
        if hba >= 3:
            mnh4_score += 1

        # [M+Na]+
        if o_atoms >= 2:
            mna_score += 2
        if hba >= 3:
            mna_score += 1
        if has_ether:
            mna_score += 2
        if has_carbonyl:
            mna_score += 1
        if mw > 500:
            mna_score += 1

        # [M-H]-
        if has_carboxylic_acid:
            mh_neg_score += 4
        if has_sulfonic_acid:
            mh_neg_score += 4
        if has_phenol:
            mh_neg_score += 2
        if hbd >= 2:
            mh_neg_score += 1

        # Mobile phase effect
        if mobile_phase == "Formic Acid":
            mh_score += 1
        elif mobile_phase == "Acetic Acid":
            mh_score += 1
        elif mobile_phase == "Ammonium Formate":
            mnh4_score += 2
        elif mobile_phase == "Ammonium Acetate":
            mnh4_score += 3
        elif mobile_phase == "Sodium Acetate":
            mna_score += 3

        mh_score = max(0, min(5, mh_score))
        mnh4_score = max(0, min(5, mnh4_score))
        mna_score = max(0, min(5, mna_score))
        mh_neg_score = max(0, min(5, mh_neg_score))

        def stars(score):
            return "★" * score + "☆" * (5 - score)

        return {
            "Formula": formula,
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "TPSA": round(tpsa, 2),
            "HBD": hbd,
            "HBA": hba,
            "Rotatable Bonds": rot_bonds,
            "Heavy Atoms": heavy_atoms,
            "[M+H]+": stars(mh_score),
            "[M+NH4]+": stars(mnh4_score),
            "[M+Na]+": stars(mna_score),
            "[M-H]-": stars(mh_neg_score),
        }

    except Exception as e:
        return {"error": f"Could not process SMILES: {str(e)}"}


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Mass Spectra OCR",
    "Formula Calculator",
    "Mass Calculator",
    "UV Estimator",
    "LC-MS Ion Estimator"
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

with tab4:
    st.header("SMILES UV Detectability Estimator")

    smiles = st.text_input(
        "Enter SMILES",
        value="C1=CC=C2C(=C1)C=CC3=C2C4=CC=CC=C4C=C3",
        key="uv_smiles"
    )

    if smiles:
        result = uv_estimator(smiles)

        if "error" in result:
            st.error(result["error"])

        else:
            mol = Chem.MolFromSmiles(smiles)

            # if mol is not None:
            #     try:
            #         from rdkit.Chem import Draw

            #         st.subheader("Chemical Structure")

            #         img = Draw.MolToImage(
            #             mol,
            #             size=(300, 220)
            #         )

            #         st.image(
            #             img,
            #             caption="Structure from SMILES",
            #             width=300
            #         )

            #     except Exception:
            #         st.warning(
            #             "Structure image could not be displayed, but UV estimation is still available."
            #         )

            st.subheader("Molecular Information")

            st.write(f"**Formula:** {result['Formula']}")
            st.write(f"**MW:** {result['MW']}")
            st.write(f"**Aromatic rings:** {result['Aromatic rings']}")
            st.write(f"**Aromatic atoms:** {result['Aromatic atoms']}")
            st.write(f"**Double bonds:** {result['Double bonds']}")
            st.write(f"**Hetero atoms:** {result['Hetero atoms']}")
            st.write(f"**π-score:** {result['Pi score']}")
            st.write(f"**Conjugated system:** {result['Conjugated system']}")

            st.subheader("Functional Group Flags")

            st.write(f"**Carbonyl:** {result['Carbonyl']}")
            st.write(f"**Nitro:** {result['Nitro']}")
            st.write(f"**Azo:** {result['Azo']}")
            st.write(f"**Sulfur:** {result['Sulfur']}")
            st.write(f"**Halogen:** {result['Halogen']}")

            st.subheader("Predicted UV Detectability")

            st.write(f"**210 nm:** {result['210 nm']}")
            st.write(f"**220 nm:** {result['220 nm']}")
            st.write(f"**254 nm:** {result['254 nm']}")
            st.write(f"**280 nm:** {result['280 nm']}")

            st.subheader("Estimated UV Range")

            st.write(f"**Estimated λmax:** {result['Estimated λmax']}")
            st.write(f"**Confidence:** {result['Confidence']}")

            st.info(result["Note"])


with tab5:
    st.header("SMILES-Based LC-MS Ionization Estimator")

    smiles = st.text_input(
        "Enter SMILES",
        value="CCOCCOCCO",
        key="lcms_ion_smiles"
    )

    mobile_phase = st.selectbox(
        "Mobile Phase Additive",
        [
            "Formic Acid",
            "Acetic Acid",
            "Ammonium Formate",
            "Ammonium Acetate",
            "Sodium Acetate"
        ],
        key="mobile_phase"
    )

    if smiles:
        result = smiles_lcms_ion_predictor(smiles, mobile_phase)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Molecular Properties")

            st.write(f"**Formula:** {result['Formula']}")
            st.write(f"**MW:** {result['MW']}")
            st.write(f"**LogP:** {result['LogP']}")
            st.write(f"**TPSA:** {result['TPSA']} Å²")
            st.write(f"**HBD:** {result['HBD']}")
            st.write(f"**HBA:** {result['HBA']}")
            st.write(f"**Rotatable Bonds:** {result['Rotatable Bonds']}")
            st.write(f"**Heavy Atoms:** {result['Heavy Atoms']}")

            st.subheader("Likely LC-MS Ions")

            st.write(f"**Mobile phase:** {mobile_phase}")
            st.write(f"**[M+H]+:** {result['[M+H]+']}")
            st.write(f"**[M+NH4]+:** {result['[M+NH4]+']}")
            st.write(f"**[M+Na]+:** {result['[M+Na]+']}")
            st.write(f"**[M-H]-:** {result['[M-H]-']}")

            st.caption(
                "Rule-based estimate only. Actual ionization depends on source settings, concentration, solvent, salts, matrix, and instrument conditions."
            )


st.caption("LCMS Assistant v0.4 | Developed by Bowen Wang")
