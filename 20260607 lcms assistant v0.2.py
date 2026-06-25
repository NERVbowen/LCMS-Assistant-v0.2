import re
import streamlit as st
from streamlit.components.v1 import html
from PIL import Image, ImageEnhance
import itertools
import pandas as pd
from collections import Counter
import easyocr
import numpy as np
from molmass import Formula

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    Crippen,
    Lipinski
)


st.set_page_config(
    page_title="LCMS Assistant",
    page_icon="🧪",
    layout="wide"
)




st.title("LCMS Assistant v0.6")




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

        # Interpretations
        if logp < 1:
            logp_note = "Hydrophilic / very polar; may have weak retention on reverse-phase LC."
        elif logp < 3:
            logp_note = "Moderately polar; often suitable for reverse-phase LC."
        elif logp < 5:
            logp_note = "Lipophilic; likely retained on reverse-phase LC."
        else:
            logp_note = "Highly lipophilic; may show strong reverse-phase retention and lower water solubility."

        if tpsa < 60:
            tpsa_note = "Low polarity; often better retained on reverse-phase LC."
        elif tpsa < 120:
            tpsa_note = "Moderate polarity."
        else:
            tpsa_note = "High polarity; may elute early in reverse-phase LC."

        if hbd == 0:
            hbd_note = "No hydrogen-bond donor groups."
        elif hbd <= 2:
            hbd_note = "Few hydrogen-bond donor groups such as OH or NH."
        else:
            hbd_note = "Multiple hydrogen-bond donor groups."

        if hba <= 2:
            hba_note = "Limited hydrogen-bond accepting ability."
        elif hba <= 6:
            hba_note = "Moderate hydrogen-bond accepting ability."
        else:
            hba_note = "Strong hydrogen-bond accepting ability; may favor adduct formation."

        if rot_bonds <= 5:
            rot_note = "Relatively rigid structure."
        elif rot_bonds <= 10:
            rot_note = "Moderately flexible molecule."
        else:
            rot_note = "Highly flexible molecule; common for oligomers, surfactants, or long-chain additives."

        if heavy_atoms <= 10:
            heavy_note = "Small molecule."
        elif heavy_atoms <= 30:
            heavy_note = "Medium-sized molecule."
        else:
            heavy_note = "Large molecule."

        return {
            "Formula": formula,
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "LogP Interpretation": logp_note,
            "TPSA": round(tpsa, 2),
            "TPSA Interpretation": tpsa_note,
            "HBD": hbd,
            "HBD Interpretation": hbd_note,
            "HBA": hba,
            "HBA Interpretation": hba_note,
            "Rotatable Bonds": rot_bonds,
            "Rotatable Bonds Interpretation": rot_note,
            "Heavy Atoms": heavy_atoms,
            "Heavy Atoms Interpretation": heavy_note,
            "[M+H]+": stars(mh_score),
            "[M+NH4]+": stars(mnh4_score),
            "[M+Na]+": stars(mna_score),
            "[M-H]-": stars(mh_neg_score),
        }

    except Exception as e:
        return {"error": f"Could not process SMILES: {str(e)}"}


home, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Home",
    "📊 Spectra OCR",
    "🧬 Formula Calculator",
    "🌈 UV Predictor",
    "⚡ LC-MS Predictor",
    "✏️ ChemDraw Lite",
    "🧩 Oligomer Finder (Beta)",
    #"ℹ️ About"
])


with home:

    st.title("UVX Chem")

    st.subheader("Scientific Tools for Chemists")

    st.write(
        """
        Free online tools for LC-MS, analytical chemistry,
        chemical structure analysis, molecular property prediction,
        and UV characterization.
        """
    )

    st.link_button(
        "🌐 Visit UVXChem.com",
        "https://uvxchem.com"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 OCR Reader")
        st.write(
            "Extract numbers from mass spectra images and identify potential m/z relationships."
        )

    with col2:
        st.markdown("### 🧬 Formula Calculator")
        st.write(
            "Calculate molecular formula properties, molecular weight, and exact mass."
        )

    with col3:
        st.markdown("### 🌈 UV Predictor")
        st.write(
            "Estimate UV detectability and likely UV response from SMILES structures."
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("### ⚡ LC-MS Predictor")
        st.write(
            "Predict logP, polarity, and common LC-MS ionization behavior."
        )

    with col5:
        st.markdown("### ✏️ ChemDraw Lite")
        st.write(
            "Draw chemical structures and calculate formula, exact mass, and common adducts."
        )
    with col6:
        st.markdown("### 🧩 Oligomer Finder(Beta)")
        st.write(
            "This tool automatically searches for repeat-unit patterns in the form."
        )
    # with col6:
    #     st.markdown("### ℹ️ About")
    #     st.write(
    #         "Free analytical chemistry tools for LC-MS, UV analysis, and molecular property prediction."
    #     )


    st.divider()

    st.info(
        "Select a tool from the tabs above to get started."
    )



with tab2:
    st.header("Mass Spectra OCR + Mass Calculator")

    uploaded_file = st.file_uploader(
        "Upload mass spectra image",
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

    target_diffs = [5, 17, 18, 22, 23, 36, 38, 44, 59]
    matches = []

    for i in range(len(mz_values)):
        for j in range(i + 1, len(mz_values)):
            diff = abs(mz_values[j] - mz_values[i])

            for target in target_diffs:
                if abs(diff - target) <= 0.5:
                    matches.append(
                        (mz_values[i], mz_values[j], round(diff, 4), target)
                    )

    st.subheader("Potential m/z Relationships")

    if matches:
        for m1, m2, diff, target in matches:
            st.write(
                f"{m1:.4f} ↔ {m2:.4f} | Δ={diff:.4f} | Match≈{target}"
            )
    else:
        st.write("No matches found")

    st.divider()

    st.subheader("Mass / Adduct Calculator")

    neutral_mass = st.number_input(
        "Neutral exact mass",
        min_value=0.0,
        value=100.0000,
        format="%.6f"
    )

    positive_adducts = {
        "[M+H]+": 1.007276,
        "[M+Na]+": 22.989218,
        "[M+K]+": 38.963158,
        "[M+NH4]+": 18.033823,
    }

    negative_adducts = {
        "[M-H]-": -1.007276,
        "[M+Cl]-": 34.969402,
        "[M+FA-H]-": 44.998201,
        "[M+Ac-H]-": 59.013851,
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Positive Mode**")
        for adduct, shift in positive_adducts.items():
            st.write(f"{adduct}: {neutral_mass + shift:.4f}")

    with col2:
        st.markdown("**Negative Mode**")
        for adduct, shift in negative_adducts.items():
            st.write(f"{adduct}: {neutral_mass + shift:.4f}")


with tab3:
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


with tab4:
    st.header("SMILES UV Detectability Estimator")

    smiles = st.text_input(
        "Enter SMILES",
        value="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
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

    st.caption(
        "Rule-based estimate of molecular properties and common LC-MS ions from SMILES. "
        "This is a screening tool, not an exact ionization prediction."
    )

    smiles = st.text_input(
        "Enter SMILES",
        value="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
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
            st.write(f"**Estimated LogP (RDKit Crippen):** {result['LogP']}")
            st.caption(result["LogP Interpretation"])

            st.write(f"**TPSA:** {result['TPSA']} Å²")
            st.caption(result["TPSA Interpretation"])

            st.write(f"**HBD:** {result['HBD']}")
            st.caption(result["HBD Interpretation"])

            st.write(f"**HBA:** {result['HBA']}")
            st.caption(result["HBA Interpretation"])

            st.write(f"**Rotatable Bonds:** {result['Rotatable Bonds']}")
            st.caption(result["Rotatable Bonds Interpretation"])

            st.write(f"**Heavy Atoms:** {result['Heavy Atoms']}")
            st.caption(result["Heavy Atoms Interpretation"])

            st.subheader("Likely LC-MS Ions")

            st.write(f"**Mobile phase:** {mobile_phase}")
            st.write(f"**[M+H]+:** {result['[M+H]+']}")
            st.write(f"**[M+NH4]+:** {result['[M+NH4]+']}")
            st.write(f"**[M+Na]+:** {result['[M+Na]+']}")
            st.write(f"**[M-H]-:** {result['[M-H]-']}")

            st.caption(
                "Ion likelihood is rule-based. Actual results depend on source conditions, salt level, solvent, concentration, matrix, and instrument settings."
            )

with tab6:

    from streamlit_ketcher import st_ketcher
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    st.header("ChemDraw Lite")

    input_smiles = st.text_input(
        "Paste SMILES",
        value=""
    )

    default_smiles = (
        input_smiles
        if input_smiles
        else "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    )

    smiles = st_ketcher(default_smiles)

    st.write("SMILES")
    st.code(smiles)

    if smiles:

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:

            formula = rdMolDescriptors.CalcMolFormula(mol)
            mw = Descriptors.MolWt(mol)
            exact_mass = Descriptors.ExactMolWt(mol)

            st.write("**Formula:**", formula)
            st.write("**MW:**", round(mw, 4))
            st.write("**Exact Mass:**", round(exact_mass, 4))

            st.subheader("Common LC-MS Adducts")

            positive_adducts = {
                "[M+H]+": 1.007276,
                "[M+Na]+": 22.989218,
                "[M+K]+": 38.963158,
                "[M+NH4]+": 18.033823,
            }

            negative_adducts = {
                "[M-H]-": -1.007276,
                "[M+Cl]-": 34.969402,
                "[M+FA-H]-": 44.998201,
                "[M+Ac-H]-": 59.013851,
            }

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Positive Mode**")
                for adduct, shift in positive_adducts.items():
                    st.write(f"{adduct}: {exact_mass + shift:.4f}")

            with col2:
                st.markdown("**Negative Mode**")
                for adduct, shift in negative_adducts.items():
                    st.write(f"{adduct}: {exact_mass + shift:.4f}")

        else:
            st.error("Invalid SMILES")


with tab7:
    st.header("🧩 Oligomer Pattern Finder")

    st.caption(
        "🚧 **Beta Version** – This feature is still under active development. "
        "Results are intended for research and screening purposes. "
        "If you find bugs or have suggestions, please leave a message via "
        "[**UVX Chem**](https://uvxchem.com)."
    )

    st.write("This tool automatically searches for repeat-unit patterns in the form:")
    st.latex(r"M = c + n\cdot u_1 + m\cdot u_2")

    mass_text = st.text_area(
        "Paste monoisotopic masses, one per line",
        height=250
    )

    tol = st.number_input("Mass tolerance (Da)", 0.0001, 0.1, 0.005, 0.001, format="%.4f")
    max_n = st.number_input("Max n", 1, 50, 12)
    max_m = st.number_input("Max m", 1, 50, 12)

    known_units = {
        "H2O / H.OH end group": 18.01056468,
        "CH2": 14.01565006,
        "C2H4": 28.03130013,
        "C2H4O / EO / PEG unit": 44.02621475,
        "C4H8": 56.06260026,
        "C3H6O / PO unit": 58.04186481,
        "C6H4": 76.03130013,
        "C6H4O": 92.02621475,
        "C8H8 / styrene unit": 104.06260026,
        "Na-H adduct shift": 21.981943,
        "K-H adduct shift": 37.955882,
    }

    def parse_masses(text):
        values = []
        for line in text.replace(",", "\n").splitlines():
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except:
                    pass
        return sorted(values)

    def unit_guess(x):
        hits = []
        for name, exact in known_units.items():
            err = abs(x - exact)
            if err <= 0.15:
                hits.append((name, exact, err))
        return sorted(hits, key=lambda z: z[2])[:3]

    def get_candidate_units(masses):
        diffs = []
        for a, b in itertools.combinations(masses, 2):
            d = abs(b - a)
            if 10 <= d <= 250:
                diffs.append(round(d, 3))

        common = Counter(diffs).most_common(8)
        units = [x[0] for x in common]

        units += [
            14.01565006,
            28.03130013,
            44.02621475,
            56.06260026,
            58.04186481,
            76.03130013,
            92.02621475,
            104.06260026,
        ]

        clean = []
        for u in sorted(units):
            if not any(abs(u - old) < 0.02 for old in clean):
                clean.append(u)

        return clean

    def fit_one_model(masses, c, u1, u2, tol, max_n, max_m):
        matched = []
        unmatched = []

        lattice = []
        for n in range(int(max_n) + 1):
            for m in range(int(max_m) + 1):
                calc = c + n * u1 + m * u2
                lattice.append((calc, n, m))

        for mass in masses:
            best_calc, best_n, best_m = min(
                lattice,
                key=lambda x: abs(mass - x[0])
            )

            err = mass - best_calc

            if abs(err) <= tol:
                matched.append({
                    "Observed Mass": mass,
                    "Calculated Mass": best_calc,
                    "Error": err,
                    "n": best_n,
                    "m": best_m
                })
            else:
                unmatched.append(mass)

        coverage = len(matched) / len(masses) if masses else 0
        avg_error = sum(abs(x["Error"]) for x in matched) / len(matched) if matched else 999

        known_bonus = 0
        for u in [u1, u2]:
            if unit_guess(u):
                known_bonus += 0.03

        score = coverage - avg_error * 10 + known_bonus

        return {
            "c": c,
            "u1": u1,
            "u2": u2,
            "matched": matched,
            "unmatched": unmatched,
            "coverage": coverage,
            "avg_error": avg_error,
            "score": score
        }


    def find_patterns(masses, tol, max_n, max_m):
        candidate_c = [
            min(masses),
            18.01056468,
        ]

        units = get_candidate_units(masses)
        unit_pairs = list(itertools.combinations(units, 2))

        total = len(candidate_c) * len(unit_pairs)
        current = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []

        for c in candidate_c:
            for u1, u2 in unit_pairs:
                current += 1

                progress_bar.progress(current / total)
                status_text.write(
                    f"Testing model {current}/{total}: "
                    f"c={c:.3f}, u1={u1:.3f}, u2={u2:.3f}"
                )

                r = fit_one_model(masses, c, u1, u2, tol, max_n, max_m)

                if r["coverage"] >= 0.3:
                    results.append(r)

        progress_bar.empty()
        status_text.empty()

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        unique = []
        for r in results:
            duplicate = False
            for old in unique:
                same_c = abs(r["c"] - old["c"]) < 0.02
                same_units = (
                                     abs(r["u1"] - old["u1"]) < 0.02 and abs(r["u2"] - old["u2"]) < 0.02
                             ) or (
                                     abs(r["u1"] - old["u2"]) < 0.02 and abs(r["u2"] - old["u1"]) < 0.02
                             )

                if same_c and same_units:
                    duplicate = True
                    break

            if not duplicate:
                unique.append(r)

        return unique


    def unit_formula(unit_mass):
        guesses = unit_guess(unit_mass)

        if not guesses:
            return f"{unit_mass:.4f}"

        name = guesses[0][0]

        mapping = {
            "CH2": "CH2",
            "C2H4": "C2H4",
            "C2H4O / EO / PEG unit": "C2H4O",
            "C3H6O / PO unit": "C3H6O",
            "C4H8": "C4H8",
            "C6H4": "C6H4",
            "C6H4O": "C6H4O",
            "C8H8 / styrene unit": "C8H8",
        }

        return mapping.get(name, name)

    def show_pattern(r, rank, total_count):
        st.subheader(f"Pattern {rank}")

        st.success(
            f"M = {r['c']:.6f} + n × {r['u1']:.6f} + m × {r['u2']:.6f}"
        )

        st.write(f"Matched: {len(r['matched'])} / {total_count}")
        st.write(f"Coverage: {r['coverage'] * 100:.1f}%")
        st.write(f"Average absolute error: {r['avg_error']:.6f} Da")

        if abs(r["u2"] - 2 * r["u1"]) < 0.03:
            st.info("Note: unit 2 is approximately 2 × unit 1.")
        if abs(r["u1"] - 2 * r["u2"]) < 0.03:
            st.info("Note: unit 1 is approximately 2 × unit 2.")

        st.write("Possible chemical meaning:")

        for label, value in [
            ("Constant c", r["c"]),
            ("Unit 1", r["u1"]),
            ("Unit 2", r["u2"]),
        ]:
            st.write(f"**{label}: {value:.6f}**")
            guesses = unit_guess(value)
            if guesses:
                for name, exact, err in guesses:
                    st.write(f"- {name} | exact {exact:.6f} | error {err:.6f}")
            else:
                st.write("- No close match in small library")

        matched_df = pd.DataFrame(r["matched"])

        st.write("Matched mass assignment:")
        st.dataframe(matched_df, use_container_width=True)

        if r["unmatched"]:
            st.write("Unmatched masses:")
            st.dataframe(
                pd.DataFrame({"Unmatched Mass": r["unmatched"]}),
                use_container_width=True
            )

        st.write("2D lattice map:")

        plot_df = matched_df.copy()

        st.info(
            "How to read this map: each dot is one matched mass. "
            "The x-axis is n, the number of unit 1 repeats. "
            "The y-axis is m, the number of unit 2 repeats. "
            "A straight row or column means a regular oligomer series."
        )



        st.scatter_chart(
            data=plot_df,
            x="n",
            y="m",
            use_container_width=True
        )

        st.caption("Each point represents one matched oligomer. Coordinates are (n, m).")

    if st.button("Find Patterns"):
        masses = parse_masses(mass_text)

        if len(masses) < 5:
            st.warning("Please paste at least 5 masses.")
        else:
            with st.spinner("Searching repeat-unit patterns..."):
                results = find_patterns(masses, tol, max_n, max_m)

            if not results:
                st.error("No clear pattern found.")
            else:
                st.header("Results")

                show_pattern(results[0], 1, len(masses))

                if len(results) > 1:
                    show_pattern(results[1], 2, len(masses))

                st.subheader("Top Alternative Patterns")
                st.write("Find patterns in the form:")
                st.latex(r"M = c + n \cdot u_1 + m \cdot u_2")

                alt = []

                for i, r in enumerate(results[:10], start=1):
                    u1_formula = unit_formula(r["u1"])
                    u2_formula = unit_formula(r["u2"])

                    formula = f"H.({u1_formula})n.({u2_formula})m.OH"

                    alt.append({
                        "Rank": i,
                        "Formula": formula,
                        "c": round(r["c"], 6),
                        "u1": round(r["u1"], 6),
                        "u2": round(r["u2"], 6),
                        "Coverage %": round(r["coverage"] * 100, 1),
                        "Matched": len(r["matched"]),
                        "Unmatched": len(r["unmatched"]),
                        "Avg Error": round(r["avg_error"], 6)
                    })

                st.dataframe(pd.DataFrame(alt), use_container_width=True)


st.caption("LCMS Assistant v0.6 | Developed by Bowen Wang")
