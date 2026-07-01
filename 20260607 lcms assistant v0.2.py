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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    Crippen,
    Lipinski
)

import pandas as pd


st.set_page_config(
    page_title="LCMS Assistant",
    page_icon="🧪",
    layout="wide"
)




st.title("LCMS Assistant v0.7")




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

def get_solvent_score(smiles, solvent):
    result, error = predict_solvent_compatibility(smiles)
    if error:
        return None

    df = result["table"]
    row = df[df["Solvent"] == solvent]

    if row.empty:
        return None

    return float(row.iloc[0]["Compatibility Score (0–100)"])


def estimate_partition_from_smiles(smiles, solvent_a, solvent_b):
    mol = Chem.MolFromSmiles(smiles)
    special_bonus = 0

    for pattern_name, pattern_info in SPECIAL_EXTRACTION_PATTERNS.items():
        patt = Chem.MolFromSmarts(pattern_info["smarts"])

        if patt is not None and mol.HasSubstructMatch(patt):
            special_bonus += pattern_info["organic_bonus"]
    if mol is None:
        return None, None

    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    charge = Chem.GetFormalCharge(mol)

    d_a = SOLVENT_LAYER_INFO[solvent_a]["density"]
    d_b = SOLVENT_LAYER_INFO[solvent_b]["density"]

    if d_a > d_b:
        organic_solvent = solvent_a
        aqueous_solvent = solvent_b
    else:
        organic_solvent = solvent_b
        aqueous_solvent = solvent_a

    organic_bonus = {
        "DCM": 0.7,
        "EtOAc": 0.4,
        "Diethyl Ether": 0.3,
        "n-Hexane": 1.0
    }.get(organic_solvent, 0.2)

    brine_bonus = 0.4 if aqueous_solvent == "Brine (Sat. NaCl)" else 0

    pseudo_logD = (
            0.9 * logp
            - 0.015 * tpsa
            - 0.25 * hbd
            - 0.08 * hba
            - 2.5 * abs(charge)
            + organic_bonus
            + brine_bonus
            + special_bonus * 2.0
    )
    K = 10 ** pseudo_logD
    organic_percent = round(K / (1 + K) * 100, 1)
    aqueous_percent = round(100 - organic_percent, 1)

    if organic_solvent == solvent_a:
        return organic_percent, aqueous_percent
    else:
        return aqueous_percent, organic_percent


def run_two_solvent_partition(smiles_list, solvent_a, solvent_b):
    results = []

    for i, smi in enumerate(smiles_list, start=1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        percent_a, percent_b = estimate_partition_from_smiles(
            smi,
            solvent_a,
            solvent_b
        )

        if percent_a is None or percent_b is None:
            continue

        results.append({
            "Compound": f"C{i}",
            "SMILES": smi,
            f"{solvent_a} %": percent_a,
            f"{solvent_b} %": percent_b,
            "Preferred Layer": solvent_a if percent_a >= percent_b else solvent_b
        })

    return pd.DataFrame(results)


def run_two_solvent_partition(smiles_list, solvent_a, solvent_b):
    results = []

    for i, smi in enumerate(smiles_list, start=1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        score_a = get_solvent_score(smi, solvent_a)
        score_b = get_solvent_score(smi, solvent_b)

        if score_a is None or score_b is None:
            continue

        percent_a, percent_b = estimate_partition_from_smiles(
            smi,
            solvent_a,
            solvent_b
        )

        results.append({
            "Compound": f"C{i}",
            "SMILES": smi,
            f"{solvent_a} Score": score_a,
            f"{solvent_b} Score": score_b,
            f"{solvent_a} %": percent_a,
            f"{solvent_b} %": percent_b,
            "Preferred Layer": solvent_a if percent_a >= percent_b else solvent_b
        })

    return pd.DataFrame(results)

from matplotlib.patches import Rectangle

def plot_separatory_funnel(df, solvent_a, solvent_b):
    d_a = SOLVENT_LAYER_INFO[solvent_a]["density"]
    d_b = SOLVENT_LAYER_INFO[solvent_b]["density"]

    if d_a > d_b:
        bottom_solvent = solvent_a
        top_solvent = solvent_b
    else:
        bottom_solvent = solvent_b
        top_solvent = solvent_a

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    ax.add_patch(Rectangle((1.2, 1.0), 7.6, 5.6, fill=False, linewidth=1.6))
    ax.add_patch(Rectangle((1.2, 3.8), 7.6, 2.8, alpha=0.18))
    ax.add_patch(Rectangle((1.2, 1.0), 7.6, 2.8, alpha=0.35))
    ax.plot([1.2, 8.8], [3.8, 3.8], color="black", linewidth=1.3)

    ax.text(5, 7.35, f"{top_solvent} / {bottom_solvent}",
            ha="center", va="center", fontsize=12, weight="bold")

    ax.text(5, 4.5, f"Top: {top_solvent}",
            ha="center", va="center", fontsize=10, weight="bold")

    ax.text(5, 1.45, f"Bottom: {bottom_solvent}",
            ha="center", va="center", fontsize=10, weight="bold")

    x_positions = [2.0, 3.5, 5.0, 6.5, 8.0]

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= 5:
            break

        x = x_positions[idx]
        color = colors[idx % len(colors)]

        name = str(row["Compound"])
        if len(name) > 10:
            name = name[:10] + "..."

        top_pct = float(row[f"{top_solvent} %"])
        bottom_pct = float(row[f"{bottom_solvent} %"])

        alpha_top = max(top_pct / 100, 0.15)
        alpha_bottom = max(bottom_pct / 100, 0.15)

        ax.scatter(x, 5.35, s=35 + top_pct * 1.2,
                   color=color, alpha=alpha_top,
                   edgecolors="black", linewidth=0.5, zorder=3)

        ax.text(x, 5.75, f"{name}\n{top_pct:.1f}%",
                ha="center", va="bottom", fontsize=7)

        ax.scatter(x, 2.45, s=35 + bottom_pct * 1.2,
                   color=color, alpha=alpha_bottom,
                   edgecolors="black", linewidth=0.5, zorder=3)

        ax.text(x, 2.85, f"{name}\n{bottom_pct:.1f}%",
                ha="center", va="bottom", fontsize=7)

    return fig

SOLVENT_LAYER_INFO = {
    "Water": {"density": 1.00, "miscible": ["Brine (Sat. NaCl)", "MeOH", "EtOH", "ACN", "IPA", "Acetone", "DMSO", "DMF", "THF"]},
    "MeOH": {"density": 0.79, "miscible": ["Water", "EtOH", "ACN", "IPA", "Acetone", "DMSO", "DMF", "THF"]},
    "EtOH": {"density": 0.79, "miscible": ["Water", "MeOH", "ACN", "IPA", "Acetone", "DMSO", "DMF", "THF"]},
    "ACN": {"density": 0.79, "miscible": ["Water", "MeOH", "EtOH", "IPA", "Acetone", "DMSO", "DMF"]},
    "IPA": {"density": 0.79, "miscible": ["Water", "MeOH", "EtOH", "ACN", "Acetone", "DMSO", "DMF"]},
    "Acetone": {"density": 0.79, "miscible": ["Water", "MeOH", "EtOH", "ACN", "IPA", "DMSO", "DMF"]},
    "DMSO": {"density": 1.10, "miscible": ["Water", "MeOH", "EtOH", "ACN", "IPA", "Acetone", "DMF"]},
    "DMF": {"density": 0.94, "miscible": ["Water", "MeOH", "EtOH", "ACN", "IPA", "Acetone", "DMSO"]},
    "THF": {"density": 0.89, "miscible": ["Water", "MeOH", "EtOH", "IPA"]},
    "EtOAc": {"density": 0.90, "miscible": []},
    "DCM": {"density": 1.33, "miscible": []},
    "Diethyl Ether": {"density": 0.71, "miscible": []},
    "n-Hexane": {"density": 0.66, "miscible": []},
    "Brine (Sat. NaCl)": {"density": 1.20, "miscible": ["Water", "MeOH", "EtOH", "ACN", "IPA", "Acetone", "DMSO", "DMF"]},
}

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple"
]

SPECIAL_EXTRACTION_PATTERNS = {
    "Purine / xanthine-like": {
        "smarts": "c1ncnc2ncnc12",
        "organic_bonus": 1.6
    },
    "Pyridine-like": {
        "smarts": "n1ccccc1",
        "organic_bonus": 0.3
    },
    "Imidazole-like": {
        "smarts": "n1cc[nH]c1",
        "organic_bonus": 0.4
    },
    "Triazine-like": {
        "smarts": "n1cncnc1",
        "organic_bonus": 0.4
    },
}

def clamp_score(x):
    return max(0, min(100, round(x)))


def score_to_recommendation(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Very Good"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Moderate"
    elif score >= 20:
        return "Poor"
    else:
        return "Very Poor"


def predict_solvent_compatibility(smiles):
    """
    Rule-based solvent compatibility estimator.
    Output is for relative solvent ranking only, not experimental solubility.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    charge = Chem.GetFormalCharge(mol)

    polarity = tpsa + 12 * hbd + 6 * hba + 25 * abs(charge)

    scores = {}

    # Polar protic / aqueous
    scores["Water"] = (
        70
        - 18 * logp
        + 0.35 * tpsa
        + 8 * hbd
        + 5 * hba
        + 30 * abs(charge)
        - 0.05 * mw
    )

    scores["Brine (Sat. NaCl)"] = (
            scores["Water"]
            - 10 * logp
            - 3 * hbd
            + 5 * abs(charge)
    )

    scores["MeOH"] = (
        78
        - 8 * abs(logp - 1.2)
        + 0.12 * tpsa
        + 5 * hbd
        + 3 * hba
        + 12 * abs(charge)
        - 0.025 * mw
    )

    scores["EtOH"] = (
        76
        - 7 * abs(logp - 1.8)
        + 0.08 * tpsa
        + 4 * hbd
        + 2.5 * hba
        + 8 * abs(charge)
        - 0.025 * mw
    )

    scores["IPA"] = (
        70
        - 7 * abs(logp - 2.0)
        + 0.05 * tpsa
        + 3 * hbd
        + 2 * hba
        - 0.025 * mw
    )

    # LC-MS organic solvents
    scores["ACN"] = (
        72
        - 7 * abs(logp - 1.5)
        + 0.06 * tpsa
        + 2 * hba
        + 8 * abs(charge)
        - 0.025 * mw
    )

    scores["Acetone"] = (
        70
        - 6 * abs(logp - 1.8)
        + 0.04 * tpsa
        + 2 * hba
        - 0.025 * mw
    )

    # Strong organic dissolving solvents
    scores["DMSO"] = (
        92
        - 3 * abs(logp - 2.0)
        + 0.03 * tpsa
        + 2 * hbd
        + 2 * hba
        + 6 * abs(charge)
        - 0.015 * mw
    )

    scores["DMF"] = (
        88
        - 4 * abs(logp - 2.0)
        + 0.03 * tpsa
        + 2 * hba
        + 4 * abs(charge)
        - 0.018 * mw
    )

    scores["THF"] = (
        72
        - 7 * abs(logp - 2.3)
        + 0.02 * tpsa
        + 1.5 * hba
        - 0.022 * mw
    )

    # Medium/nonpolar organic solvents
    scores["EtOAc"] = (
        65
        - 7 * abs(logp - 2.5)
        - 0.05 * tpsa
        + 1.5 * hba
        - 0.022 * mw
    )

    scores["DCM"] = (
            65
            - 5 * abs(logp - 1.2)
            - 0.06 * tpsa
            - 2 * hbd
            - 0.5 * hba
            - 0.02 * mw
    )

    scores["Diethyl Ether"] = (
        55
        - 8 * abs(logp - 3.0)
        - 0.07 * tpsa
        - 3 * hbd
        + 1 * hba
        - 0.02 * mw
    )

    scores["n-Hexane"] = (
        45
        + 10 * logp
        - 0.20 * tpsa
        - 8 * hbd
        - 4 * hba
        - 25 * abs(charge)
        - 0.015 * mw
    )

    rows = []
    for solvent, raw_score in scores.items():
        score = clamp_score(raw_score)
        rows.append({
            "Solvent": solvent,
            "Compatibility Score (0–100)": score,
            "Recommendation": score_to_recommendation(score)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by="Compatibility Score (0–100)",
        ascending=False
    ).reset_index(drop=True)

    best_overall = df.iloc[0]["Solvent"]

    lcms_candidates = df[
        df["Solvent"].isin(["MeOH", "ACN", "EtOH", "Water"])
    ].sort_values(
        by="Compatibility Score (0–100)",
        ascending=False
    )

    best_lcms = lcms_candidates.iloc[0]["Solvent"]

    avoid = df[df["Compatibility Score (0–100)"] < 25]["Solvent"].tolist()

    summary = {
        "Best Stock Solution": best_overall,
        "Best LC-MS Compatible": best_lcms,
        "Recommended Injection Solvent": f"{best_lcms}/Water or Water/{best_lcms}",
        "Avoid": ", ".join(avoid) if avoid else "None"
    }

    return {
        "descriptors": {
            "MW": round(mw, 2),
            "logP": round(logp, 2),
            "TPSA": round(tpsa, 2),
            "HBD": hbd,
            "HBA": hba,
            "Rotatable Bonds": rot,
            "Formal Charge": charge
        },
        "summary": summary,
        "table": df
    }, None

home, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Home",
    "📊 Spectra OCR",
    "🧬 Formula Calculator",
    "🌈 UV Predictor",
    "⚡ LC-MS Predictor",
    "✏ ChemDraw Lite",
    "🧩 Oligomer Finder (Beta)",
    "🧪 Solvent Compatibility",
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

    # First row
    col1, col2, col3, col4 = st.columns(4)

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

    with col4:
        st.markdown("### ⚡ LC-MS Predictor")
        st.write(
            "Predict logP, polarity, and common LC-MS ionization behavior."
        )

    # Second row
    col5, col6, col7 = st.columns(3)

    with col5:
        st.markdown("### ✏️ ChemDraw Lite")
        st.write(
            "Draw chemical structures and calculate formula, exact mass, and common adducts."
        )

    with col6:
        st.markdown("### 🧩 Oligomer Finder (Beta)")
        st.write(
            "Automatically search for repeat-unit patterns from measured masses."
        )

    with col7:
        st.markdown("### 🧪 Solvent Compatibility")
        st.write(
            "Estimate solvent compatibility from SMILES and simulate two-solvent liquid–liquid extraction."
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
        "Layer percentages are estimated using a descriptor-based pseudo-logD model. "
        "They are intended for visualization only and are not experimental partition coefficients."
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

with tab8:

    st.header("🧪 Solvent Compatibility Estimator")

    smiles = st.text_input(
        "Enter SMILES",
        key="solvent_smiles"
    )

    if st.button("Estimate Solvent Compatibility"):

        result, error = predict_solvent_compatibility(smiles)

        if error:
            st.error(error)

        else:
            st.subheader("📊 Solvent Compatibility")

            st.dataframe(
                result["table"],
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Compatibility Score (0–100) is a rule-based estimate for relative solvent suitability. "
                "It is intended to rank common laboratory solvents and does not represent experimental solubility."
            )
    # ==========================================================
    # Two-Solvent Extraction Simulator
    # ==========================================================

    st.markdown("---")
    st.subheader("🧪 Two-Solvent Extraction Simulator")


    st.write("Enter up to five compounds:")

    df_input = pd.DataFrame({
        "Compound Name": [
            "BHT",
            "Dibutyl phthalate",
            "Caffeine",
            "ε-Caprolactam",
            ""
        ],
        "SMILES": [
            "CC(C)(C)c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1",
            "CCCCOC(=O)c1ccccc1C(=O)OCCCC",
            "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
            "O=C1CCCCCN1",
            ""
        ]
    })

    edited_df = st.data_editor(
        df_input,
        hide_index=True,
        use_container_width=True,
        disabled=False
    )

    solvent_list = [
        "Water",
        "Brine (Sat. NaCl)",
        "MeOH",
        "EtOH",
        "ACN",
        "IPA",
        "Acetone",
        "DMSO",
        "DMF",
        "THF",
        "EtOAc",
        "DCM",
        "Diethyl Ether",
        "n-Hexane"
    ]

    c1, c2 = st.columns(2)

    with c1:
        solvent_a = st.selectbox(
            "Solvent 1",
            solvent_list,
            index=0,
            key="partition_solvent1"
        )

    with c2:
        solvent_b = st.selectbox(
            "Solvent 2",
            solvent_list,
            index=11,
            key="partition_solvent2"
        )

    if st.button("Simulate Extraction"):

        if solvent_a == solvent_b:
            st.error("Please choose two different solvents.")

        else:

            # Check miscibility
            if solvent_b in SOLVENT_LAYER_INFO[solvent_a]["miscible"]:
                st.warning(
                    f"{solvent_a} and {solvent_b} are completely miscible. "
                    "Two liquid layers will not form."
                )

            compound_list = []

            for _, row in edited_df.iterrows():

                smiles = str(row["SMILES"]).strip()

                if smiles == "":
                    continue

                name = str(row["Compound Name"]).strip()

                if name == "":
                    name = f"Compound {len(compound_list) + 1}"

                compound_list.append({
                    "Name": name,
                    "SMILES": smiles
                })

            if len(compound_list) == 0:
                st.error("Please enter at least one SMILES.")

            else:
                df_partition = run_two_solvent_partition(
                    [c["SMILES"] for c in compound_list],
                    solvent_a,
                    solvent_b
                )

                df_partition["Compound"] = [
                    c["Name"] for c in compound_list
                ]

                st.subheader("Partition Prediction")

                st.dataframe(
                    df_partition,
                    use_container_width=True,
                    hide_index=True
                )

                if solvent_b not in SOLVENT_LAYER_INFO[solvent_a]["miscible"]:
                    st.subheader("Extraction Vial")

                    fig = plot_separatory_funnel(
                        df_partition,
                        solvent_a,
                        solvent_b
                    )

                    st.pyplot(fig)

                st.caption(
                    "Layer percentages are estimated using a pseudo-partition model derived from "
                    "relative solvent compatibility scores. They are intended for visualization only "
                    "and are not experimental partition coefficients (logP or logD)."
                )



st.caption("LCMS Assistant v0.7 | Developed by Bowen Wang")
