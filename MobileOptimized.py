import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors
)

# 3D Visualization
import py3Dmol
import streamlit.components.v1 as components

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image
)
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import tempfile
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="C",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#142238; --muted:#63738a; --line:#e2e8f0; --blue:#2458d3; --blue-dark:#173e9c; --surface:#ffffff; --canvas:#f4f7fb; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
.stApp { background:var(--canvas); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(244,247,251,.92); }
[data-testid="stAppViewContainer"] > .main { background:transparent; }
.block-container { max-width:1480px; padding-top:2.0rem; padding-bottom:3rem; }
section.main > div { max-width:100%; }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; color:var(--ink) !important; letter-spacing:-.035em; }
h1 { font-weight:800 !important; font-size:2.35rem !important; line-height:1.15 !important; }
h2 { font-weight:750 !important; font-size:1.65rem !important; }
h3 { font-weight:700 !important; font-size:1.18rem !important; letter-spacing:-.02em; }
p,li,label,[data-testid="stMarkdownContainer"] { color:#334155; }
[data-testid="stSidebar"] { background:#101d33; border-right:1px solid #22334d; }
[data-testid="stSidebar"] * { color:#e7edf7 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#fff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:10px; padding:7px 10px; margin:2px 0; transition:background .15s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#1b2d49; }
[data-testid="stSidebar"] [data-testid="stTextInput"] input { background:#17263d; border-color:#31445f; color:#fff; }
[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:18px 20px; box-shadow:0 3px 12px rgba(20,34,56,.035); }
[data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:.82rem; font-weight:600; }
[data-testid="stMetricValue"] { color:var(--ink); font-family:'Manrope',sans-serif; font-weight:800; }
.stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid #cbd5e1; min-height:42px; font-weight:650; transition:all .18s ease; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background:var(--blue); color:white; border-color:var(--blue); }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--blue); color:var(--blue); transform:translateY(-1px); box-shadow:0 5px 14px rgba(36,88,211,.10); }
.stButton > button[kind="primary"]:hover { background:var(--blue-dark); color:#fff; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div[data-baseweb="select"] { border-radius:10px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stDataFrame"] > div { max-width:100%; }
[data-testid="stAlert"] { border-radius:12px; border-width:1px; }
hr { border-color:var(--line); }
.lab-hero { background:linear-gradient(120deg,#10213b 0%,#183b72 58%,#2458d3 100%); border-radius:22px; padding:34px 36px; color:#fff; margin:0 0 26px; box-shadow:0 16px 36px rgba(18,47,91,.16); }
.lab-hero .eyebrow { color:#b9d0ff; font-size:.76rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:10px; }
.lab-hero h1 { color:#fff !important; margin:0 0 12px; font-size:2.35rem !important; }
.lab-hero p { color:#e0eaff; font-size:1rem; line-height:1.65; max-width:850px; margin:0; }
.feature-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:21px 22px; min-height:145px; box-shadow:0 4px 14px rgba(20,34,56,.025); }
.feature-card h3 { margin:0 0 9px; color:#183b72 !important; }
.feature-card p { margin:0; color:var(--muted); line-height:1.55; font-size:.92rem; }
.section-kicker { color:var(--blue); font-size:.75rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:5px; }
.small-note { color:var(--muted); font-size:.9rem; }
[data-testid="stIFrame"] { width:100% !important; max-width:100%; }
[data-testid="stCaptionContainer"] { color:var(--muted); }
@media (max-width: 768px) {
    .block-container { padding:1rem .75rem 2rem; }
    .lab-hero { padding:24px 18px; border-radius:15px; margin-bottom:20px; overflow:hidden; }
    .lab-hero h1, h1 { font-size:1.75rem !important; line-height:1.12 !important; }
    .lab-hero p { font-size:.94rem; line-height:1.5; }
    h2 { font-size:1.35rem !important; }
    h3 { font-size:1.08rem !important; }
    .feature-card { min-height:0; padding:17px 16px; }
    [data-testid="stMetric"] { padding:13px 14px; }
    [data-testid="stMetricValue"] { font-size:1.35rem; }
    [data-testid="stMetricLabel"] { font-size:.76rem; }
    .stButton > button, .stDownloadButton > button { width:100%; min-height:44px; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { font-size:16px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { font-size:.82rem; }
    [data-testid="stDataFrame"] > div { overflow-x:auto; }
}
@media (max-width: 480px) {
    .block-container { padding-left:.6rem; padding-right:.6rem; }
    .lab-hero { padding:20px 16px; }
    .lab-hero h1, h1 { font-size:1.55rem !important; }
    .section-kicker { font-size:.68rem; letter-spacing:.1em; }
    .feature-card p { font-size:.88rem; }
    [data-testid="stMetricValue"] { font-size:1.2rem; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

# ============================================================
# SHARED MOLECULE INPUT STATE
# ============================================================

# Curated examples used throughout the virtual laboratory.
# "Dichlorophenol" is explicitly defined as 2,4-dichlorophenol to avoid isomer ambiguity.
EXAMPLE_MOLECULES = {
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "2,4-Dichlorophenol": "Oc1c(Cl)cc(Cl)cc1",
    "Toluene": "Cc1ccccc1",
    "Benzene": "c1ccccc1",
}

if "active_smiles" not in st.session_state or st.session_state.active_smiles == "CCO":
    # Migrate the previous default selection to the new default.
    st.session_state.active_smiles = EXAMPLE_MOLECULES["Paracetamol"]

if "selected_example" not in st.session_state or st.session_state.selected_example not in EXAMPLE_MOLECULES:
    st.session_state.selected_example = "Paracetamol"

def load_shared_example():
    """Load the selected example into the shared SMILES field."""
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]

def get_molecule(smiles):
    """Convert SMILES into an RDKit molecule."""

    if not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol if mol is not None and mol.GetNumAtoms() > 0 else None
    except (TypeError, ValueError, RuntimeError):
        return None


# ============================================================
# CALCULATE SIGMA AND PI BONDS
# ============================================================

def calculate_bond_information(mol):
    """
    Calculate:
    - Heavy-atom bonds
    - Sigma (σ) bonds, including bonds to implicit hydrogens
    - Pi (π) bonds
    - Total bond order count (σ + π)
    """

    # Heavy-atom graph bonds (RDKit default molecule representation)
    heavy_atom_bonds = mol.GetNumBonds()

    # Count π bonds using a Kekulé representation for aromatic systems
    kekule_mol = Chem.Mol(mol)

    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        # If Kekulization is not possible, continue with the available form
        pass

    pi_bonds = 0

    for bond in kekule_mol.GetBonds():
        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            # Fallback for aromatic bonds if Kekulization did not clear them
            pi_bonds += 0.5

    # Add implicit hydrogens. Every atom-to-atom connection contains one σ bond.
    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()

    total_bonds = sigma_bonds + pi_bonds

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": total_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


# ============================================================
# CALCULATE COMPLETE MOLECULAR PROPERTIES
# ============================================================

def calculate_properties(mol):

    # Calculate bond information
    bond_info = calculate_bond_information(mol)

    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    properties = {

        "Molecular Formula": molecular_formula,

        "Molecular Weight": round(
            Descriptors.MolWt(mol),
            3
        ),

        "Exact Molecular Weight": round(
            Descriptors.ExactMolWt(mol),
            4
        ),

        "LogP": round(
            Crippen.MolLogP(mol),
            3
        ),

        "TPSA": round(
            rdMolDescriptors.CalcTPSA(mol),
            3
        ),

        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),

        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),

        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),

        "Ring Count": Lipinski.RingCount(mol),

        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),

        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),

        "Number of Atoms": mol.GetNumAtoms(),

        "Heavy Atoms": mol.GetNumHeavyAtoms(),

        # Bond information
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],

        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],

        "Pi (π) Bonds": bond_info["Pi Bonds"],

        "Total Bonds (σ + π)": bond_info["Total Bonds"],

        "Fraction Csp3": round(
            rdMolDescriptors.CalcFractionCSP3(mol),
            3
        ),

        "Molar Refractivity": round(
            Crippen.MolMR(mol),
            3
        )
    }

    return properties

# ============================================================
# GENERATE 2D IMAGE
# ============================================================

def generate_2d_image(mol):

    image = Draw.MolToImage(
        mol,
        size=(700, 500)
    )

    return image


# ============================================================
# GENERATE 3D MOLECULE
# ============================================================

def generate_3d_molecule(mol):

    try:

        mol3d = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()

        params.randomSeed = 42

        status = AllChem.EmbedMolecule(
            mol3d,
            params
        )

        if status != 0:
            return None

        try:

            AllChem.MMFFOptimizeMolecule(
                mol3d
            )

        except Exception:

            try:

                AllChem.UFFOptimizeMolecule(
                    mol3d
                )

            except Exception:
                pass

        return mol3d

    except Exception:

        return None


# ============================================================
# DISPLAY 3D STRUCTURE
# ============================================================

def display_3d_structure(mol3d, style, show_surface=False):

    mol_block = Chem.MolToMolBlock(mol3d)

    view = py3Dmol.view(
        width="100%",
        height=500
    )

    view.addModel(
        mol_block,
        "mol"
    )

    # --------------------------------------------------------
    # BALL AND STICK
    # --------------------------------------------------------

    if style == "Ball and Stick":

        view.setStyle({

            "stick": {
                "radius": 0.15
            },

            "sphere": {
                "scale": 0.30
            }

        })

    # --------------------------------------------------------
    # STICK
    # --------------------------------------------------------

    elif style == "Stick":

        view.setStyle({

            "stick": {
                "radius": 0.20
            }

        })

    # --------------------------------------------------------
    # SPACE FILLING
    # --------------------------------------------------------

    elif style == "Space Filling":

        view.setStyle({

            "sphere": {
                "scale": 1.0
            }

        })

    # --------------------------------------------------------
    # WIREFRAME
    # --------------------------------------------------------

    elif style == "Wireframe":

        view.setStyle({

            "line": {
                "linewidth": 2
            }

        })

    # --------------------------------------------------------
    # CARTOON STYLE
    # --------------------------------------------------------

    elif style == "Cartoon":

        # Cartoon is mainly useful for biomolecules,
        # but this provides a molecular representation.

        view.setStyle({

            "stick": {},

            "sphere": {
                "scale": 0.20
            }

        })

    # --------------------------------------------------------
    # OPTIONAL SURFACE
    # --------------------------------------------------------

    if show_surface:

        view.addSurface(
            py3Dmol.VDW,
            {
                "opacity": 0.7
            }
        )

    view.setBackgroundColor("white")

    view.zoomTo()

    html = view._make_html()

    components.html(
        html,
        height=520
    )


# ============================================================
# CREATE MOLECULAR PDF REPORT
# ============================================================

def generate_molecule_pdf(
    input_smiles,
    mol,
    properties
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=20,
        alignment=TA_CENTER,
        textColor=HexColor("#17365D")
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=HexColor("#17365D")
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )

    story = []

    story.append(
        Paragraph(
            "MOLECULAR PROPERTIES REPORT",
            title_style
        )
    )

    story.append(
        Spacer(1, 0.3 * inch)
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Molecular Information",
            heading_style
        )
    )

    canonical_smiles = Chem.MolToSmiles(mol)

    basic_data = [

        ["Property", "Value"],

        ["Input SMILES", input_smiles],

        ["Canonical SMILES", canonical_smiles],

        [
            "Molecular Formula",
            properties["Molecular Formula"]
        ]
    ]

    basic_table = Table(
        basic_data,
        colWidths=[2 * inch, 4 * inch]
    )

    basic_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("PADDING", (0, 0), (-1, -1), 7)

        ])
    )

    story.append(basic_table)

    story.append(
        Spacer(1, 0.3 * inch)
    )

    # --------------------------------------------------------
    # PROPERTY TABLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Molecular Properties",
            heading_style
        )
    )

    property_data = [
        ["Property", "Value"]
    ]

    for key, value in properties.items():

        property_data.append(
            [key, str(value)]
        )

    property_table = Table(
        property_data,
        colWidths=[3 * inch, 3 * inch],
        repeatRows=1
    )

    property_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("PADDING", (0, 0), (-1, -1), 6)

        ])
    )

    story.append(property_table)

    story.append(
        Spacer(1, 0.3 * inch)
    )

    story.append(
        Paragraph(
            "3. Interpretation",
            heading_style
        )
    )

    interpretation = """
    This report was generated automatically using the
    Cheminformatics Virtual Laboratory. The molecular properties
    were calculated from the chemical structure represented by
    the submitted SMILES notation using the RDKit cheminformatics
    toolkit.
    """

    story.append(
        Paragraph(
            interpretation,
            body_style
        )
    )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    doc.build(story)

    pdf_data = buffer.getvalue()

    buffer.close()

    return pdf_data


# ============================================================
# REPORT IMAGE HELPERS
# ============================================================

def save_2d_structure_png(mol):
    """Create a temporary PNG for inclusion in PDF reports."""
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name


def save_3d_structure_png(mol):
    """
    Create a high-clarity ball-and-stick image for the PDF report.

    The RDKit 3D coordinates are projected onto the best viewing plane so
    elongated molecules fill the page instead of appearing as a tiny object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()

    # Collect 3D coordinates.
    coords = np.array([
        [
            conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z,
        ]
        for i in range(n_atoms)
    ], dtype=float)

    # Principal-component projection chooses the clearest overall view.
    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    # CPK-style colours and readable atom sizes.
    cpk = {
        "H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF",
        "O": "#E53935", "F": "#65C466", "P": "#F39C12",
        "S": "#F4D03F", "Cl": "#27AE60", "Br": "#8E3B2F",
        "I": "#8E44AD",
    }
    sizes = {
        "H": 70, "C": 240, "N": 270, "O": 285, "F": 260,
        "P": 300, "S": 310, "Cl": 310, "Br": 330, "I": 350,
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")

    # Draw bonds first. Multiple bonds are shown with parallel lines.
    for bond in mol3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]

        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue

        # Perpendicular direction for double/triple bond separation.
        px, py = -dy / length, dx / length
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            offsets = (-0.035, 0.035)
        elif btype == Chem.BondType.TRIPLE:
            offsets = (-0.055, 0.0, 0.055)
        else:
            offsets = (0.0,)

        for off in offsets:
            ax.plot(
                [x1 + px * off, x2 + px * off],
                [y1 + py * off, y2 + py * off],
                color="#707070", linewidth=3.0,
                solid_capstyle="round", zorder=1
            )

    # Draw atoms over bonds.
    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(
            x, y,
            s=sizes.get(symbol, 250),
            c=cpk.get(symbol, "#A0A0A0"),
            edgecolors="#222222",
            linewidths=1.0,
            zorder=3
        )

        # Label important/non-carbon hetero atoms only; this keeps the model clean.
        if symbol not in ("H", "C"):
            ax.text(
                x, y, symbol,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="black", zorder=4
            )

    # Tight, equal-aspect limits make the molecule large and clear.
    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    pad = 0.22 * max(width, height)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("3D Molecular Structure (Ball-and-Stick)", fontsize=18, fontweight="bold", pad=18)

    fig.tight_layout(pad=1.2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(
        tmp.name,
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.12
    )
    plt.close(fig)
    return tmp.name


# ============================================================
# PROPERTY INTERPRETATION HELPERS
# ============================================================

def get_property_interpretation(property_name, value, properties):
    """Return a student-friendly interpretation for each reported property."""

    if property_name == "Molecular Formula":
        return (
            "Shows the elemental composition of the molecule and is useful "
            "for understanding its basic chemical composition."
        )

    if property_name == "Molecular Weight":
        return (
            "Represents the average molecular mass. Molecular size and mass "
            "can influence diffusion, formulation and many physicochemical behaviours."
        )

    if property_name == "Exact Molecular Weight":
        return (
            "The monoisotopic mass calculated from exact isotopic masses; "
            "this is especially useful when comparing mass-spectrometry data."
        )

    if property_name == "LogP":
        if value < 0:
            return "A negative value indicates a preference toward the aqueous phase and relatively high hydrophilicity."
        elif value < 1:
            return "Indicates low to moderate lipophilicity, suggesting a reasonable preference for polar environments."
        elif value < 3:
            return "Indicates moderate lipophilicity, often reflecting a balance between hydrophilic and hydrophobic character."
        else:
            return "Indicates relatively high lipophilicity and a stronger preference for non-polar environments."

    if property_name == "TPSA":
        if value < 60:
            return "A relatively low polar surface area, indicating limited overall molecular polarity."
        elif value < 140:
            return "A moderate polar surface area, reflecting the contribution of heteroatoms to molecular polarity."
        else:
            return "A high polar surface area, indicating substantial polarity and strong hydrogen-bonding potential."

    if property_name == "Hydrogen Bond Donors":
        return (
            f"The molecule has {value} donor site(s) capable of donating hydrogen in hydrogen-bond interactions."
        )

    if property_name == "Hydrogen Bond Acceptors":
        return (
            f"The molecule has {value} acceptor site(s) capable of accepting hydrogen in hydrogen-bond interactions."
        )

    if property_name == "Rotatable Bonds":
        if value == 0:
            return "No freely rotatable bonds are detected, suggesting a relatively rigid molecular framework."
        return f"{value} rotatable bond(s) indicate conformational flexibility in the molecular structure."

    if property_name == "Ring Count":
        return f"The molecule contains {value} ring(s), which contribute to its overall shape and structural rigidity."

    if property_name == "Aromatic Rings":
        return f"{value} aromatic ring(s) contribute to aromaticity, planarity and delocalized π-electron character."

    if property_name == "Aliphatic Rings":
        return f"{value} non-aromatic ring(s) contribute to cyclic structure without aromatic π-electron delocalization."

    if property_name == "Number of Atoms":
        return "Counts atoms explicitly present in the RDKit molecular representation; implicit hydrogens are not included here."

    if property_name == "Heavy Atoms":
        return "Counts all non-hydrogen atoms and provides a useful measure of molecular structural size."

    if property_name == "Heavy Atom Bonds":
        return "Counts connections between non-hydrogen atoms in the molecular graph."

    if property_name == "Sigma (σ) Bonds":
        return "Every bonded atom pair contains one σ bond; this value includes bonds involving explicit or added hydrogens."

    if property_name == "Pi (π) Bonds":
        return "π bonds arise from multiple bonds and aromatic systems and are associated with electron delocalization."

    if property_name == "Total Bonds (σ + π)":
        return "Represents the total bond components obtained by adding the calculated σ and π bond counts."

    if property_name == "Fraction Csp3":
        if value < 0.25:
            return "A low sp³ fraction suggests a comparatively planar, unsaturated or aromatic structural character."
        elif value < 0.60:
            return "An intermediate sp³ fraction indicates a balance between saturated and unsaturated structural features."
        return "A high sp³ fraction indicates a more saturated, three-dimensional molecular framework."

    if property_name == "Molar Refractivity":
        return "Relates to molecular volume and electronic polarizability and reflects how the electron cloud responds to an electric field."

    return "This descriptor provides additional quantitative information about the molecular structure and physicochemical behaviour."


def build_key_interpretation_points(properties):
    """Create concise point-wise conclusions for the final report."""
    points = []

    points.append(
        f"<b>• Molecular identity:</b> The submitted structure has molecular formula "
        f"<b>{properties['Molecular Formula']}</b> and molecular weight "
        f"<b>{properties['Molecular Weight']}</b>."
    )

    logp = properties["LogP"]
    if logp < 0:
        lipophilic_text = "more hydrophilic than lipophilic"
    elif logp < 1:
        lipophilic_text = "low in lipophilicity"
    elif logp < 3:
        lipophilic_text = "moderately lipophilic"
    else:
        lipophilic_text = "strongly lipophilic"

    points.append(
        f"<b>• Lipophilicity:</b> LogP = <b>{logp}</b>, indicating that the molecule is "
        f"<b>{lipophilic_text}</b>. This descriptor describes the balance between "
        f"preference for non-polar and polar environments."
    )

    points.append(
        f"<b>• Molecular polarity:</b> TPSA = <b>{properties['TPSA']}</b> Å². "
        f"TPSA reflects the contribution of polar atoms and functional groups to the "
        f"overall surface polarity of the molecule."
    )

    points.append(
        f"<b>• Hydrogen bonding:</b> The structure contains "
        f"<b>{properties['Hydrogen Bond Donors']}</b> hydrogen-bond donor(s) and "
        f"<b>{properties['Hydrogen Bond Acceptors']}</b> acceptor(s), which helps "
        f"describe possible intermolecular interactions."
    )

    points.append(
        f"<b>• Flexibility:</b> {properties['Rotatable Bonds']} rotatable bond(s) "
        f"indicate the degree of conformational freedom available to the molecule."
    )

    points.append(
        f"<b>• Cyclic and aromatic character:</b> The molecule contains "
        f"{properties['Ring Count']} ring(s), including "
        f"{properties['Aromatic Rings']} aromatic ring(s). Aromatic rings are "
        f"associated with delocalized π-electron systems."
    )

    points.append(
        f"<b>• Bonding pattern:</b> The calculated structure contains "
        f"<b>{properties['Sigma (σ) Bonds']}</b> σ bonds and "
        f"<b>{properties['Pi (π) Bonds']}</b> π bonds. This helps distinguish the "
        f"framework-forming σ bonds from multiple-bond and delocalized π bonding."
    )

    points.append(
        f"<b>• Three-dimensional character:</b> Fraction Csp³ = "
        f"<b>{properties['Fraction Csp3']}</b>. This descriptor provides a simple "
        f"indication of the balance between saturated 3D carbon environments and "
        f"planar/unsaturated environments."
    )

    points.append(
        "<b>• Overall conclusion:</b> The calculated descriptors should be interpreted "
        "together with the 2D connectivity and generated 3D geometry to understand "
        "the molecule's composition, bonding, polarity, flexibility and structural character."
    )

    return points


# ============================================================
# GENERATE GENERAL FINAL PROJECT REPORT
# ============================================================

def generate_final_report(student_name, registration_number, input_smiles, mol, properties):
    """Generate a detailed personalized final report for the student's molecule."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=HexColor("#17365D")
    )

    heading_style = ParagraphStyle(
        "ProjectHeading",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=HexColor("#17365D"),
        spaceBefore=8,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ProjectBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY
    )

    small_style = ParagraphStyle(
        "SmallTableText",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9.5
    )

    header_style = ParagraphStyle(
        "HeaderTableText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=9,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    story = []

    story += [
        Spacer(1, 0.25 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.12 * inch),
        Paragraph("FINAL STUDENT MOLECULAR ANALYSIS REPORT", heading_style),
        Spacer(1, 0.18 * inch)
    ]

    student_data = [
        ["Student Name", student_name or "Not provided"],
        ["Registration Number", registration_number or "Not provided"],
        ["Input SMILES", input_smiles],
        ["Canonical SMILES", Chem.MolToSmiles(mol)],
        ["Molecular Formula", properties["Molecular Formula"]]
    ]

    student_table = Table(student_data, colWidths=[2.0 * inch, 4.7 * inch])
    student_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story += [student_table, Spacer(1, 0.22 * inch)]

    # --------------------------------------------------------
    # 1. STRUCTURES
    # --------------------------------------------------------
    story.append(Paragraph("1. Molecular Structures", heading_style))

    img2d = save_2d_structure_png(mol)
    img3d = save_3d_structure_png(mol)

    if img2d and img3d:
        structure_table = Table(
            [[
                Image(img2d, width=3.15 * inch, height=2.25 * inch),
                Image(img3d, width=3.15 * inch, height=2.25 * inch)
            ]],
            colWidths=[3.3 * inch, 3.3 * inch]
        )

        structure_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ]))

        story += [
            structure_table,
            Spacer(1, 0.08 * inch),
            Paragraph(
                "<b>Figure 1.</b> Left: 2D molecular connectivity. "
                "Right: generated 3D ball-and-stick geometry showing the spatial arrangement of atoms.",
                body_style
            )
        ]

    elif img2d:
        story.append(Image(img2d, width=5.8 * inch, height=4.1 * inch))

    story += [Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 2. COMPLETE PROPERTIES WITH INTERPRETATION COLUMN
    # --------------------------------------------------------
    story.append(Paragraph("2. Complete Molecular Properties and Interpretation", heading_style))

    prop_data = [[
        Paragraph("<b>Property</b>", header_style),
        Paragraph("<b>Value</b>", header_style),
        Paragraph("<b>Interpretation</b>", header_style)
    ]]

    for key, value in properties.items():
        interpretation = get_property_interpretation(key, value, properties)
        prop_data.append([
            Paragraph(str(key), small_style),
            Paragraph(str(value), small_style),
            Paragraph(interpretation, small_style)
        ])

    property_table = Table(
        prop_data,
        colWidths=[1.75 * inch, 1.0 * inch, 3.85 * inch],
        repeatRows=1
    )

    property_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4)
    ]))

    story += [property_table, Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 3. POINT-WISE INTERPRETATION
    # --------------------------------------------------------
    story.append(Paragraph("3. Point-wise Molecular Interpretation", heading_style))

    intro = (
        "The following points summarize the most important structural and "
        "physicochemical features calculated from the student's submitted SMILES."
    )
    story.append(Paragraph(intro, body_style))
    story.append(Spacer(1, 0.08 * inch))

    for point in build_key_interpretation_points(properties):
        story.append(Paragraph(point, body_style))
        story.append(Spacer(1, 0.06 * inch))

    # --------------------------------------------------------
    # 4. LEARNING NOTE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.12 * inch),
        Paragraph("4. Learning Note", heading_style),
        Paragraph(
            "The numerical descriptors in this report are computationally calculated "
            "from the submitted molecular structure. They are useful for molecular "
            "comparison and cheminformatics learning, but experimental measurements "
            "may differ depending on conditions and methodology.",
            body_style
        )
    ]

    doc.build(story)

    pdf_data = buffer.getvalue()
    buffer.close()

    for path in [img2d, img3d]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    return pdf_data


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("<div style='font-size:.72rem;font-weight:800;letter-spacing:.16em;color:#9fb6d8;text-transform:uppercase;margin:8px 0 7px'>CHEMINFORMATICS</div><div style='font-family:Manrope,sans-serif;font-size:1.45rem;font-weight:800;color:#ffffff;line-height:1.15'>Virtual Laboratory</div><div style='font-size:.82rem;color:#a9bad2;margin-top:8px'>Molecular learning workspace</div>", unsafe_allow_html=True)

page = st.sidebar.radio(

    "Navigation",

    [

        "Home",

        "Theory",

        "Molecular Visualization",

        "Molecular Properties Report",

        "Structure–Property Analysis",

        "Assessment",

        "Final Project Report"

    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number")

st.sidebar.markdown("""
### Learning Objectives

Students can learn:

- SMILES notation
- Molecular representation
- 2D visualization
- 3D visualization
- Molecular descriptors
- Structure-property relationships
""")


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.markdown("""
    <section class="lab-hero">
      <div class="eyebrow">CHEMISTRY MEETS COMPUTATION</div>
      <h1>Cheminformatics Virtual Laboratory</h1>
      <p>A focused workspace for molecular structure visualization, chemical descriptor analysis, structure–property comparison, and guided assessment. Build a structure from SMILES and explore its 2D representation, 3D geometry, and computed properties.</p>
    </section>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Workspace overview</div>', unsafe_allow_html=True)
    st.subheader("Everything you need to explore a molecule")
    features = [
      ("01 · Molecular visualization", "Validate SMILES and inspect 2D structures alongside interactive 3D molecular models."),
      ("02 · Molecular properties", "Review molecular weight, LogP, TPSA, hydrogen bonding, ring counts, and other descriptors."),
      ("03 · Structure–property analysis", "Compare a curated set of molecules in a consistent table and export the results."),
      ("04 · Theory and concepts", "Review SMILES notation, molecular descriptors, bond types, and visualization models."),
      ("05 · Knowledge assessment", "Check your understanding with a structured cheminformatics assessment."),
      ("06 · Project reporting", "Generate downloadable molecular and student project reports for documentation.")
    ]
    for offset in range(0, len(features), 3):
        cols = st.columns(3, gap="medium")
        for col, (heading, description) in zip(cols, features[offset:offset + 3]):
            col.markdown(f'<div class="feature-card"><h3>{heading}</h3><p>{description}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Curated molecular library</div>', unsafe_allow_html=True)
    st.subheader("Available reference molecules")
    st.caption("These four structures are available throughout the laboratory. Dichlorophenol is specified as the 2,4-isomer to avoid ambiguity.")
    example_df = pd.DataFrame({"Molecule": list(EXAMPLE_MOLECULES.keys()), "SMILES": list(EXAMPLE_MOLECULES.values())})
    st.dataframe(example_df, use_container_width=True, hide_index=True, column_config={"Molecule": st.column_config.TextColumn("Molecule", width="medium"), "SMILES": st.column_config.TextColumn("SMILES notation", width="large")})

    st.markdown("""
    <div style="margin-top:24px;padding:18px 20px;background:#eaf0fb;border:1px solid #d6e2f8;border-radius:14px;">
      <div style="font-weight:750;color:#183b72;margin-bottom:4px;">Developed by K. Jeevani</div>
      <div style="color:#53657e;font-size:.9rem;">Built with Python, Streamlit, and RDKit.</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "Theory":

    st.markdown('<div class="section-kicker">Learning module</div>', unsafe_allow_html=True)
    st.title("Introduction to Cheminformatics")

    st.markdown("""
## What is Cheminformatics?

Cheminformatics combines:

- Chemistry 
- Computer Science 
- Data Analysis 

It uses computational techniques to store, process,
visualize and analyze chemical information.
""")

    st.markdown("""
## SMILES

**SMILES** means:

### Simplified Molecular Input Line Entry System

It represents a molecular structure using a text string.

Examples:

| Molecule | SMILES |
|---|---|
| Paracetamol | CC(=O)NC1=CC=C(C=C1)O |
| 2,4-Dichlorophenol | Oc1c(Cl)cc(Cl)cc1 |
| Toluene | Cc1ccccc1 |
| Benzene | c1ccccc1 |
""")

    st.markdown("""
## Molecular Descriptors

Molecular descriptors provide numerical information
about a chemical structure.

Examples include:

- Molecular Weight
- LogP
- TPSA
- Hydrogen Bond Donors
- Hydrogen Bond Acceptors
- Rotatable Bonds
""")

    st.markdown("""
## 3D Molecular Models

Different visualization models help students understand
the molecular geometry.

###  Ball and Stick

Shows atoms as spheres and bonds as sticks.

###  Stick Model

Emphasizes chemical bonding.

###  Space Filling

Shows the approximate molecular volume.

###  Wireframe

Provides a simplified representation of the molecule.
""")


# ============================================================
# MOLECULAR VISUALIZATION PAGE
# ============================================================

elif page == "Molecular Visualization":

    st.markdown('<div class="section-kicker">Interactive workspace</div>', unsafe_allow_html=True)
    st.title("Molecular Visualization Laboratory")

    st.info("Choose Paracetamol, 2,4-Dichlorophenol, Toluene or Benzene, or enter a valid SMILES. The selected molecule will be used across the laboratory pages.")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles = st.text_input(
            "Enter SMILES",
            key="active_smiles"
        )

    with col2:
        st.selectbox(
            "Choose Example Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example"
        )

        st.button(
            "Load Example",
            on_click=load_shared_example,
            key="load_visualization_example"
        )

    # Keep the latest valid molecule available for the other pages.
    mol = get_molecule(smiles)

    if mol is None:

        st.error(" Invalid SMILES notation.")

    else:

        st.success(" Valid Molecular Structure!")
        st.session_state.latest_smiles = smiles
        st.session_state.latest_properties = calculate_properties(mol)

        col1, col2 = st.columns([1.2, 1])

        with col1:

            st.subheader("2D Molecular Structure")

            image = generate_2d_image(mol)

            st.image(image)

        with col2:

            st.subheader("Molecular Information")

            st.write(
                "**Input SMILES:**",
                smiles
            )

            st.write(
                "**Canonical SMILES:**",
                Chem.MolToSmiles(mol)
            )

            st.write(
                "**Formula:**",
                rdMolDescriptors.CalcMolFormula(mol)
            )

            bond_info = calculate_bond_information(mol)

            st.write(
                "**Heavy Atoms:**",
                mol.GetNumHeavyAtoms()
            )

            st.write(
                "**Total Atoms (including H):**",
                bond_info["Total Atoms Including H"]
            )

            st.write(
                "**Heavy-Atom Bonds:**",
                bond_info["Heavy Atom Bonds"]
            )

            st.write(
                "**Sigma (σ) Bonds:**",
                bond_info["Sigma Bonds"]
            )

            st.write(
                "**Pi (π) Bonds:**",
                bond_info["Pi Bonds"]
            )

            st.write(
                "**Total Bonds (σ + π):**",
                bond_info["Total Bonds"]
            )

        st.divider()

        st.subheader(" Interactive 3D Molecular Structure")

        visualization_style = st.selectbox(

            "Select Molecular Model",

            [

                "Ball and Stick",

                "Stick",

                "Space Filling",

                "Wireframe",

                "Cartoon"
            ]
        )

        show_surface = st.checkbox(
            "Show van der Waals Surface"
        )

        if st.button(" Generate 3D Structure"):

            with st.spinner(
                "Generating 3D molecular structure..."
            ):

                mol3d = generate_3d_molecule(mol)

            if mol3d is None:

                st.error(
                    "Unable to generate the 3D structure."
                )

            else:

                display_3d_structure(

                    mol3d,

                    visualization_style,

                    show_surface
                )


# ============================================================
# COMPLETE MOLECULAR PROPERTIES REPORT
# ============================================================

elif page == "Molecular Properties Report":

    st.markdown('<div class="section-kicker">Descriptor analysis</div>', unsafe_allow_html=True)
    st.title("Complete Molecular Properties Report")

    st.info("""
Enter any valid SMILES notation. The system will automatically
generate a complete molecular properties report.
""")

    st.write("### Select or Enter Your Molecule")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles = st.text_input(
            "Enter New SMILES",
            key="active_smiles"
        )

    with col2:
        st.selectbox(
            "Choose Example Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example"
        )

        st.button(
            "Load Example",
            on_click=load_shared_example,
            key="load_properties_example"
        )

    st.caption("Once you enter a valid SMILES or load an example, the complete 2D/3D structures and molecular properties below are generated for the same molecule.")

    if smiles:

        mol = get_molecule(smiles)

        if mol is None:

            st.error(
                " Invalid SMILES notation. Please check the structure."
            )

        else:

            st.success(
                " Valid Molecular Structure Detected!"
            )

            properties = calculate_properties(mol)

            canonical_smiles = Chem.MolToSmiles(mol)

            # Save the latest student-entered molecule for the Final Project Report
            st.session_state.latest_smiles = smiles
            st.session_state.latest_properties = properties

            # ------------------------------------------------
            # BASIC INFORMATION
            # ------------------------------------------------

            col1, col2 = st.columns([1.2, 1])

            with col1:

                st.subheader(" 2D Molecular Structure")

                image = generate_2d_image(mol)

                st.image(image)

            with col2:

                st.subheader(" Basic Information")

                st.write(
                    "**Input SMILES:**",
                    smiles
                )

                st.write(
                    "**Canonical SMILES:**",
                    canonical_smiles
                )

                st.write(
                    "**Molecular Formula:**",
                    properties["Molecular Formula"]
                )

                st.write(
                    "**Atoms:**",
                    properties["Number of Atoms"]
                )

                st.write(
                    "**Heavy Atoms:**",
                    properties["Heavy Atoms"]
                )

                st.write(
                    "**Heavy-Atom Bonds:**",
                    properties["Heavy Atom Bonds"]
                )

                st.write(
                    "**Sigma (σ) Bonds:**",
                    properties["Sigma (σ) Bonds"]
                )

                st.write(
                    "**Pi (π) Bonds:**",
                    properties["Pi (π) Bonds"]
                )

                st.write(
                    "**Total Bonds (σ + π):**",
                    properties["Total Bonds (σ + π)"]
                )

            st.divider()

            # ------------------------------------------------
            # PROPERTY METRICS
            # ------------------------------------------------

            st.subheader(" Important Molecular Properties")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Molecular Weight",
                f"{properties['Molecular Weight']} g/mol"
            )

            col2.metric(
                "LogP",
                properties["LogP"]
            )

            col3.metric(
                "TPSA",
                f"{properties['TPSA']} Å²"
            )

            col4.metric(
                "Ring Count",
                properties["Ring Count"]
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "H-Bond Donors",
                properties["Hydrogen Bond Donors"]
            )

            col2.metric(
                "H-Bond Acceptors",
                properties["Hydrogen Bond Acceptors"]
            )

            col3.metric(
                "Rotatable Bonds",
                properties["Rotatable Bonds"]
            )

            col4.metric(
                "Fraction Csp3",
                properties["Fraction Csp3"]
            )

            st.divider()

            # ------------------------------------------------
            # COMPLETE TABLE
            # ------------------------------------------------

            st.subheader(" Complete Molecular Property Table")

            report_data = [

                ["Input SMILES", smiles],

                ["Canonical SMILES", canonical_smiles]
            ]

            for key, value in properties.items():

                report_data.append(
                    [key, value]
                )

            report_df = pd.DataFrame(

                report_data,

                columns=[
                    "Property",
                    "Value"
                ]
            )

            st.dataframe(

                report_df,

                use_container_width=True,

                hide_index=True
            )

            # ------------------------------------------------
            # DOWNLOAD CSV
            # ------------------------------------------------

            csv = report_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                "⬇ Download Properties Report (CSV)",

                data=csv,

                file_name="Molecular_Properties_Report.csv",

                mime="text/csv"
            )

            # ------------------------------------------------
            # DOWNLOAD PDF
            # ------------------------------------------------

            if st.button(
                " Generate Molecular PDF Report"
            ):

                with st.spinner(
                    "Generating molecular PDF report..."
                ):

                    pdf = generate_molecule_pdf(

                        smiles,

                        mol,

                        properties
                    )

                st.download_button(

                    "⬇ Download Molecular Properties PDF",

                    data=pdf,

                    file_name="Molecular_Properties_Report.pdf",

                    mime="application/pdf"
                )

            st.divider()

            # ------------------------------------------------
            # 3D VISUALIZATION
            # ------------------------------------------------

            st.subheader(" 3D Molecular Visualization")

            model_style = st.selectbox(

                "Select 3D Model",

                [

                    "Ball and Stick",

                    "Stick",

                    "Space Filling",

                    "Wireframe",

                    "Cartoon"
                ],

                key="properties_3d_style"
            )

            surface = st.checkbox(

                "Show van der Waals Surface",

                key="properties_surface"
            )

            if st.button(
                " Generate 3D Molecular Model",
                key="properties_generate_3d"
            ):

                with st.spinner(
                    "Generating 3D molecular model..."
                ):

                    mol3d = generate_3d_molecule(mol)

                if mol3d is None:

                    st.error(
                        "Unable to generate the 3D structure."
                    )

                else:

                    display_3d_structure(

                        mol3d,

                        model_style,

                        surface
                    )


# ============================================================
# STRUCTURE PROPERTY ANALYSIS
# ============================================================

elif page == "Structure–Property Analysis":

    st.markdown('<div class="section-kicker">Comparative analysis</div>', unsafe_allow_html=True)
    st.title("Structure–Property Analysis")

    st.write("""
Compare the properties of multiple molecules.
Enter one molecule per line in the format:

**Molecule Name, SMILES**
""")

    default_data = """Paracetamol,CC(=O)NC1=CC=C(C=C1)O
2,4-Dichlorophenol,Oc1c(Cl)cc(Cl)cc1
Toluene,Cc1ccccc1
Benzene,c1ccccc1"""

    molecule_input = st.text_area(

        "Enter Molecules",

        value=default_data,

        height=200
    )

    rows = []

    for line in molecule_input.splitlines():

        if "," not in line:

            continue

        name, smiles = line.split(",", 1)

        mol = get_molecule(smiles)

        if mol:

            prop = calculate_properties(mol)

            rows.append({

                "Molecule": name.strip(),

                "MW": prop["Molecular Weight"],

                "LogP": prop["LogP"],

                "TPSA": prop["TPSA"],

                "HBD": prop["Hydrogen Bond Donors"],

                "HBA": prop["Hydrogen Bond Acceptors"],

                "Rotatable Bonds": prop["Rotatable Bonds"],

                "Ring Count": prop["Ring Count"]
            })

    if rows:

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(" Comparison Table")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Comparison Table (CSV)",
            data=csv_data,
            file_name="structure_property_comparison.csv",
            mime="text/csv",
            key="download_structure_property_csv"
        )

    else:
        st.warning("Please enter at least one valid molecule in the format: Molecule Name, SMILES")


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "Assessment":

    st.markdown('<div class="section-kicker">Knowledge check</div>', unsafe_allow_html=True)
    st.title("Cheminformatics Assessment")

    questions = [
        {"question":"What is the full form of SMILES?", "options":["Simplified Molecular Input Line Entry System","Standard Molecular Information Language Encoding System","Simple Molecular Identification and Labeling System"], "answer":"Simplified Molecular Input Line Entry System"},
        {"question":"What does a SMILES string represent?", "options":["A molecular structure using text","A spectroscopy technique","A laboratory instrument"], "answer":"A molecular structure using text"},
        {"question":"Which SMILES represents ethanol?", "options":["CCO","c1ccccc1","CC(=O)O"], "answer":"CCO"},
        {"question":"Which SMILES represents benzene?", "options":["c1ccccc1","CCO","O=C=O"], "answer":"c1ccccc1"},
        {"question":"In SMILES, parentheses are mainly used to indicate:", "options":["Branches","Molecular weight","Atom colour"], "answer":"Branches"},
        {"question":"Which property is commonly associated with molecular lipophilicity?", "options":["LogP","TPSA","HBD"], "answer":"LogP"},
        {"question":"What does TPSA describe?", "options":["Topological Polar Surface Area","Total Pi Surface Area","Thermal Property Surface Analysis"], "answer":"Topological Polar Surface Area"},
        {"question":"What does HBA mean?", "options":["Hydrogen Bond Acceptor","Heavy Bond Atom","Hydrogen Bond Analysis"], "answer":"Hydrogen Bond Acceptor"},
        {"question":"What does HBD mean?", "options":["Hydrogen Bond Donor","Heavy Bond Descriptor","Hydrogen Bond Distance"], "answer":"Hydrogen Bond Donor"},
        {"question":"A double bond contains how many pi (π) bonds?", "options":["1","2","0"], "answer":"1"},
        {"question":"A triple bond contains how many pi (π) bonds?", "options":["2","1","3"], "answer":"2"},
        {"question":"Which model is useful for visualizing approximate molecular volume?", "options":["Space Filling","Wireframe only","Text SMILES"], "answer":"Space Filling"}
    ]

    student_answers = []

    for i, q in enumerate(questions):

        answer = st.radio(
            q["question"],
            q["options"],
            index=None,
            key=f"assessment_v2_{i}"
        )

        student_answers.append(answer)

    if st.button("Submit Assessment"):

        if any(answer is None for answer in student_answers):
            st.warning("Please answer all questions before submitting the assessment.")
            st.stop()

        score = 0

        for i, q in enumerate(questions):

            if student_answers[i] == q["answer"]:
                score += 1

        percentage = (score / len(questions)) * 100

        st.success(
            f"Your Score: {score}/{len(questions)}"
        )

        st.metric(
            "Percentage",
            f"{percentage:.1f}%"
        )

        if percentage >= 80:

            st.success(" Excellent performance!")

        elif percentage >= 50:

            st.info(" Good! Continue practicing.")

        else:

            st.warning(
                " Please review the Theory section and try again."
            )

# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "Final Project Report":

    st.markdown('<div class="section-kicker">Report generation</div>', unsafe_allow_html=True)
    st.title("Final Student Molecular Analysis Report")
    st.write("This report uses the latest valid SMILES entered in the Molecular Properties Report page.")

    name = st.text_input("Student Name for Report", value=st.session_state.get("student_name", ""), key="final_report_name")
    reg = st.text_input("Registration Number for Report", value=st.session_state.get("registration_number", ""), key="final_report_reg")

    default_smiles = st.session_state.get(
        "latest_smiles",
        st.session_state.get("active_smiles", EXAMPLE_MOLECULES["Paracetamol"])
    )
    report_smiles = st.text_input(
        "SMILES for Final Report",
        value=default_smiles,
        key="final_report_smiles"
    )
    report_mol = get_molecule(report_smiles)

    if report_mol is None:
        st.error(" Please enter a valid SMILES notation for the final report.")
    else:
        report_properties = calculate_properties(report_mol)
        st.success(" Valid molecule ready for the final report.")
        st.write("**Molecular Formula:**", report_properties["Molecular Formula"])
        st.write("**Molecular Weight:**", report_properties["Molecular Weight"], "g/mol")
        st.write("**Sigma Bonds:**", report_properties["Sigma (σ) Bonds"])
        st.write("**Pi Bonds:**", report_properties["Pi (π) Bonds"])

        if st.button(" Generate Personalized Final Project Report"):
            with st.spinner("Generating your personalized report with molecular structures and properties..."):
                pdf = generate_final_report(name, reg, report_smiles, report_mol, report_properties)
            st.success(" Final Project Report Generated Successfully!")
            st.download_button(
                label="⬇ Download Personalized Final Project Report PDF",
                data=pdf,
                file_name="Cheminformatics_Final_Molecular_Report.pdf",
                mime="application/pdf"
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cheminformatics Virtual Laboratory | "
    "Developed using Python, Streamlit and RDKit"
)