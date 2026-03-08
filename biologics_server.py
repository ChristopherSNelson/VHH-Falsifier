"""
VHH-Falsifier — FastMCP Biophysical Profiling Server
=====================================================

Provides deterministic, agent-readable tools for evaluating
developability constraints on VHH (nanobody) sequences.

Biophysical thresholds follow standard developability guidance:
  - pI  > 7.5   → acceptable (avoids precipitation near physiological pH)
  - GRAVY ≤ 0.0 → acceptable (hydrophilic → lower aggregation propensity)

References:
  Kyte & Doolittle (1982) for hydropathy; IPC2 / Bjellqvist for pI.
  Escalante blog for the broader Nipah VHH design strategy:
  https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from fastmcp import FastMCP

# --- Logging setup ------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("vhh-falsifier")
logger.setLevel(logging.INFO)

_handler = logging.FileHandler(LOG_DIR / "biologics_server.log")
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_handler)

mcp = FastMCP("VHH-Falsifier")


def _clean_sequence(seq: str) -> str:
    """Strip whitespace/digits and upper-case a raw protein sequence."""
    return re.sub(r"[^A-Za-z]", "", seq).upper()


@mcp.tool()
def calculate_biophysical_profile(sequence: str) -> str:
    """Calculate pI and GRAVY for a protein sequence and flag aggregation risk.

    Args:
        sequence: Single-letter amino-acid sequence (whitespace/digits ignored).

    Returns:
        JSON string with fields:
            sequence_length, isoelectric_point, gravy,
            pI_flag, gravy_flag, overall_risk, flags (list of human-readable warnings).
    """
    seq = _clean_sequence(sequence)

    if not seq:
        return json.dumps({"error": "Empty or invalid sequence provided."})

    # Reject non-standard residues that ProteinAnalysis cannot handle
    invalid = set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
    if invalid:
        return json.dumps(
            {
                "error": f"Non-standard residues detected: {sorted(invalid)}. "
                "Remove or replace them before profiling.",
            }
        )

    analysis = ProteinAnalysis(seq)

    pi = round(analysis.isoelectric_point(), 2)
    gravy = round(analysis.gravy(), 4)

    # --- Flag logic -----------------------------------------------------------
    flags: list[str] = []

    pi_flag = "PASS" if pi > 7.5 else "FAIL"
    if pi_flag == "FAIL":
        flags.append(
            f"pI = {pi} (< 7.5): High risk of precipitation near physiological pH. "
            "Consider charge-engineering (Lys/Arg substitutions in framework)."
        )

    gravy_flag = "PASS" if gravy <= 0.0 else "FAIL"
    if gravy_flag == "FAIL":
        flags.append(
            f"GRAVY = {gravy} (> 0.0): Elevated hydrophobicity — aggregation-prone. "
            "Inspect solvent-exposed hydrophobic patches."
        )

    overall_risk = (
        "Low"
        if (pi_flag == "PASS" and gravy_flag == "PASS")
        else "High-Risk for Aggregation"
    )

    report = {
        "sequence_length": len(seq),
        "isoelectric_point": pi,
        "gravy": gravy,
        "pI_flag": pi_flag,
        "gravy_flag": gravy_flag,
        "overall_risk": overall_risk,
        "flags": flags,
    }

    result = json.dumps(report, indent=2)
    logger.info("biophysical_profile | %s", result)
    return result


# --- Structural liability patterns --------------------------------------------
# Deterministic regex rules — no LLM inference. These represent well-characterized
# post-translational modification hotspots that cause manufacturing failure.

_LIABILITY_PATTERNS: list[tuple[str, str, str]] = [
    # (motif_name, regex_pattern, mechanism)
    ("Deamidation", r"N[GSA]", "Asn deamidation via succinimide intermediate"),
    ("Isomerization", r"DG", "Asp isomerization to iso-Asp via succinimide"),
    (
        "N-Glycosylation",
        r"N[^P][ST]",
        "N-linked glycosylation sequon (Asn-X-Ser/Thr, X≠Pro)",
    ),
]

# Pre-compile for performance
_COMPILED_LIABILITIES: list[tuple[str, re.Pattern[str], str]] = [
    (name, re.compile(pat), mech) for name, pat, mech in _LIABILITY_PATTERNS
]


@mcp.tool()
def scan_structural_liabilities(sequence: str) -> str:
    """Scan a protein sequence for post-translational modification hotspots.

    Identifies deterministic sequence motifs that are known to cause
    manufacturing failures in biologics:
      - Deamidation: NG, NS, NA
      - Isomerization: DG
      - N-glycosylation: N[^P][ST] (Asn-X-Ser/Thr where X ≠ Pro)

    Args:
        sequence: Single-letter amino-acid sequence.

    Returns:
        JSON string with fields:
            sequence_length, liabilities (list of hits), liability_count,
            overall_flag ("PASS" or "FAIL").
    """
    seq: str = _clean_sequence(sequence)

    if not seq:
        return json.dumps({"error": "Empty or invalid sequence provided."})

    liabilities: list[dict[str, str | int]] = []

    for name, pattern, mechanism in _COMPILED_LIABILITIES:
        for match in pattern.finditer(seq):
            liabilities.append(
                {
                    "liability_type": name,
                    "motif": match.group(),
                    "position": match.start() + 1,  # 1-based for biologists
                    "mechanism": mechanism,
                }
            )

    # Sort by position for readability
    liabilities.sort(key=lambda h: h["position"])

    report: dict = {
        "sequence_length": len(seq),
        "liabilities": liabilities,
        "liability_count": len(liabilities),
        "overall_flag": "FAIL" if liabilities else "PASS",
    }

    result: str = json.dumps(report, indent=2)
    logger.info("scan_structural_liabilities | %s", result)
    return result


# --- VHH Hallmark Tetrad Audit ------------------------------------------------
# Kabat/Chothia FR2 positions that distinguish camelid VHH from conventional VH.
# Canonical camelid residues: F37, E44, R45, G47
# Humanizing substitutions: V37, G44, L45, W47
#
# WARNING: Humanization of these positions can destabilize the VHH. The camelid
# hallmarks compensate for the absence of VL by providing a hydrophilic interface
# where conventional VH has a hydrophobic VH-VL contact surface.

_HALLMARK_POSITIONS: list[dict[str, str | int]] = [
    {
        "kabat_position": 37,
        "camelid_residue": "F",
        "human_vh_residue": "V",
        "role": "Core packing; compensates for missing VL contact",
    },
    {
        "kabat_position": 44,
        "camelid_residue": "E",
        "human_vh_residue": "G",
        "role": "Hydrophilic substitution at former VH-VL interface",
    },
    {
        "kabat_position": 45,
        "camelid_residue": "R",
        "human_vh_residue": "L",
        "role": "Charged residue replacing hydrophobic VL contact",
    },
    {
        "kabat_position": 47,
        "camelid_residue": "G",
        "human_vh_residue": "W",
        "role": "Flexible Gly replacing bulky Trp at VL interface",
    },
]

_HUMANIZATION_WARNING: str = (
    "Humanization of FR2 hallmarks in VHH can significantly reduce solubility "
    "and increase aggregation propensity. The camelid tetrad (F37/E44/R45/G47) "
    "evolved to compensate for the absence of VL. Reverting to human VH residues "
    "re-exposes the hydrophobic VH-VL interface without a binding partner, "
    "often leading to self-association. Proceed only with experimental validation."
)


@mcp.tool()
def vhh_hallmark_audit(sequence: str, framework2_start: int = 36) -> str:
    """Audit FR2 hallmark positions for camelid vs. human VH identity.

    Checks Kabat/Chothia positions 37, 44, 45, and 47 for the canonical
    camelid VHH tetrad (F, E, R, G). If camelid residues are present,
    suggests humanizing mutations but warns about solubility trade-offs.

    Args:
        sequence: Single-letter amino-acid VHH sequence.
        framework2_start: 0-based index where FR2 begins in the linear
            sequence. Default 36 assumes standard VHH numbering where
            Kabat position 36 maps to index 35 (0-based), making position
            37 = index 36. Adjust if your numbering differs.

    Returns:
        JSON string with per-position audit, humanization suggestions,
        and an overall assessment.
    """
    seq: str = _clean_sequence(sequence)

    if not seq:
        return json.dumps({"error": "Empty or invalid sequence provided."})

    # Map Kabat positions to 0-based sequence indices.
    # Kabat 37 → framework2_start + (37 - 37) = framework2_start + 0
    # Offsets relative to Kabat 37:
    kabat_to_offset: dict[int, int] = {37: 0, 44: 7, 45: 8, 47: 10}

    audits: list[dict[str, str | int | bool]] = []
    camelid_count: int = 0

    for hallmark in _HALLMARK_POSITIONS:
        kabat_pos: int = hallmark["kabat_position"]  # type: ignore[assignment]
        offset: int = kabat_to_offset[kabat_pos]
        seq_index: int = framework2_start + offset

        if seq_index >= len(seq):
            audits.append(
                {
                    "kabat_position": kabat_pos,
                    "status": "ERROR",
                    "detail": f"Sequence too short to contain Kabat position {kabat_pos} "
                    f"(need index {seq_index}, have {len(seq)} residues).",
                }
            )
            continue

        observed: str = seq[seq_index]
        is_camelid: bool = observed == hallmark["camelid_residue"]
        is_human: bool = observed == hallmark["human_vh_residue"]

        if is_camelid:
            camelid_count += 1

        audit_entry: dict[str, str | int | bool] = {
            "kabat_position": kabat_pos,
            "sequence_index": seq_index + 1,  # 1-based
            "observed_residue": observed,
            "expected_camelid": hallmark["camelid_residue"],
            "expected_human": hallmark["human_vh_residue"],
            "is_camelid_hallmark": is_camelid,
            "is_human_vh": is_human,
            "role": hallmark["role"],
        }

        if is_camelid:
            audit_entry["suggestion"] = (
                f"Humanize {observed}{kabat_pos}{hallmark['human_vh_residue']} "
                f"if regulatory/immunogenicity concerns require it."
            )

        audits.append(audit_entry)

    # Overall assessment
    if camelid_count == 4:
        identity = "Fully camelid VHH (all 4 hallmarks present)"
    elif camelid_count == 0:
        identity = "Fully humanized FR2 (no camelid hallmarks)"
    else:
        identity = f"Chimeric FR2 ({camelid_count}/4 camelid hallmarks)"

    report: dict = {
        "sequence_length": len(seq),
        "framework2_start_index": framework2_start + 1,  # 1-based
        "hallmark_audit": audits,
        "camelid_hallmark_count": camelid_count,
        "identity": identity,
    }

    if camelid_count > 0:
        report["humanization_warning"] = _HUMANIZATION_WARNING

    result: str = json.dumps(report, indent=2)
    logger.info("vhh_hallmark_audit | %s", result)
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
