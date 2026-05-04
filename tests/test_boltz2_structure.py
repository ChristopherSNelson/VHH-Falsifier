"""
Tests for the Boltz-2 structure prediction tool.

All tests use dry_run=True - no GPU or model weights required.
These validate input handling, YAML generation, and output schema.
GPU inference tests are integration tests run manually on a GPU instance.
"""

import json
from pathlib import Path

from biologics_server import predict_vhh_complex_structure
from tools.boltz2_structure import (
    PD1_ECTODOMAIN,
    _clean,
    _validate_sequence,
    _write_boltz_yaml,
    predict_structure,
)


def parse(result_str: str) -> dict:
    return json.loads(result_str)


VHH_SEQ = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWFRQAPGKGLEWVSSISSSSSYIYVDSVKG"
    "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAAADYGMDVWGQGTLVTVSS"
)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


class TestValidateSequence:
    def test_valid_sequence_returns_none(self):
        assert _validate_sequence("EVQLVESGGM", "test") is None

    def test_empty_sequence_returns_error(self):
        err = _validate_sequence("", "VHH")
        assert err is not None
        assert "empty" in err.lower()

    def test_whitespace_only_returns_error(self):
        err = _validate_sequence("   \n\t", "VHH")
        assert err is not None

    def test_nonstandard_residue_returns_error(self):
        err = _validate_sequence("EVQLVEXB", "VHH")
        assert err is not None
        assert "non-standard" in err.lower()

    def test_too_short_returns_error(self):
        err = _validate_sequence("EVQ", "VHH")
        assert err is not None
        assert "too short" in err.lower()

    def test_10_residues_is_valid(self):
        assert _validate_sequence("EVQLVESGGM", "VHH") is None


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


class TestWriteBoltzYaml:
    def test_single_chain_yaml(self, tmp_path):
        out = tmp_path / "test.yaml"
        _write_boltz_yaml("EVQLVES", None, out)
        content = out.read_text()
        assert "version: 1" in content
        assert "EVQLVES" in content
        assert "id: A" in content
        assert "id: B" not in content

    def test_two_chain_yaml(self, tmp_path):
        out = tmp_path / "test.yaml"
        _write_boltz_yaml("EVQLVES", "MDQLTEEQIA", out)
        content = out.read_text()
        assert "id: A" in content
        assert "id: B" in content
        assert "EVQLVES" in content
        assert "MDQLTEEQIA" in content

    def test_yaml_is_valid_yaml(self, tmp_path):
        import yaml

        out = tmp_path / "test.yaml"
        _write_boltz_yaml("EVQLVES", "MDQLTEEQIA", out)
        data = yaml.safe_load(out.read_text())
        assert data["version"] == 1
        assert len(data["sequences"]) == 2


# ---------------------------------------------------------------------------
# predict_structure dry_run
# ---------------------------------------------------------------------------


class TestPredictStructureDryRun:
    def test_dry_run_returns_dry_run_status(self, tmp_path):
        result = predict_structure(VHH_SEQ, dry_run=True, out_dir=str(tmp_path))
        assert result["status"] == "dry_run"

    def test_dry_run_writes_input_yaml(self, tmp_path):
        result = predict_structure(VHH_SEQ, dry_run=True, out_dir=str(tmp_path))
        assert Path(result["input_yaml"]).exists()

    def test_dry_run_reports_vhh_length(self, tmp_path):
        result = predict_structure(VHH_SEQ, dry_run=True, out_dir=str(tmp_path))
        assert result["vhh_length"] == len(_clean(VHH_SEQ))

    def test_dry_run_with_antigen_reports_antigen_length(self, tmp_path):
        antigen = "MDQLTEEQIAEFKEAFSLF"
        result = predict_structure(
            VHH_SEQ, antigen_sequence=antigen, dry_run=True, out_dir=str(tmp_path)
        )
        assert result["antigen_length"] == len(antigen)

    def test_dry_run_without_antigen_uses_pd1_default(self, tmp_path):
        # When no antigen is provided via the MCP tool, PD-1 ectodomain is used
        from tools.boltz2_structure import PD1_ECTODOMAIN

        result = predict_structure(
            VHH_SEQ, antigen_sequence=PD1_ECTODOMAIN, dry_run=True, out_dir=str(tmp_path)
        )
        assert result["antigen_length"] == len(_clean(PD1_ECTODOMAIN))

    def test_dry_run_has_message_about_gpu(self, tmp_path):
        result = predict_structure(VHH_SEQ, dry_run=True, out_dir=str(tmp_path))
        assert "GPU" in result["message"] or "gpu" in result["message"].lower()

    def test_dry_run_confidence_is_none(self, tmp_path):
        result = predict_structure(VHH_SEQ, dry_run=True, out_dir=str(tmp_path))
        assert result["confidence"] is None

    def test_invalid_vhh_returns_error(self, tmp_path):
        result = predict_structure("EVQLVEX", dry_run=True, out_dir=str(tmp_path))
        assert result["status"] == "error"
        assert "error" in result

    def test_empty_vhh_returns_error(self, tmp_path):
        result = predict_structure("", dry_run=True, out_dir=str(tmp_path))
        assert result["status"] == "error"

    def test_return_structure_has_required_keys(self, tmp_path):
        result = predict_structure(VHH_SEQ, dry_run=True, out_dir=str(tmp_path))
        for key in (
            "status",
            "vhh_length",
            "antigen_length",
            "input_yaml",
            "output_dir",
            "confidence",
            "structure_path",
            "error",
        ):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# MCP tool wrapper (predict_vhh_complex_structure)
# ---------------------------------------------------------------------------


class TestPredictVhhComplexStructureMcpTool:
    def test_dry_run_returns_json_string(self, tmp_path):
        result_str = predict_vhh_complex_structure(
            vhh_sequence=VHH_SEQ, dry_run=True, out_dir=str(tmp_path)
        )
        assert isinstance(result_str, str)
        result = parse(result_str)
        assert result["status"] == "dry_run"

    def test_defaults_to_pd1_antigen(self, tmp_path):
        result = parse(
            predict_vhh_complex_structure(
                vhh_sequence=VHH_SEQ, dry_run=True, out_dir=str(tmp_path)
            )
        )
        # Antigen length should match PD-1 ectodomain
        assert result["antigen_length"] == len(_clean(PD1_ECTODOMAIN))

    def test_invalid_sequence_returns_error_json(self, tmp_path):
        result = parse(
            predict_vhh_complex_structure(
                vhh_sequence="NOTASEQUENCEXXX", dry_run=True, out_dir=str(tmp_path)
            )
        )
        assert result["status"] == "error"
