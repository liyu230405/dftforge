from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._helpers import _error_response, _success_response


def cmd_result_parse(args: argparse.Namespace) -> int:
    from dft_forge.parser import QEParser, ParsedVCResult, ParsedBands, ParsedDOS

    output_file = Path(args.input)
    if not output_file.exists():
        return _error_response(f"Output file not found: {output_file}")

    result_type = args.type
    xml_path = Path(args.xml_path) if args.xml_path else None
    if xml_path and not xml_path.exists():
        xml_path = None

    text = output_file.read_text(errors="replace")
    parsed: Any = None
    if result_type == "vc-relax":
        parsed = QEParser.parse_vc_relax(text, xml_path=xml_path)
    elif result_type == "scf":
        parsed = QEParser.parse_scf(text, xml_path=xml_path)
    elif result_type == "nscf":
        parsed = QEParser.parse_nscf(text, xml_path=xml_path)
    elif result_type == "bands":
        parsed = QEParser.parse_bands(text, xml_path=xml_path)
    elif result_type == "dos":
        parsed = QEParser.parse_dos(text, xml_path=xml_path)
    else:
        return _error_response(f"Unsupported result type: {result_type}")

    if hasattr(parsed, "to_dict"):
        data = parsed.to_dict()
    elif hasattr(parsed, "__dict__"):
        data = {k: v for k, v in parsed.__dict__.items() if not k.startswith("_")}
    else:
        data = {"value": str(parsed)}

    data.setdefault("command", "result.parse")
    data.setdefault("input_file", str(output_file))
    data.setdefault("result_type", result_type)
    return _success_response(data, args.output)



def cmd_result_verify(args: argparse.Namespace) -> int:
    from dft_forge.verifier import ScientificVerifier, ConvergenceReport
    from dft_forge.protocol.schemas import VerificationInput

    result_path = Path(args.input)
    if not result_path.exists():
        return _error_response(f"Result file not found: {result_path}")

    result = json.loads(result_path.read_text())
    task_type = getattr(args, "task_type", None) or result.get("task_type", "T1")

    verifier = ScientificVerifier()
    inp = VerificationInput(
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        job_success=result.get("status") == "pass" or bool(result.get("success", True)),
        metadata=result,
    )

    # Reconstruct parsed data if present
    parsed = result.get("parsed") or result.get("physical_results") or {}
    if hasattr(parsed, "to_dict"):
        parsed_obj = parsed
    else:
        from types import SimpleNamespace
        parsed_obj = SimpleNamespace(**{k: v for k, v in parsed.items() if isinstance(v, (int, float, str, list, dict))})

    if task_type == "T2":
        xml_candidates = [
            str(f) for f in (result.get("output_files") or []) if str(f).endswith(".xml")
        ]
        bands_xml = Path(xml_candidates[0]) if xml_candidates else Path(result.get("bands_xml", "missing_bands.xml"))
        report = verifier.verify_t2_bands(inp.stdout, inp.stdout, bands_xml)
    else:
        report = verifier.verify_t1(inp, parsed_obj)

    # ConvergenceReport.checks is dict[str, dict] — serialize fields, not attributes
    if isinstance(report, ConvergenceReport):
        data = {
            "command": "result.verify",
            "task_type": task_type,
            "passed": report.passed,
            "checks": report.checks,
            "warnings": report.warnings,
            "failure_reasons": report.failure_reasons,
            "summary": report.summary,
        }
    else:
        data = {"command": "result.verify", "task_type": task_type, "passed": bool(report)}

    if not data["passed"]:
        data.setdefault("error", "Verification failed")
        return _error_response("Verification failed", code=4, details=data)
    return _success_response(data, args.output)
