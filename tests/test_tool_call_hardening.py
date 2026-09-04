"""_call_cmd payload extraction: stdout pollution tolerance and error visibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dft_forge.tools.definitions import _balanced_json_objects, _call_cmd, _parse_payload


PRETTY = {"ok": True, "command": "structure.import", "formula": "MoS2", "natoms": 3,
          "nested": {"sites": ["Mo", "S", "S"]}}


def _ns(output=None):
    return argparse.Namespace(output=output)


class TestBalancedExtraction:
    def test_clean_stdout(self):
        text = json.dumps(PRETTY, indent=2)
        assert _parse_payload(text) == PRETTY

    def test_debug_prints_around_payload(self):
        text = 'Loading pseudos...\n' + json.dumps(PRETTY, indent=2) + '\nDone.\n'
        assert _parse_payload(text) == PRETTY

    def test_debug_print_inside_envelope(self):
        # pollution before AND after the pretty-printed payload
        text = 'debug {"a": 1}\n' + json.dumps(PRETTY, indent=2) + '\ntail print(x)'
        assert _parse_payload(text) == PRETTY

    def test_multi_line_indent_survives(self):
        # indent=2 payloads span lines; per-line scans fail but balancing works
        text = json.dumps(PRETTY, indent=2)
        assert len(text.splitlines()) > 5
        assert _parse_payload("noise\n" + text) == PRETTY

    def test_braces_inside_strings_do_not_confuse_depth(self):
        text = json.dumps({"msg": "brace } inside { string"}, indent=2)
        assert _parse_payload(text) == {"msg": "brace } inside { string"}

    def test_unparseable_garbage_returns_none(self):
        assert _parse_payload("no json here at all") is None
        assert _parse_payload("") is None

    def test_balanced_objects_finds_both(self):
        text = '{"a": 1}\n{"b": 2}'
        assert len(_balanced_json_objects(text)) == 2


class TestCallCmdPaths:
    def test_polluted_stdout_still_yields_json(self):
        def fake_cmd(ns):
            print("debug print that would break naive parsing")
            print(json.dumps(PRETTY, indent=2))
            return 0

        payload = _call_cmd(fake_cmd, _ns())
        assert payload["json"] == PRETTY
        assert "json_parse_failed" not in payload

    def test_error_stderr_becomes_structured_json(self):
        # _error_response prints {"ok": false, "error": ...} to STDERR only
        err = {"ok": False, "error": "Import failed", "code": 1}

        def fake_cmd(ns):
            print(json.dumps(err, indent=2), file=__import__("sys").stderr)
            return 1

        payload = _call_cmd(fake_cmd, _ns())
        assert payload["json"]["ok"] is False
        assert payload["json"]["error"] == "Import failed"

    def test_output_file_is_authoritative(self, tmp_path: Path):
        # with --output set, _write_output writes JSON to the file, not stdout
        out_file = tmp_path / "result.json"
        out_file.write_text(json.dumps(PRETTY, indent=2))

        def fake_cmd(ns):
            print("some stray stdout")  # would have been unparsable before
            return 0

        payload = _call_cmd(fake_cmd, _ns(output=str(out_file)))
        assert payload["json"] == PRETTY

    def test_loud_flag_when_nothing_parseable(self):
        def fake_cmd(ns):
            print("just text, no structure")
            return 0

        payload = _call_cmd(fake_cmd, _ns())
        assert "json" not in payload
        assert payload["json_parse_failed"] is True

    def test_silent_when_no_output_at_all(self):
        def fake_cmd(ns):
            return 0

        payload = _call_cmd(fake_cmd, _ns())
        assert "json" not in payload
        assert "json_parse_failed" not in payload
