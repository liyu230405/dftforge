import json

import pytest

from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.graph import GraphTemplate, NodeSpec, load_named_template


def _template(template_id):
    return GraphTemplate(
        template_id=template_id,
        nodes=[NodeSpec(id="n", tool="fake")],
    ).to_dict()


def _write_template(directory, filename, template_id):
    path = directory / filename
    path.write_text(json.dumps(_template(template_id)))
    return path


def _engine(tmp_path, templates_dir):
    return GraphEngine(
        tmp_path / "runtime",
        store_path=tmp_path / "runtime" / "state.db",
        tools={"fake": object()},
        templates_dir=templates_dir,
    )


def test_named_resolver_accepts_json_yaml_and_yml_ids(tmp_path):
    _write_template(tmp_path, "t1.json", "t1")
    _write_template(tmp_path, "t5-custom.yaml", "t5-custom")
    _write_template(tmp_path, "t6.custom.yml", "t6.custom")

    assert load_named_template(tmp_path, "t1").template_id == "t1"
    assert load_named_template(tmp_path, "t5-custom").template_id == "t5-custom"
    assert load_named_template(tmp_path, "t6.custom").template_id == "t6.custom"


@pytest.mark.parametrize("template_id", [".", "..", "../secret", "nested/template", "/tmp/template", ""])
def test_named_resolver_rejects_path_like_ids(tmp_path, template_id):
    with pytest.raises(ValueError, match="unsafe template id"):
        load_named_template(tmp_path, template_id)


def test_named_resolver_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-template.json"
    outside.write_text(json.dumps(_template("escaped")))
    link = tmp_path / "escaped.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="escapes"):
        load_named_template(tmp_path, "escaped")


def test_named_resolver_rejects_embedded_id_mismatch(tmp_path):
    _write_template(tmp_path, "requested.json", "declared")

    with pytest.raises(ValueError, match="id mismatch"):
        load_named_template(tmp_path, "requested")


def test_engine_uses_shared_resolver_for_get_and_list(tmp_path):
    _write_template(tmp_path, "good.json", "good")
    _write_template(tmp_path, "fifth.yaml", "fifth")
    _write_template(tmp_path, "wrong.json", "different")
    outside = tmp_path.parent / "engine-outside.json"
    outside.write_text(json.dumps(_template("outside")))
    try:
        (tmp_path / "outside.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    engine = _engine(tmp_path, tmp_path)
    assert engine.get_template("good").template_id == "good"
    assert engine.get_template("fifth").template_id == "fifth"
    with pytest.raises(ValueError, match="id mismatch"):
        engine.get_template("wrong")
    with pytest.raises(ValueError, match="escapes"):
        engine.get_template("outside")
    with pytest.raises(ValueError, match="unsafe template id"):
        engine.get_template("../outside")

    assert set(engine.list_templates()) >= {"good", "fifth"}
    assert "wrong" not in engine.list_templates()
    assert "outside" not in engine.list_templates()


def test_registered_dynamic_template_keeps_precedence_over_disk_safety(tmp_path):
    engine = _engine(tmp_path, tmp_path)
    dynamic = GraphTemplate(
        template_id="dynamic/template",
        nodes=[NodeSpec(id="n", tool="fake")],
    )
    engine.register_template(dynamic)

    assert engine.get_template("dynamic/template") is dynamic
    assert "dynamic/template" in engine.list_templates()
