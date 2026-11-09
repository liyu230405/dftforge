from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_sqlite_store_and_run_views_concurrent_process(tmp_path: Path):
    """Exercise independent stores against one DB without risking a hung pytest."""
    root = tmp_path / "session"
    script = r'''
import concurrent.futures
import sys
from pathlib import Path

from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import create_run
from dft_forge.runtime.store import SQLiteStateStore
from dft_forge.web.run_views import inspect_run, list_runs

root = Path(sys.argv[1])
db = root / "graphs" / "graph_state.db"
db.parent.mkdir(parents=True)
template = GraphTemplate(
    template_id="concurrency",
    inputs={"worker": 0},
    nodes=[NodeSpec(id="a", tool="fake")],
)
stores = [SQLiteStateStore(db) for _ in range(12)]

def worker(index):
    store = stores[index]
    run = create_run(template, {"worker": index})
    store.save_run(run, template)
    store.save_node_run(run, run.nodes["a"])
    for round_no in range(30):
        run.inputs["worker"] = index
        store.save_run(run)
        assert store.load_run(run.run_id, template) is not None
        assert store.list_runs()
        if round_no % 3 == 0:
            visible = list_runs(root)
            assert any(item["run_id"] == run.run_id for item in visible)
            detail = inspect_run(root, run.run_id)
            assert detail is not None
            assert detail["nodes"]["a"]["state"] == "pending"
    return run.run_id

with concurrent.futures.ThreadPoolExecutor(max_workers=len(stores)) as pool:
    run_ids = list(pool.map(worker, range(len(stores))))

runs = list_runs(root)
assert {item["run_id"] for item in runs} == set(run_ids)
for run_id in run_ids:
    detail = inspect_run(root, run_id)
    assert detail is not None
    assert detail["run_id"] == run_id
    assert "a" in detail["nodes"]
print("SQLITE_CONCURRENCY_OK", len(runs))
'''
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(repo_root), env.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SQLITE_CONCURRENCY_OK 12" in completed.stdout
