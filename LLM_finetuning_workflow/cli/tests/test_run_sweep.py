"""Unit tests for scripts/run_sweep.py — the laptop driver.

Covers the pure grid/tag logic and the config renderers (train, axolotl, and the
new eval spec), plus the subprocess-backed helpers with `subprocess.run` mocked so
no `air`/`databricks` CLI or network is touched. The renderers read the REAL
templates in configs/ but are pointed at a tmp GENERATED dir so they never write
into configs/generated/.
"""
import json
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

import run_sweep


@pytest.fixture
def grid():
    """A self-contained grid dict (does not read the repo's grid.yaml)."""
    return {
        "model": "Qwen/Qwen3-8B",
        "data_dir": "/Volumes/cat/schema/training_data/qwen_sweep",
        "checkpoints_dir": "/Volumes/cat/schema/checkpoints/qwen_sweep",
        "mlflow_experiment": "qwen-ft-sweep",
        "mlflow_experiment_path": "/Users/me@databricks.com/qwen-ft-sweep",
        "num_accelerators": 8,
        "accelerator_type": "GPU_8xH100",
        "learning_rates": [1.0e-6, 2.0e-6],
        "epochs": [3, 5],
    }


# --- pure grid/tag logic -----------------------------------------------------
@pytest.mark.parametrize("lr,expected", [
    (2.0e-6, "2e-6"),
    (1.0e-5, "1e-5"),
    (5.0e-6, "5e-6"),
    (1.0e-6, "1e-6"),
])
def test_lr_tag(lr, expected):
    assert run_sweep.lr_tag(lr) == expected


def test_grid_cells_cartesian_product(grid):
    cells = list(run_sweep.grid_cells(grid))
    assert len(cells) == len(grid["learning_rates"]) * len(grid["epochs"])
    tags = {tag for _, _, tag in cells}
    assert {"lr1e-6_ep3", "lr1e-6_ep5", "lr2e-6_ep3", "lr2e-6_ep5"} == tags


# --- eval spec renderer (the new code path) ----------------------------------
def test_render_eval_air_config_all_checkpoints(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    out = run_sweep.render_eval_air_config(grid, None)
    cfg = yaml.safe_load(out.read_text())

    assert out.name == "eval_all.yaml"
    assert cfg["environment"]["version"] == "databricks_ai_v5"   # the FIPS lever
    assert cfg["compute"]["num_accelerators"] == 1
    assert cfg["compute"]["accelerator_type"] == "GPU_1xH100"
    assert cfg["experiment_name"] == grid["mlflow_experiment"]
    assert cfg["code_source"]["snapshot"]["root_path"].endswith("/cli")

    cmd = cfg["command"]
    assert "scripts/eval_cli.py" in cmd
    assert f"--data-dir {grid['data_dir']}" in cmd
    assert f"--checkpoints-dir {grid['checkpoints_dir']}" in cmd
    assert f"--experiment {grid['mlflow_experiment_path']}" in cmd
    assert "--tag" not in cmd                                    # all-checkpoints mode


def test_render_eval_air_config_single_tag(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    out = run_sweep.render_eval_air_config(grid, "lr2e-6_ep5")
    cfg = yaml.safe_load(out.read_text())

    assert out.name == "eval_lr2e-6_ep5.yaml"
    assert "--tag lr2e-6_ep5" in cfg["command"]
    # eval never inherits axolotl's telemetry export
    assert "AXOLOTL_DO_NOT_TRACK" not in cfg["command"]
    # the $CODE resolver (AppleDouble ._ workaround) is present
    assert 'CODE="$CODE_SOURCE_PATH"' in cfg["command"]


# --- train spec renderer -----------------------------------------------------
def test_render_air_config_multi_gpu_uses_accelerate(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    out = run_sweep.render_air_config(grid, "lr2e-6_ep5", "axolotl_lr2e-6_ep5.yaml")
    cfg = yaml.safe_load(out.read_text())

    assert cfg["experiment_name"] == "qwen-ft-sweep"
    cmd = cfg["command"]
    assert "accelerate launch" in cmd
    assert "AXOLOTL_DO_NOT_TRACK=1" in cmd
    assert "axolotl_lr2e-6_ep5.yaml" in cmd


def test_render_air_config_single_gpu_bare_axolotl(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    grid1 = {**grid, "num_accelerators": 1, "accelerator_type": "GPU_1xH100"}
    out = run_sweep.render_air_config(grid1, "t", "ax.yaml")
    cmd = yaml.safe_load(out.read_text())["command"]
    assert "axolotl train" in cmd
    assert "accelerate launch" not in cmd


def test_render_axolotl_config_overrides(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    out = run_sweep.render_axolotl_config(grid, 2.0e-6, 5, "lr2e-6_ep5")
    cfg = yaml.safe_load(out.read_text())

    assert cfg["base_model"] == grid["model"]
    assert cfg["learning_rate"] == 2.0e-6
    assert cfg["num_epochs"] == 5
    # checkpoint goes to <checkpoints_dir>/<tag>/ — NOT under data_dir/out
    assert cfg["output_dir"] == grid["checkpoints_dir"] + "/lr2e-6_ep5"
    assert "/out/" not in cfg["output_dir"]
    assert cfg["datasets"][0]["path"].endswith("/train.jsonl")
    assert cfg["test_datasets"][0]["path"].endswith("/val.jsonl")
    # must NOT set an experiment on the axolotl side (would fork a 2nd experiment)
    assert "mlflow_experiment_name" not in cfg


# --- subprocess-backed helpers (mocked; no CLI / network) --------------------
def test_completed_tags_keys_off_config_json(monkeypatch, grid):
    out_dir = grid["checkpoints_dir"]     # checkpoints live under checkpoints_dir now

    def fake_run(cmd, capture_output, text):
        path = cmd[3]  # dbfs:<...>
        if path == f"dbfs:{out_dir}":
            return SimpleNamespace(returncode=0, stdout="lr2e-6_ep5\nlr1e-6_ep3\n")
        if path.endswith("/lr2e-6_ep5"):
            return SimpleNamespace(returncode=0, stdout="config.json\nmodel.safetensors\n")
        return SimpleNamespace(returncode=0, stdout="model.safetensors\n")  # no config.json

    monkeypatch.setattr(run_sweep.subprocess, "run", fake_run)
    done = run_sweep.completed_tags(grid, "prof")
    assert done == {"lr2e-6_ep5"}                    # only the one with config.json


def test_completed_tags_empty_on_ls_error(monkeypatch, grid):
    monkeypatch.setattr(run_sweep.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    assert run_sweep.completed_tags(grid, "prof") == set()


def test_active_run_count_parses_json(monkeypatch):
    monkeypatch.setattr(run_sweep.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(
                            returncode=0, stdout=json.dumps({"data": {"runs": [1, 2, 3]}})))
    assert run_sweep.active_run_count("prof") == 3


def test_active_run_count_negative_on_error(monkeypatch):
    monkeypatch.setattr(run_sweep.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    assert run_sweep.active_run_count("prof") == -1


# --- submit command construction ---------------------------------------------
def test_submit_includes_idem_key_and_profile(monkeypatch, tmp_path, grid):
    captured = {}

    def fake_run(cmd):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_sweep.subprocess, "run", fake_run)
    f = tmp_path / "eval.yaml"
    f.write_text("x")
    rc = run_sweep.submit(f, grid, "prof", False, "tag", False, "qwen-sweep-eval-tag")

    assert rc == 0
    cmd = captured["cmd"]
    assert "--idempotency-key" in cmd and "qwen-sweep-eval-tag" in cmd
    assert cmd[cmd.index("--profile") + 1] == "prof"
    assert "--dry-run" not in cmd and "--watch" not in cmd


def test_submit_no_idem_key_and_dry_run(monkeypatch, tmp_path, grid):
    captured = {}

    def fake_run(cmd):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_sweep.subprocess, "run", fake_run)
    f = tmp_path / "eval.yaml"
    f.write_text("x")
    run_sweep.submit(f, grid, "", True, "tag", False, None)

    cmd = captured["cmd"]
    assert "--idempotency-key" not in cmd    # eval default: no dedup key
    assert "--dry-run" in cmd
    assert "--profile" not in cmd            # empty profile omitted


# --- run_eval orchestration --------------------------------------------------
def _eval_args(**over):
    base = dict(only="", print_only=False, dry_run=True, serialize_start=False,
                max_active=1, profile="p", watch=False, idem_suffix="")
    base.update(over)
    return SimpleNamespace(**base)


def test_run_eval_requires_experiment_path(grid):
    g = {k: v for k, v in grid.items() if k != "mlflow_experiment_path"}
    assert run_sweep.run_eval(g, _eval_args()) == 1


def _capture_submit(captured):
    def fake_submit(*a):
        captured["args"] = a
        return 0
    return fake_submit


def test_run_eval_all_submits_without_idem_key(monkeypatch, tmp_path, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    captured = {}
    monkeypatch.setattr(run_sweep, "submit", _capture_submit(captured))
    rc = run_sweep.run_eval(grid, _eval_args(only=""))
    assert rc == 0
    assert captured["args"][-1] is None                    # idem_key: None by default


def test_run_eval_idem_suffix_forces_key(monkeypatch, tmp_path, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    captured = {}
    monkeypatch.setattr(run_sweep, "submit", _capture_submit(captured))
    run_sweep.run_eval(grid, _eval_args(only="lr2e-6_ep5", idem_suffix="v2"))
    assert captured["args"][-1] == "qwen-sweep-eval-lr2e-6_ep5-v2"


def test_run_eval_print_only_does_not_submit(monkeypatch, tmp_path, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    called = {"n": 0}
    monkeypatch.setattr(run_sweep, "submit", lambda *a: called.__setitem__("n", called["n"] + 1))
    rc = run_sweep.run_eval(grid, _eval_args(print_only=True))
    assert rc == 0 and called["n"] == 0


def test_run_eval_guards_missing_checkpoint(monkeypatch, tmp_path, grid):
    # --eval --only <tag> for a checkpoint not on the Volume should fail locally,
    # not spin up a GPU worker (mirrors run_register's guard).
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    monkeypatch.setattr(run_sweep, "completed_tags", lambda g, p: {"lr1e-5_ep5"})
    submitted = {"n": 0}
    monkeypatch.setattr(run_sweep, "submit",
                        lambda *a: submitted.__setitem__("n", submitted["n"] + 1) or 0)
    rc = run_sweep.run_eval(grid, _eval_args(only="lr1e-5_ep4", dry_run=False))
    assert rc == 1 and submitted["n"] == 0


# --- register spec renderer + run_register orchestration ---------------------
def test_render_register_air_config(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    g = {**grid, "registered_model": "cat.sch.mymodel"}
    out = run_sweep.render_register_air_config(g, "lr1e-5_ep4")
    cfg = yaml.safe_load(out.read_text())

    assert out.name == "register_lr1e-5_ep4.yaml"
    assert cfg["environment"]["version"] == "databricks_ai_v5"
    assert cfg["compute"]["accelerator_type"] == "GPU_1xH100"
    cmd = cfg["command"]
    assert "scripts/register_model.py" in cmd
    assert f"--checkpoints-dir {g['checkpoints_dir']}" in cmd
    assert "--tag lr1e-5_ep4" in cmd
    assert "--uc-model-name cat.sch.mymodel" in cmd


def test_main_only_no_match_exits(monkeypatch, grid):
    # A --only tag that isn't a grid cell must fail loudly, not be a silent no-op.
    monkeypatch.setattr(run_sweep, "load_grid", lambda: grid)
    monkeypatch.setattr(sys, "argv", ["run_sweep.py", "--only", "nope_ep9", "--print-only"])
    with pytest.raises(SystemExit) as ei:
        run_sweep.main()
    assert ei.value.code == 2


def test_run_register_requires_registered_model(grid):
    # grid fixture has no registered_model
    assert run_sweep.run_register(grid, _eval_args(only="lr1e-5_ep4")) == 1


def test_run_register_requires_explicit_tag(grid):
    g = {**grid, "registered_model": "c.s.m"}
    assert run_sweep.run_register(g, _eval_args(only="")) == 1


def test_run_register_submits_with_per_tag_idem_key(monkeypatch, tmp_path, grid):
    monkeypatch.setattr(run_sweep, "GENERATED", tmp_path)
    g = {**grid, "registered_model": "c.s.m"}
    captured = {}
    monkeypatch.setattr(run_sweep, "submit", _capture_submit(captured))
    # dry_run=True (from _eval_args) skips the Volume checkpoint guard
    rc = run_sweep.run_register(g, _eval_args(only="lr1e-5_ep4"))
    assert rc == 0
    assert captured["args"][-1] == "qwen-sweep-register-lr1e-5_ep4"


# --- pick_best experiment lookup (regression: slug vs absolute path) ----------
def test_pick_best_looks_up_by_absolute_path(monkeypatch, grid):
    """Regression: on Databricks the experiment is named by its absolute workspace
    path, so pick_best must look it up by mlflow_experiment_path — NOT the slug
    `mlflow_experiment` (get_experiment_by_name(slug) returns None and the winner
    is never found)."""
    pd = pytest.importorskip("pandas")
    rec = {}

    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda uri: None

    def _get(name):
        rec["looked_up"] = name
        return SimpleNamespace(experiment_id="123")

    def _search(experiment_ids, filter_string, order_by):
        rec["filter"] = filter_string
        return pd.DataFrame({
            "params.checkpoint_tag": ["lr1e-5_ep4"],
            "metrics.all_f1": [0.93],
            "metrics.top8_f1": [0.94],
        })

    fake.get_experiment_by_name = _get
    fake.search_runs = _search
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    rc = run_sweep.pick_best(grid, "prof")
    assert rc == 0
    assert rec["looked_up"] == grid["mlflow_experiment_path"]   # absolute path, not the slug
    assert "stage" in rec["filter"]
