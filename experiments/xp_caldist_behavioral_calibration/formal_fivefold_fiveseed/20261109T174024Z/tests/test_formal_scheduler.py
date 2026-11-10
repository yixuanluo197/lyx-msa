import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "formal_scheduler.py"


def load_module():
    spec = importlib.util.spec_from_file_location("formal_scheduler_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def task(mod, task_id, *, deps=(), command=None, reused=False):
    return mod.Task(
        task_id=task_id,
        kind=task_id.split(":", 1)[0],
        dataset="IEMOCAP",
        fold=1,
        seed=None,
        dependencies=tuple(deps),
        command=tuple(command or (sys.executable, "-c", "pass")),
        reused=reused,
    )


def test_protocol_plan_has_all_formal_jobs_and_pilot_reuse_without_evaluation():
    mod = load_module()
    plan = mod.build_protocol_b_plan(Path("/formal"), Path("/teacher-pilot"), Path("/cmerc-pilot"))
    counts = {}
    for item in plan.tasks.values():
        counts[item.kind] = counts.get(item.kind, 0) + 1
        assert "evaluation" not in item.task_id.lower()
        assert all("evaluation" not in token.lower() for token in item.command)
    assert counts == {"teacher": 10, "difficulty": 10, "cmerc": 50, "calibration": 50}
    assert sum(t.reused for t in plan.tasks.values() if t.kind == "teacher") == 2
    assert sum(t.reused for t in plan.tasks.values() if t.kind == "difficulty") == 2
    assert sum(t.reused for t in plan.tasks.values() if t.kind == "cmerc") == 2
    assert sum(t.reused for t in plan.tasks.values() if t.kind == "calibration") == 2


def test_dependencies_are_teacher_then_difficulty_then_cmerc_then_calibration():
    mod = load_module()
    plan = mod.build_protocol_b_plan(Path("/formal"), Path("/teacher-pilot"), Path("/cmerc-pilot"))
    difficulty = plan.tasks["difficulty:IEMOCAP:fold2"]
    cmerc = plan.tasks["cmerc:IEMOCAP:fold2:seed42"]
    calibration = plan.tasks["calibration:IEMOCAP:fold2:seed42"]
    assert difficulty.dependencies == ("teacher:IEMOCAP:fold2",)
    assert cmerc.dependencies == (difficulty.task_id,)
    assert calibration.dependencies == (cmerc.task_id,)
    assert any(value.endswith("/runs/shared/IEMOCAP/fold2/artifacts/teacher_training_role_artifact.pt")
               for value in difficulty.command)
    assert any(value.endswith("/runs/shared/IEMOCAP/fold2/artifacts/training_role_artifact.pt")
               for value in cmerc.command)
    assert any(value.endswith("/runs/shared/IEMOCAP/fold2/artifacts/development_role_artifact.pt")
               for value in cmerc.command)
    assert calibration.command[-2:] == ("--seed", "20261109")


def test_plan_rejects_resume_and_any_evaluation_surface():
    mod = load_module()
    with pytest.raises(ValueError, match="resume"):
        mod.Plan([task(mod, "cmerc:x", command=("python", "runner.py", "--resume", "latest.pt"))])
    with pytest.raises(ValueError, match="Evaluation"):
        mod.Plan([task(mod, "cmerc:x", command=("python", "evaluation_runner.py"))])


def test_gpu_inventory_requires_unique_uuid_and_at_most_four():
    mod = load_module()
    with pytest.raises(ValueError, match="unique"):
        mod.validate_gpu_uuids(["GPU-a", "GPU-a"])
    with pytest.raises(ValueError, match="four"):
        mod.validate_gpu_uuids([f"GPU-{i}" for i in range(5)])
    assert mod.validate_gpu_uuids(["GPU-a", "GPU-b"]) == ("GPU-a", "GPU-b")


def test_disk_gate_requires_fifty_gib(monkeypatch, tmp_path):
    mod = load_module()
    usage = type("Usage", (), {"free": 50 * 1024**3 - 1})()
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _: usage)
    with pytest.raises(RuntimeError, match="50 GiB"):
        mod.require_disk_space(tmp_path)
    usage.free = 50 * 1024**3
    assert mod.require_disk_space(tmp_path) == usage.free


def test_failure_retry_uses_new_attempt_and_never_adds_resume(tmp_path):
    mod = load_module()
    marker = tmp_path / "count.txt"
    code = (
        "from pathlib import Path; import json,sys; "
        f"p=Path({str(marker)!r}); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
        "o=Path(sys.argv[1])/'output'; "
        "(o.mkdir(parents=True), (o/'teacher_final.pt').write_bytes(b'x'), "
        "(o/'access_audit.json').write_text(json.dumps({'exact_final_coverage':True,'rejected_ids':[]})), "
        "(o/'teacher_checkpoint_metadata.json').write_text('{}')) if n > 1 else None; "
        "sys.exit(7 if n == 1 else 0)"
    )
    plan = mod.Plan([task(mod, "teacher:IEMOCAP:fold2", command=(sys.executable, "-c", code, "{attempt_dir}"))])
    scheduler = mod.Scheduler(
        plan, tmp_path / "state", ("GPU-test-uuid",), max_attempts=2,
        artifact_validator=lambda task, output: {"status": "passed"},
    )
    scheduler.run(check_disk=False)
    ledger = json.loads((tmp_path / "state" / "task_ledger.json").read_text())
    row = ledger["tasks"]["teacher:IEMOCAP:fold2"]
    assert row["status"] == "completed"
    assert [a["attempt"] for a in row["attempts"]] == [1, 2]
    assert row["attempts"][0]["directory"] != row["attempts"][1]["directory"]
    assert all("--resume" not in a["command"] for a in row["attempts"])
    assert all(a["gpu_uuid"] == "GPU-test-uuid" for a in row["attempts"])


def test_independent_ready_tasks_are_dispatched_to_distinct_uuid_workers(tmp_path):
    mod = load_module()
    plan = mod.Plan([
        task(mod, "teacher:IEMOCAP:fold2"),
        task(mod, "teacher:MELD:fold2"),
    ])
    seen = []

    def runner(command, *, env, stdout, stderr):
        seen.append(env["CUDA_VISIBLE_DEVICES"])
        output = Path(stdout.name).parent / "output"
        output.mkdir()
        (output / "teacher_final.pt").write_bytes(b"x")
        (output / "access_audit.json").write_text(json.dumps({"exact_final_coverage": True, "rejected_ids": []}))
        (output / "teacher_checkpoint_metadata.json").write_text("{}")
        return type("Result", (), {"returncode": 0})()

    scheduler = mod.Scheduler(
        plan, tmp_path / "state", ("GPU-a", "GPU-b"), max_attempts=1,
        process_runner=runner,
        artifact_validator=lambda task, output: {"status": "passed"},
    )
    scheduler.run(check_disk=False)
    assert set(seen) == {"GPU-a", "GPU-b"}


def test_reused_tasks_are_registered_without_process_execution(tmp_path):
    mod = load_module()
    called = []
    reused_output = tmp_path / "frozen-teacher"
    reused_output.mkdir()
    (reused_output / "teacher_final.pt").write_bytes(b"x")
    (reused_output / "access_audit.json").write_text(json.dumps({"exact_final_coverage": True, "rejected_ids": []}))
    (reused_output / "teacher_checkpoint_metadata.json").write_text("{}")
    reused = task(mod, "teacher:IEMOCAP:fold1", reused=True)
    reused = mod.Task(**{**reused.__dict__, "reused_output": str(reused_output)})
    plan = mod.Plan([reused])
    scheduler = mod.Scheduler(
        plan, tmp_path / "state", ("GPU-a",),
        process_runner=lambda *a, **k: called.append(True),
    )
    scheduler.run(check_disk=False)
    ledger = json.loads((tmp_path / "state" / "task_ledger.json").read_text())
    assert ledger["tasks"]["teacher:IEMOCAP:fold1"]["status"] == "reused"
    assert called == []


def test_generated_difficulty_dependency_resolves_to_json_not_directory(tmp_path):
    mod = load_module()
    teacher = task(mod, "teacher:IEMOCAP:fold2")
    difficulty = task(mod, "difficulty:IEMOCAP:fold2", deps=(teacher.task_id,))
    cmerc = task(
        mod, "cmerc:IEMOCAP:fold2:seed42", deps=(difficulty.task_id,),
        command=("python", "runner.py", "--difficulty", "{dep:difficulty:IEMOCAP:fold2}"),
    )
    scheduler = mod.Scheduler(mod.Plan([teacher, difficulty, cmerc]), tmp_path / "state", ("GPU-a",))
    attempt = tmp_path / "state" / "attempts" / "difficulty__IEMOCAP__fold2" / "attempt-0001"
    (attempt / "output").mkdir(parents=True)
    (attempt / "output" / "difficulty.json").write_text("{}")
    scheduler.ledger["tasks"][difficulty.task_id].update(
        status="completed",
        attempts=[{"returncode": 0, "directory": str(attempt)}],
    )
    expanded = scheduler._expand(cmerc, tmp_path / "next")
    assert expanded[-1] == str(attempt / "output" / "difficulty.json")


def test_dependency_placeholder_can_be_followed_by_relative_file(tmp_path):
    mod = load_module()
    cmerc = task(mod, "cmerc:IEMOCAP:fold1:seed42")
    calibration = task(
        mod, "calibration:IEMOCAP:fold1:seed42", deps=(cmerc.task_id,),
        command=("python", "calibrate.py", "--checkpoint", "{dep:cmerc:IEMOCAP:fold1:seed42}/best.pt"),
    )
    scheduler = mod.Scheduler(mod.Plan([cmerc, calibration]), tmp_path / "state", ("GPU-a",))
    attempt = tmp_path / "state" / "attempts" / "cmerc__IEMOCAP__fold1__seed42" / "attempt-0001"
    (attempt / "output").mkdir(parents=True)
    scheduler.ledger["tasks"][cmerc.task_id].update(
        status="completed", attempts=[{"returncode": 0, "directory": str(attempt)}],
    )
    expanded = scheduler._expand(calibration, tmp_path / "next")
    assert expanded[-1] == str(attempt / "output" / "best.pt")


def test_terminal_failure_is_reported_after_fresh_attempt_budget(tmp_path):
    mod = load_module()
    failing = task(mod, "calibration:IEMOCAP:fold2", command=(sys.executable, "-c", "raise SystemExit(9)"))
    scheduler = mod.Scheduler(mod.Plan([failing]), tmp_path / "state", ("GPU-a",), max_attempts=2)
    with pytest.raises(RuntimeError, match="failed tasks"):
        scheduler.run(check_disk=False)
    ledger = json.loads((tmp_path / "state" / "task_ledger.json").read_text())
    assert len(ledger["tasks"][failing.task_id]["attempts"]) == 2

def test_zero_exit_without_required_artifacts_is_not_completed(tmp_path):
    mod = load_module()
    plan = mod.Plan([task(mod, 'teacher:IEMOCAP:fold2')])
    scheduler = mod.Scheduler(
        plan, tmp_path / 'state', ('GPU-a',), max_attempts=1,
        process_runner=lambda *a, **k: type('Result', (), {'returncode': 0})(),
    )
    with pytest.raises(RuntimeError, match='failed tasks after fresh attempts'):
        scheduler.run(check_disk=False)
    row = scheduler.ledger['tasks']['teacher:IEMOCAP:fold2']
    assert row['status'] == 'failed'
    assert row['attempts'][0]['process_returncode'] == 0
    assert row['attempts'][0]['artifact_validation']['status'] == 'failed'


def test_reused_output_must_exist_and_match_kind(tmp_path):
    mod = load_module()
    reused = mod.Task('teacher:IEMOCAP:fold1', 'teacher', 'IEMOCAP', 1, None, (), (), True,
                      str(tmp_path / 'missing'))
    scheduler = mod.Scheduler(mod.Plan([reused]), tmp_path / 'state', ('GPU-a',))
    with pytest.raises(RuntimeError, match='existing artifact validation failed'):
        scheduler.run(check_disk=False)


def test_recovery_preserves_completed_and_requeues_failed_and_stale_running(tmp_path):
    mod = load_module()
    tasks = [
        task(mod, "teacher:IEMOCAP:fold2"),
        task(mod, "teacher:IEMOCAP:fold3"),
        task(mod, "teacher:IEMOCAP:fold4"),
    ]
    state = tmp_path / "state"
    scheduler = mod.Scheduler(mod.Plan(tasks), state, ("GPU-a",))
    for index, status in enumerate(("completed", "failed", "running"), 2):
        attempt = state / "attempts" / f"teacher__IEMOCAP__fold{index}" / "attempt-0001"
        (attempt / "output").mkdir(parents=True)
        scheduler.ledger["tasks"][f"teacher:IEMOCAP:fold{index}"].update(
            status=status, attempts=[{"attempt": 1, "directory": str(attempt), "returncode": 0}],
        )
    scheduler._write_ledger()
    recovered = mod.Scheduler(mod.Plan(tasks), state, ("GPU-a",), recover_existing=True)
    assert recovered.ledger["tasks"]["teacher:IEMOCAP:fold2"]["status"] == "completed"
    assert recovered.ledger["tasks"]["teacher:IEMOCAP:fold3"]["status"] == "pending"
    assert recovered.ledger["tasks"]["teacher:IEMOCAP:fold4"]["status"] == "pending"
