"""Tests for the rocq-doc-manager corpus and result runner."""

import json
import os
import subprocess

import pytest

from coq_tools import rdm_validate


def test_manifest_names_every_example_and_twenty_archived_bugs():
    repository = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(repository, "tests", "rdm_corpus.json")
    has_examples = os.path.isdir(os.path.join(repository, "examples"))
    manifest = rdm_validate.load_manifest(
        path, repository=(repository if has_examples else None)
    )
    assert len(manifest["cases"]) == 78
    assert sum(case["archived_bug"] for case in manifest["cases"]) == 20
    if has_examples:
        assert {case["script"] for case in manifest["cases"]} == rdm_validate._example_scripts(repository)


def _manifest_case(identifier="example-000"):
    return {
        "id": identifier,
        "kind": "example-script",
        "script": "examples/run-example-000.sh",
        "source_revision": "0" * 40,
        "expected_category": "ordinary-command-error",
        "expected_fallback": None,
        "platforms": ["linux"],
        "toolchains": ["rocq-9.2-pinned"],
        "environment_keys": ["PATH"],
        "archived_bug": True,
        "archive_provenance": "fixture",
        "expected_outcome": "pass",
    }


def test_manifest_rejects_duplicate_case(tmp_path):
    case = {
        "id": "example-000",
        "kind": "example-script",
        "script": "examples/run-example-000.sh",
        "source_revision": "0" * 40,
        "expected_category": "ordinary-command-error",
        "expected_fallback": None,
        "platforms": ["linux"],
        "toolchains": ["rocq-9.2-pinned"],
        "environment_keys": ["PATH"],
        "archived_bug": True,
        "archive_provenance": "fixture",
        "expected_outcome": "pass",
    }
    manifest = {
        "schema": 1,
        "declared_configuration": {},
        "archive_policy": {"required_count": 2},
        "cases": [case, dict(case)],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate case id"):
        rdm_validate.load_manifest(str(path))


def test_manifest_rejects_case_id_path_traversal(tmp_path):
    manifest = {
        "schema": 1,
        "declared_configuration": {},
        "archive_policy": {"required_count": 0},
        "cases": [_manifest_case("../../caller")],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid case id"):
        rdm_validate.load_manifest(str(path))


def test_summary_classifies_direction_and_rollout_gates():
    manifest = {
        "declared_configuration": {"minimum_comparisons": 2},
        "cases": [
            {"archived_bug": True},
            {"archived_bug": True},
        ]
        + [{"archived_bug": True} for _ in range(18)],
    }
    records = (
        [{"event": "case-start", "case_id": "case-%d" % index} for index in range(20)]
        + [
            {
                "event": "case-end",
                "case_id": "case-%d" % index,
                "returncode": 0,
                "timed_out": False,
            }
            for index in range(20)
        ]
        + [
            {
                "event": "candidate-comparison",
            "agreement": False,
            "document_verdict": False,
            "compiler_verdict": True,
            "reason": "classified",
            "compiler_runtime": 2.0,
            "document_runtime": 1.0,
            "document_split_runtime": 0.1,
            "edit_strategy": "clear",
            "edit_kind": "delete",
        },
        {
            "event": "candidate-comparison",
            "agreement": True,
            "document_verdict": True,
            "compiler_verdict": True,
            "compiler_runtime": 3.0,
            "document_runtime": 1.5,
            "document_split_runtime": 0.2,
            "edit_strategy": "replace",
            "edit_kind": "inferred",
        },
        ]
    )
    summary = rdm_validate.summarize_records(records, manifest)
    assert summary["false_negative_count"] == 1
    assert summary["false_positive_count"] == 0
    assert summary["compiler_runtime_total"] == 5.0
    assert summary["document_runtime_total"] == 2.5
    assert summary["edit_strategy_counts"] == {"clear": 1, "replace": 1}
    assert summary["edit_kind_counts"] == {"delete": 1, "inferred": 1}
    assert summary["edit_strategy_split_runtime_total"] == {
        "clear": 0.1,
        "replace": 0.2,
    }
    assert summary["case_attempts"] == 20
    assert all(summary["gates"].values())


def test_summary_uses_latest_rerun_for_case_outcome():
    manifest = {
        "declared_configuration": {"minimum_comparisons": 0},
        "cases": [{"archived_bug": True} for _ in range(20)],
    }
    records = []
    for index in range(20):
        records.append({"event": "case-start", "case_id": str(index)})
        records.append(
            {
                "event": "case-end",
                "case_id": str(index),
                "returncode": 1 if index == 0 else 0,
                "timed_out": False,
            }
        )
    records.extend(
        [
            {"event": "case-start", "case_id": "0"},
            {"event": "case-end", "case_id": "0", "returncode": 0, "timed_out": False},
        ]
    )
    summary = rdm_validate.summarize_records(records, manifest)
    assert summary["case_attempts"] == 21
    assert summary["case_ends"] == 20
    assert summary["failed_cases"] == 0
    assert summary["gates"]["all_executed_cases_passed"]


def test_summary_rejects_unclassified_disagreement():
    summary = rdm_validate.summarize_records(
        [
            {
                "event": "candidate-comparison",
                "agreement": False,
                "document_verdict": True,
                "compiler_verdict": False,
            }
        ]
    )
    assert summary["false_positive_count"] == 1
    assert summary["unclassified_disagreement_count"] == 1
    assert not summary["gates"]["disagreements_classified"]


def _git(repository, *arguments):
    subprocess.check_call(["git"] + list(arguments), cwd=str(repository))


def test_isolated_repository_preserves_caller_generated_files(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "validator@example.invalid")
    _git(repository, "config", "user.name", "Validator")
    examples = repository / "examples"
    examples.mkdir()
    script = examples / "run-example-000.sh"
    script.write_text("#!/bin/sh\nprintf generated > bug_000.v\n")
    script.chmod(0o755)
    suffixed_directory = examples / "example_008"
    suffixed_directory.mkdir()
    (suffixed_directory / "tracked.v").write_text("Check nat.\n")
    (repository / ".gitignore").write_text(
        "examples/caller-ignored.txt\n"
        "examples/bug_000.v\n"
        "examples/example_008/bug_008_2.v\n"
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    ignored = examples / "caller-ignored.txt"
    untracked = examples / "caller-untracked.txt"
    ignored.write_text("keep ignored")
    untracked.write_text("keep untracked")

    isolated = tmp_path / "isolated"
    rdm_validate._create_isolated_repository(
        str(repository), str(isolated)
    )
    assert (isolated / ".git").is_dir()
    assert not (isolated / "examples" / "caller-ignored.txt").exists()
    assert not (isolated / "examples" / "caller-untracked.txt").exists()
    subprocess.check_call(
        ["bash", str(isolated / "examples" / "run-example-000.sh")],
        cwd=str(isolated / "examples"),
    )
    retained = rdm_validate._retain_outputs(
        str(isolated), str(tmp_path / "output"), {"id": "example-000"}
    )
    assert len(retained) == 1
    assert retained[0]["source_path"] == "examples/bug_000.v"
    assert subprocess.check_output(
        ["git", "check-ignore", "examples/bug_000.v"], cwd=str(isolated)
    ).strip()
    suffixed_output = isolated / "examples" / "example_008" / "bug_008_2.v"
    suffixed_output.write_text("Fail suffix.\n")
    assert rdm_validate._untracked_v_files(
        str(isolated), {"id": "example-008-2"}
    ) == ["examples/example_008/bug_008_2.v"]
    rdm_validate._clean_case(
        str(isolated), {"id": "example-008-2"}
    )
    assert not suffixed_output.exists()
    assert (isolated / "examples" / "example_008" / "tracked.v").exists()
    assert ignored.read_text() == "keep ignored"
    assert untracked.read_text() == "keep untracked"


def test_run_corpus_uses_disposable_clone_and_retains_outputs(
    tmp_path, monkeypatch
):
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "validator@example.invalid")
    _git(repository, "config", "user.name", "Validator")
    examples = repository / "examples"
    examples.mkdir()
    script = examples / "run-example-000.sh"
    script.write_text(
        '#!/bin/sh\ncd "$(dirname "$0")"\nprintf generated > bug_000.v\n'
    )
    script.chmod(0o755)
    (repository / ".gitignore").write_text(
        "examples/caller-ignored.txt\nexamples/bug_000.v\n"
    )
    manifest = {
        "schema": 1,
        "declared_configuration": {"minimum_comparisons": 0},
        "archive_policy": {"required_count": 0},
        "cases": [
            {
                "id": "example-000",
                "kind": "example-script",
                "script": "examples/run-example-000.sh",
                "source_revision": "fixture",
                "expected_category": "fixture",
                "expected_fallback": None,
                "platforms": ["test"],
                "toolchains": ["test"],
                "environment_keys": [],
                "archived_bug": False,
                "archive_provenance": "fixture",
                "expected_outcome": "pass",
            }
        ],
    }
    manifest_path = repository / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    ignored = examples / "caller-ignored.txt"
    untracked = examples / "caller-untracked.txt"
    ignored.write_text("keep ignored")
    untracked.write_text("keep untracked")

    work_directory = tmp_path / "validator-work"

    def make_work_directory(*args, **kwargs):
        work_directory.mkdir()
        return str(work_directory)

    monkeypatch.setattr(
        rdm_validate.tempfile, "mkdtemp", make_work_directory
    )
    output = tmp_path / "output"
    arguments = rdm_validate.build_parser().parse_args(
        [
            "--repository",
            str(repository),
            "run-corpus",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--manager",
            str(tmp_path / "missing-manager"),
            "--coqc",
            str(tmp_path / "missing-coqc"),
        ]
    )
    assert arguments.action(arguments) == 2
    assert ignored.read_text() == "keep ignored"
    assert untracked.read_text() == "keep untracked"
    assert not (examples / "bug_000.v").exists()
    retained = list((output / "final" / "example-000").iterdir())
    assert len(retained) == 1
    assert retained[0].read_text() == "generated"
    assert not work_directory.exists()

    inside_output = repository / "validation-output"
    inside_arguments = rdm_validate.build_parser().parse_args(
        [
            "--repository",
            str(repository),
            "run-corpus",
            "--manifest",
            str(manifest_path),
            "--output",
            str(inside_output),
        ]
    )
    with pytest.raises(RuntimeError, match="outside"):
        inside_arguments.action(inside_arguments)
    assert not inside_output.exists()
    assert ignored.read_text() == "keep ignored"
    assert untracked.read_text() == "keep untracked"

    hostile_output = tmp_path / "hostile-output"
    hostile_output.mkdir()
    (hostile_output / rdm_validate.OUTPUT_MARKER).write_text(
        json.dumps({"schema": 1, "kind": "rdm-corpus-output"})
    )
    original_manifest = manifest_path.read_bytes()
    (hostile_output / "summary.json").symlink_to(manifest_path)
    hostile_arguments = rdm_validate.build_parser().parse_args(
        [
            "--repository",
            str(repository),
            "run-corpus",
            "--manifest",
            str(manifest_path),
            "--output",
            str(hostile_output),
            "--resume",
        ]
    )
    with pytest.raises(RuntimeError, match="symlinked"):
        hostile_arguments.action(hostile_arguments)
    assert manifest_path.read_bytes() == original_manifest

    hardlink_output = tmp_path / "hardlink-output"
    hardlink_output.mkdir()
    (hardlink_output / rdm_validate.OUTPUT_MARKER).write_text(
        json.dumps({"schema": 1, "kind": "rdm-corpus-output"})
    )
    os.link(str(manifest_path), str(hardlink_output / "summary.json"))
    hardlink_arguments = rdm_validate.build_parser().parse_args(
        [
            "--repository",
            str(repository),
            "run-corpus",
            "--manifest",
            str(manifest_path),
            "--output",
            str(hardlink_output),
            "--resume",
        ]
    )
    with pytest.raises(RuntimeError, match="hard-linked"):
        hardlink_arguments.action(hardlink_arguments)
    assert manifest_path.read_bytes() == original_manifest


def test_isolated_repository_rejects_tracked_caller_changes(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "validator@example.invalid")
    _git(repository, "config", "user.name", "Validator")
    tracked = repository / "tracked"
    tracked.write_text("committed")
    _git(repository, "add", "tracked")
    _git(repository, "commit", "-m", "fixture")
    tracked.write_text("modified")
    with pytest.raises(RuntimeError, match="clean index"):
        rdm_validate._create_isolated_repository(
            str(repository), str(tmp_path / "isolated")
        )


def test_resume_validation_propagates_walk_errors(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / rdm_validate.OUTPUT_MARKER).write_text(
        json.dumps({"schema": 1, "kind": "rdm-corpus-output"})
    )

    def failing_walk(path, followlinks=False, onerror=None):
        assert onerror is not None
        error = OSError(13, "unreadable subtree", str(output / "wrappers"))
        onerror(error)
        return iter(())

    monkeypatch.setattr(rdm_validate.os, "walk", failing_walk)
    with pytest.raises(RuntimeError, match="unreadable subtree"):
        rdm_validate._initialize_output_directory(str(output), True)


def test_clean_and_output_discovery_report_git_failures(tmp_path):
    case = {"id": "example-000"}
    with pytest.raises(RuntimeError, match="git clean"):
        rdm_validate._clean_case(str(tmp_path), case)
    with pytest.raises(RuntimeError, match="git ls-files"):
        rdm_validate._untracked_v_files(str(tmp_path), case)


def test_run_corpus_help_describes_disposable_clone(capsys):
    parser = rdm_validate.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run-corpus", "--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "disposable clone" in output
    assert "not copied or cleaned" in output
    assert "--output" in output


def test_wrapper_adds_backend_to_calls_without_script_argument_forwarding(tmp_path):
    path = tmp_path / "find-bug-wrapper"
    rdm_validate._write_wrapper(
        str(path),
        "/python",
        "/repo",
        "rdm-shadow",
        "/manager",
    )
    contents = path.read_text()
    assert '"$@"' in contents
    assert "--backend=rdm-shadow" in contents
    assert "--rdm=/manager" in contents


def test_benchmark_source_changes_only_late_suffix():
    prefix, baseline = rdm_validate._benchmark_source(25)
    candidate = prefix + "Check I.\n"
    assert baseline.startswith(prefix)
    assert candidate.startswith(prefix)
    assert prefix.count("Definition bench_") == 25
