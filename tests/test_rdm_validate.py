"""Tests for the rocq-doc-manager corpus and result runner."""

import json
import os

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


def test_manifest_rejects_duplicate_case(tmp_path):
    case = {
        "id": "same",
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
        },
        ]
    )
    summary = rdm_validate.summarize_records(records, manifest)
    assert summary["false_negative_count"] == 1
    assert summary["false_positive_count"] == 0
    assert summary["compiler_runtime_total"] == 5.0
    assert summary["document_runtime_total"] == 2.5
    assert summary["edit_strategy_counts"] == {"clear": 1, "replace": 1}
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
