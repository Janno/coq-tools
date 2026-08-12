"""Characterization tests for the minimizer's candidate transaction seam."""

import os

import pytest

from coq_tools import diagnose_error, find_bug
from coq_tools.candidate_evaluator import (
    CHANGE_FAILURE,
    CHANGE_SUCCESS,
    CONTENTS_UNCHANGED,
    CandidateChange,
    CandidateCheckCoordinator,
    CandidateCheckpoint,
    CandidateEvaluator,
    CandidateFinalizationError,
    Evaluation,
    EvaluationContext,
    EvaluationStatus,
    EvaluationTrial,
    MemoryLimitPlan,
    ResourcePolicy,
    TimeoutPolicy,
)


ERROR_TARGET = 'File "candidate.v", line 1, characters 0-1:\nError:\ntarget\n'
ERROR_OTHER = 'File "candidate.v", line 1, characters 0-1:\nError:\nother\n'


@pytest.fixture(autouse=True)
def avoid_real_compiler_version_probe(monkeypatch):
    monkeypatch.setattr(
        find_bug,
        "get_header_dict",
        lambda contents, **kwargs: kwargs.get(
            "header_dict",
            {
                "recent_runtime": 0,
                "recent_peak_rss_kb": 0,
                "old_header": "original input",
            },
        ),
    )


def observation(output, returncode=0, runtime=1.0, peak=10.0):
    return Evaluation(
        EvaluationStatus.SUCCESS,
        output,
        ("coqc", "candidate.v"),
        returncode,
        runtime,
        peak,
    )


class FakeEvaluator(CandidateEvaluator):
    def __init__(self, observations, events=None, output_path=None):
        self.observations = list(observations)
        self.events = [] if events is None else events
        self.output_path = output_path
        self.begin_count = 0
        self.materialized = []
        self.finished = []

    @property
    def identity(self):
        return "find-bug-fake"

    @property
    def requires_materialization_for_accept(self):
        return False

    def materialize_context(self, spec):
        self.materialized.append(spec)
        request = spec.resource_request
        policy = ResourcePolicy(
            request,
            TimeoutPolicy(request.requested_timeout, None, False),
            MemoryLimitPlan(request, None, None, None, None, "fixed"),
        )
        return EvaluationContext(
            self.identity,
            spec.executable,
            spec.arguments,
            spec.cwd,
            spec.environment,
            spec.logical_file,
            spec.top_name,
            spec.is_toplevel,
            spec.pass_on_stdin,
            spec.checker_executable,
            spec.checker_arguments,
            policy,
            role=spec.role,
        )

    def begin(self, context, candidate, target_policy=None):
        assert isinstance(candidate, CandidateChange)
        self.events.append("begin:%s" % context.executable[0])
        result = self.observations.pop(0)
        token = self.begin_count
        self.begin_count += 1
        return EvaluationTrial(result, token, True, False)

    def finish(self, trial, accepted):
        if accepted and self.output_path is not None:
            assert os.path.exists(self.output_path)
        self.events.append("finish:%s:%s" % (trial.token, accepted))
        self.finished.append((trial.token, accepted))

    def reset_calibration(self, context=None):
        self.events.append("reset")

    def close(self):
        self.events.append("close")


def test_final_hybrid_verification_is_fresh_and_checks_both_roles(monkeypatch):
    events = []
    policy = find_bug.StrictHybridTargetPolicy(False, "target")
    primary = object()
    passing = object()
    coordinator = type("Coordinator", (), {})()
    coordinator.last_checkpoint = CandidateCheckpoint(
        "raw",
        "serialized",
        "out.v",
        primary,
        passing,
        policy,
    )

    class Verifier(object):
        def __init__(self, log, verbose_base=2):
            events.append("construct")

        def begin(self, context, candidate, target_policy=None):
            assert isinstance(candidate, CandidateChange)
            events.append(("begin", context, candidate, target_policy))
            if context is primary:
                value = Evaluation(
                    EvaluationStatus.COMMAND_ERROR, ERROR_TARGET, (), 1
                )
            else:
                value = Evaluation(EvaluationStatus.SUCCESS, "", (), 0)
            return EvaluationTrial(value, context, False, False)

        def finish(self, trial, accepted):
            events.append(("finish", trial.token, accepted))

        def close(self):
            events.append("close")

    monkeypatch.setattr(find_bug, "CoqcEvaluator", Verifier)
    find_bug.verify_final_hybrid_checkpoint(
        coordinator, lambda *args, **kwargs: None
    )
    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "begin"] == [primary, passing]
    assert events[-1] == "close"


def _candidate(old_source, source):
    return CandidateChange.from_sources(old_source, source)


def test_definition_change_emits_exact_statement_range():
    old = (
        {"statement": "Definition a := 0.\n"},
        {"statement": "Definition b := 1.\n"},
        {"statement": "Check b.\n"},
    )
    coordinator = type("Coordinator", (), {})()
    coordinator.accepted_source = find_bug.join_definitions(old)
    deleted = find_bug._candidate_from_definition_change(
        old, old[:1] + old[2:], coordinator
    )
    assert deleted.edits[0].kind == "delete"
    assert deleted.source == find_bug.join_definitions(old[:1] + old[2:])
    assert deleted.edits[0].start == len(old[0]["statement"])
    assert deleted.edits[0].end == (
        len(old[0]["statement"]) + 1 + len(old[1]["statement"])
    )

    changed = dict(old[1], statement="Definition b := 2.\n")
    replaced = find_bug._candidate_from_definition_change(
        old, old[:1] + (changed,) + old[2:], coordinator
    )
    assert replaced.edits[0].kind == "replace"
    assert replaced.source == find_bug.join_definitions(
        old[:1] + (changed,) + old[2:]
    )
    assert replaced.edits[0].replacement == "\nDefinition b := 2.\n"
    assert replaced.edits[0].start == len(old[0]["statement"])


def test_definition_change_falls_back_when_metadata_base_is_stale():
    coordinator = type("Coordinator", (), {})()
    coordinator.accepted_source = "prefix accepted suffix"
    candidate = find_bug._candidate_from_definition_change(
        ({"statement": "stale"},),
        ({"statement": "prefix changed suffix"},),
        coordinator,
    )
    assert candidate.base_source == coordinator.accepted_source
    assert candidate.edits[0].kind == "inferred"


def make_env(tmp_path, observations, passing=False, events=None):
    evaluator = FakeEvaluator(observations, events=events)
    coordinator = CandidateCheckCoordinator(evaluator, "old")
    env = {
        "candidate_check_coordinator": coordinator,
        "coqc": ("bad-coqc",),
        "coqc_args": (),
        "timeout": None,
        "base_dir": str(tmp_path),
        "coqc_is_coqtop": False,
        "nonpassing_ocamlpath": None,
        "coqchk": None,
        "coqchk_args": (),
        "passing_coqc": (("good-coqc",) if passing else None),
        "passing_coqc_args": (),
        "passing_timeout": None,
        "passing_base_dir": str(tmp_path),
        "passing_coqc_is_coqtop": False,
        "passing_ocamlpath": None,
        "passing_coqchk": None,
        "passing_coqchk_args": (),
        "max_mem_rss": None,
        "max_mem_as": None,
        "max_mem_rss_multiplier": None,
        "max_mem_as_multiplier": None,
        "mem_limit_method": "none",
        "cgroup": None,
        "error_reg_string": "target",
        "should_succeed": False,
        "header": "",
        "dynamic_header": "",
        "header_dict": {
            "recent_runtime": 0,
            "recent_peak_rss_kb": 0,
            "old_header": "original input",
        },
        "inline_failure_libnames": [],
        "log": lambda *args, **kwargs: None,
        "color_on": False,
        "remove_temp_file": True,
        "temp_file_name": str(tmp_path / "rejected.v"),
        "temp_file_log_name": str(tmp_path / "rejected.log"),
    }
    return env, evaluator, coordinator


@pytest.mark.parametrize(
    "primary,passing,passing_configured,should_succeed,expected,index",
    (
        (observation(ERROR_TARGET, 1), None, False, False, CHANGE_SUCCESS, None),
        (observation(ERROR_OTHER, 1), None, False, False, CHANGE_FAILURE, 0),
        (
            observation(ERROR_TARGET, 1),
            observation("ok", 0),
            True,
            False,
            CHANGE_SUCCESS,
            None,
        ),
        (
            observation(ERROR_TARGET, 1),
            observation("plain nonzero stderr", 7),
            True,
            False,
            CHANGE_SUCCESS,
            None,
        ),
        (
            observation(ERROR_TARGET, 1),
            observation(ERROR_OTHER, 1),
            True,
            False,
            CHANGE_FAILURE,
            1,
        ),
        (
            observation(ERROR_TARGET, 1),
            observation("slow" + diagnose_error.TIMEOUT_POSTFIX, 1),
            True,
            False,
            CHANGE_FAILURE,
            1,
        ),
        (observation("plain stderr", 7), None, False, True, CHANGE_SUCCESS, None),
        (observation(ERROR_OTHER, 0), None, False, True, CHANGE_FAILURE, 0),
    ),
)
def test_candidate_decision_matrix(
    tmp_path,
    primary,
    passing,
    passing_configured,
    should_succeed,
    expected,
    index,
):
    observations = [primary] + ([] if passing is None else [passing])
    env, evaluator, coordinator = make_env(
        tmp_path, observations, passing=passing_configured
    )
    env["should_succeed"] = should_succeed
    decision = find_bug.classify_candidate(
        _candidate("old", "new"),
        logical_file_name=str(tmp_path / "out.v"),
        **env
    )
    assert decision.result_type == expected
    assert decision.bad_output_index == index
    expected_outputs = (primary.output,) + (
        () if passing is None else (passing.output,)
    )
    assert decision.outputs == expected_outputs
    coordinator.discard_candidate(decision.attempt)


def test_context_specs_follow_each_request_and_preserve_explicit_memory_key(
    tmp_path
):
    env, evaluator, coordinator = make_env(tmp_path, [])
    first = find_bug._candidate_context_spec(str(tmp_path / "out.v"), **env)
    changed_env = dict(env, coqc_args=("-debug",))
    second = find_bug._candidate_context_spec(
        str(tmp_path / "out.v"), **changed_env
    )
    assert first != second
    assert first.arguments == ()
    assert second.arguments == ("-debug",)
    assert first.resource_request.memory_usage_key == env["coqc"]
    explicit = dict(env, memory_usage_key=None)
    explicit_spec = find_bug._candidate_context_spec(
        str(tmp_path / "out.v"), **explicit
    )
    assert explicit_spec.resource_request.memory_usage_key is None


def test_unchanged_short_circuit_does_not_materialize_or_execute(tmp_path):
    env, evaluator, coordinator = make_env(tmp_path, [])
    coordinator.accepted_source = "same"
    decision = find_bug.classify_candidate(
        _candidate("same", "same"),
        logical_file_name=str(tmp_path / "out.v"),
        **env
    )
    assert decision.result_type == CONTENTS_UNCHANGED
    assert decision.evaluations == ()
    assert decision.attempt is None
    assert evaluator.materialized == []
    assert evaluator.begin_count == 0


def test_header_uses_primary_or_passing_runtime_and_peak(tmp_path):
    primary = observation(ERROR_TARGET, 1, runtime=2.5, peak=25.0)
    passing = observation("ok", 0, runtime=7.5, peak=75.0)
    env, evaluator, coordinator = make_env(tmp_path, [primary, passing], passing=True)
    env["dynamic_header"] = (
        "(* runtime %(recent_runtime)s rss %(recent_peak_rss_kb)s *)"
    )
    decision = find_bug.classify_candidate(
        _candidate("old", "new"),
        logical_file_name=str(tmp_path / "out.v"),
        **env
    )
    assert "runtime 7.5 rss 75.0" in decision.serialized_contents
    assert decision.runtime == 7.5
    assert decision.peak_rss_kb == 75.0
    coordinator.discard_candidate(decision.attempt)

    env, evaluator, coordinator = make_env(tmp_path, [primary])
    env["dynamic_header"] = (
        "(* runtime %(recent_runtime)s rss %(recent_peak_rss_kb)s *)"
    )
    decision = find_bug.classify_candidate(
        _candidate("old", "new"),
        logical_file_name=str(tmp_path / "out.v"),
        **env
    )
    assert "runtime 2.5 rss 25.0" in decision.serialized_contents
    coordinator.discard_candidate(decision.attempt)


def test_accepted_write_precedes_finish_and_checkpoint_separates_raw(tmp_path):
    output_path = str(tmp_path / "accepted.v")
    events = []
    env, evaluator, coordinator = make_env(
        tmp_path, [observation(ERROR_TARGET, 1)], events=events
    )
    evaluator.output_path = output_path
    assert find_bug.check_candidate_and_write_to_file(
        _candidate("old", "raw candidate"), output_path, **env
    )
    assert os.path.exists(output_path)
    assert events[-1] == "finish:0:True"
    assert coordinator.last_checkpoint.raw_source == "raw candidate"
    assert coordinator.last_checkpoint.serialized_contents != "raw candidate"
    assert coordinator.last_checkpoint.output_file_name == output_path


def test_rejection_does_not_modify_main_output_and_writer_failure_discards(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "out.v"
    output_path.write_text("canonical")
    env, evaluator, coordinator = make_env(
        tmp_path, [observation(ERROR_OTHER, 1)]
    )
    assert not find_bug.check_candidate_and_write_to_file(
        _candidate("old", "rejected"), str(output_path), **env
    )
    assert output_path.read_text() == "canonical"
    assert evaluator.finished == [(0, False)]

    env, evaluator, coordinator = make_env(
        tmp_path, [observation(ERROR_TARGET, 1)]
    )

    def failing_writer(*args, **kwargs):
        raise IOError("writer failed")

    monkeypatch.setattr(find_bug, "write_to_file_or_shorten_name", failing_writer)
    with pytest.raises(IOError, match="writer failed"):
        find_bug.check_candidate_and_write_to_file(
            _candidate("old", "accepted"), str(output_path), **env
        )
    assert evaluator.finished == [(0, False)]
    assert output_path.read_text() == "canonical"


def test_timeout_retry_count_bypasses_cache_and_writes_diagnostics_first(
    tmp_path, monkeypatch
):
    events = []
    timeout = observation("slow" + diagnose_error.TIMEOUT_POSTFIX, 1)
    env, evaluator, coordinator = make_env(
        tmp_path, [timeout, timeout, timeout], events=events
    )
    env["remove_temp_file"] = False
    writes = []

    def recording_writer(path, contents):
        writes.append(path)
        events.append("write:%s" % os.path.basename(path))
        return path

    monkeypatch.setattr(
        find_bug, "write_to_file_or_shorten_name", recording_writer
    )
    assert not find_bug.check_candidate_and_write_to_file(
        _candidate("old", "candidate"),
        str(tmp_path / "main.v"),
        timeout_retry_count=find_bug.SENSITIVE_TIMEOUT_RETRY_COUNT,
        write_to_temp_file=True,
        **env,
    )
    assert evaluator.begin_count == 3
    assert evaluator.finished == [(0, False), (1, False), (2, False)]
    assert len(writes) == 6
    first_finish = events.index("finish:0:False")
    assert events.index("write:rejected.v") < first_finish
    assert events.index("write:rejected.log") < first_finish
    assert events[first_finish + 1] == "begin:bad-coqc"


def test_passing_timeout_retry_reruns_primary_and_passing(tmp_path):
    primary = observation(ERROR_TARGET, 1)
    passing_timeout = observation(
        "passing slow" + diagnose_error.TIMEOUT_POSTFIX, 1
    )
    observations = []
    for _ in range(find_bug.SENSITIVE_TIMEOUT_RETRY_COUNT):
        observations.extend((primary, passing_timeout))
    events = []
    env, evaluator, coordinator = make_env(
        tmp_path, observations, passing=True, events=events
    )
    assert not find_bug.check_candidate_and_write_to_file(
        _candidate("old", "candidate"),
        str(tmp_path / "main.v"),
        timeout_retry_count=find_bug.SENSITIVE_TIMEOUT_RETRY_COUNT,
        **env,
    )
    assert evaluator.begin_count == 6
    assert [event for event in events if event.startswith("begin:")] == [
        "begin:bad-coqc",
        "begin:good-coqc",
    ] * 3
    assert evaluator.finished == [
        (1, False),
        (0, False),
        (3, False),
        (2, False),
        (5, False),
        (4, False),
    ]


def test_rejected_temp_writer_exception_still_discards(tmp_path, monkeypatch):
    env, evaluator, coordinator = make_env(
        tmp_path, [observation(ERROR_OTHER, 1)]
    )
    env["remove_temp_file"] = False

    def failing_writer(*args, **kwargs):
        raise RuntimeError("diagnostic write failed")

    monkeypatch.setattr(find_bug, "write_to_file_or_shorten_name", failing_writer)
    with pytest.raises(RuntimeError, match="diagnostic write failed"):
        find_bug.check_candidate_and_write_to_file(
            _candidate("old", "candidate"),
            str(tmp_path / "main.v"),
            write_to_temp_file=True,
            **env,
        )
    assert evaluator.finished == [(0, False)]


def test_passing_rejection_description_preserves_legacy_command_text(tmp_path):
    env, evaluator, coordinator = make_env(
        tmp_path,
        [observation(ERROR_TARGET, 1), observation(ERROR_OTHER, 1)],
        passing=True,
    )
    decision = find_bug.classify_candidate(
        _candidate("old", "candidate"),
        logical_file_name=str(tmp_path / "out.v"),
        **env
    )
    assert decision.description == (
        "The alternate coqc (good-coqc) was supposed to pass, but instead emitted an error.  "
    )
    coordinator.discard_candidate(decision.attempt)


def test_serialization_error_wins_over_discard_and_logging_errors(
    tmp_path, monkeypatch
):
    env, evaluator, coordinator = make_env(
        tmp_path, [observation(ERROR_TARGET, 1)]
    )

    def failing_finish(trial, accepted):
        raise RuntimeError("discard failed")

    def failing_log(*args, **kwargs):
        raise RuntimeError("cleanup log failed")

    evaluator.finish = failing_finish
    env["log"] = failing_log
    monkeypatch.setattr(
        find_bug,
        "prepend_header",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("serialize failed")),
    )
    with pytest.raises(ValueError, match="serialize failed"):
        find_bug.classify_candidate(
            _candidate("old", "candidate"),
            logical_file_name=str(tmp_path / "out.v"),
            **env
        )


def test_success_message_logging_failure_discards_before_return(tmp_path):
    env, evaluator, coordinator = make_env(
        tmp_path, [observation(ERROR_TARGET, 1)]
    )

    def selective_log(message, **kwargs):
        if "Change successful." in str(message):
            raise RuntimeError("success log failed")

    env["log"] = selective_log
    with pytest.raises(RuntimeError, match="success log failed"):
        find_bug.check_candidate_and_write_to_file(
            _candidate("old", "candidate"), str(tmp_path / "out.v"), **env
        )
    assert evaluator.finished == [(0, False)]
    assert not coordinator._outstanding


def test_accepted_finalization_failure_leaves_written_file_and_checkpoint(
    tmp_path
):
    output_path = tmp_path / "accepted.v"
    env, evaluator, coordinator = make_env(
        tmp_path, [observation(ERROR_TARGET, 1)]
    )
    original_finish = evaluator.finish

    def failing_accept(trial, accepted):
        original_finish(trial, accepted)
        if accepted:
            raise RuntimeError("promotion failed")

    evaluator.finish = failing_accept
    with pytest.raises(CandidateFinalizationError) as excinfo:
        find_bug.check_candidate_and_write_to_file(
            _candidate("old", "raw candidate"), str(output_path), **env
        )
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert output_path.exists()
    serialized = output_path.read_text()
    assert coordinator.last_checkpoint.raw_source == "raw candidate"
    assert coordinator.last_checkpoint.serialized_contents == serialized
    assert coordinator.last_checkpoint.output_file_name == str(output_path)


def test_primary_and_passing_specs_capture_all_role_shaping_fields(tmp_path):
    env, evaluator, coordinator = make_env(tmp_path, [])
    env.update(
        {
            "coqc_args": ("-primary",),
            "timeout": 3,
            "nonpassing_ocamlpath": "primary-ocaml",
            "coqc_is_coqtop": True,
            "coqchk": ("primary-checker",),
            "coqchk_args": ("-primary-check",),
            "passing_coqc": ("passing",),
            "passing_coqc_args": ("-passing",),
            "passing_timeout": 7,
            "passing_base_dir": str(tmp_path / "passing-cwd"),
            "passing_ocamlpath": "passing-ocaml",
            "passing_coqc_is_coqtop": True,
            "passing_coqchk": ("passing-checker",),
            "passing_coqchk_args": ("-passing-check",),
        }
    )
    (tmp_path / "passing-cwd").mkdir()
    primary = find_bug._candidate_context_spec(str(tmp_path / "out.v"), **env)
    passing = find_bug._candidate_context_spec(
        str(tmp_path / "out.v"), passing=True, **env
    )
    assert primary.arguments == ("-primary",)
    assert primary.role == "primary"
    assert primary.resource_request.requested_timeout == 3
    assert primary.environment.as_dict()["OCAMLPATH"] == "primary-ocaml"
    assert primary.is_toplevel
    assert primary.checker_executable == ("primary-checker",)
    assert primary.checker_arguments == ("-primary-check",)
    assert passing.executable == ("passing",)
    assert passing.arguments == ("-passing",)
    assert passing.role == "passing"
    assert passing.resource_request.requested_timeout == 7
    assert passing.cwd == str(tmp_path / "passing-cwd")
    assert passing.environment.as_dict()["OCAMLPATH"] == "passing-ocaml"
    assert passing.is_toplevel
    assert passing.checker_executable == ("passing-checker",)
    assert passing.checker_arguments == ("-passing-check",)
