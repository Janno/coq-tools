"""Pure tests for the candidate evaluator contract and coordinator."""

import os
import stat

import pytest

from coq_tools import diagnose_error
from coq_tools.candidate_evaluator import (
    CHANGE_FAILURE,
    CHANGE_SUCCESS,
    CandidateCheckCoordinator,
    CandidateCoordinatorUnhealthy,
    CandidateEvaluator,
    CandidateEvaluatorError,
    CandidateFinalizationError,
    CandidateLifecycleError,
    CoqcEvaluator,
    EnvironmentSnapshot,
    Evaluation,
    EvaluationContext,
    EvaluationContextSpec,
    EvaluationStatus,
    EvaluationTrial,
    LegacyTargetPolicy,
    MemoryLimitPlan,
    ResourcePolicy,
    ResourceRequest,
    StrictHybridTargetPolicy,
    TimeoutPolicy,
)


def _evaluation(output="", returncode=0, runtime=1.0, peak=2.0):
    status = EvaluationStatus.SUCCESS if returncode == 0 else EvaluationStatus.CRASH
    return Evaluation(status, output, ("coqc", "tmp.v"), returncode, runtime, peak)


def _policy(request=None):
    request = request or ResourceRequest(None, memory_usage_key=("coqc",))
    return ResourcePolicy(
        request,
        TimeoutPolicy(request.requested_timeout, None, False),
        MemoryLimitPlan(request, None, None, None, None, "fixed"),
    )


def _context(spec, identity="fake"):
    return EvaluationContext(
        identity,
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
        _policy(spec.resource_request),
        role=spec.role,
    )


class RecordingEvaluator(CandidateEvaluator):
    def __init__(self, outputs, requires_materialization=False, fail_accept_at=None):
        self.outputs = list(outputs)
        self.materialized = []
        self.begun = []
        self.finished = []
        self.reset_count = 0
        self.closed = False
        self._requires_materialization = requires_materialization
        self.fail_accept_at = fail_accept_at

    @property
    def identity(self):
        return "recording"

    @property
    def requires_materialization_for_accept(self):
        return self._requires_materialization

    def materialize_context(self, spec):
        self.materialized.append(spec)
        return _context(spec, self.identity)

    def begin(self, context, source, target_policy=None):
        evaluation = self.outputs.pop(0)
        token = len(self.begun)
        trial = EvaluationTrial(evaluation, token, True, False)
        self.begun.append((context, source, trial))
        return trial

    def finish(self, trial, accepted):
        self.finished.append((trial.token, accepted))
        if accepted and self.fail_accept_at == trial.token:
            raise RuntimeError("promotion failed")

    def reset_calibration(self, context=None):
        self.reset_count += 1

    def close(self):
        self.closed = True


def _spec(
    name="coqc", timeout=None, multiplier=None, environment=None, role="primary"
):
    executable = (name,)
    return EvaluationContextSpec(
        executable,
        ("-q",),
        environment=environment or {"PATH": os.environ.get("PATH", "")},
        logical_file="candidate.v",
        resource_request=ResourceRequest(
            timeout,
            max_mem_rss_multiplier=multiplier,
            memory_usage_key=executable,
        ),
        role=role,
    )


def test_environment_and_specs_are_deeply_immutable_and_redacted(tmp_path):
    source_env = {"PATH": str(tmp_path), "SECRET_TOKEN": "do-not-print"}
    source_args = ["-q"]
    snapshot = EnvironmentSnapshot(source_env)
    spec = EvaluationContextSpec(
        ("coqc",), source_args, environment=snapshot, logical_file="x.v"
    )
    before_hash = hash(spec)
    source_env["SECRET_TOKEN"] = "changed"
    source_args.append("-debug")
    os.environ["ENVIRONMENT_SNAPSHOT_TEST"] = "changed"
    try:
        assert spec.arguments == ("-q",)
        assert spec.environment.as_dict()["SECRET_TOKEN"] == "do-not-print"
        assert hash(spec) == before_hash
        assert "do-not-print" not in repr(snapshot)
        assert "entry_count=2" in repr(snapshot)
        with pytest.raises((AttributeError, TypeError)):
            spec.cwd = "/other"
    finally:
        os.environ.pop("ENVIRONMENT_SNAPSHOT_TEST", None)


def test_context_role_defaults_to_primary_and_participates_in_identity():
    primary = _spec()
    passing = EvaluationContextSpec(
        primary.executable,
        primary.arguments,
        cwd=primary.cwd,
        environment=primary.environment,
        logical_file=primary.logical_file,
        resource_request=primary.resource_request,
        role="passing",
    )
    assert primary.role == "primary"
    assert passing.role == "passing"
    assert primary != passing
    assert _context(primary).role == "primary"
    assert _context(passing).role == "passing"
    with pytest.raises(ValueError, match="role"):
        EvaluationContextSpec(
            ("coqc",),
            environment={"PATH": ""},
            role="unknown",
        )


def test_evaluation_status_and_legacy_tuple_are_stable():
    evaluation = Evaluation(
        EvaluationStatus.CRASH,
        "raw output",
        ["coqc", "x.v"],
        7,
        1.25,
        42.0,
        details={"signal": 9},
    )
    assert evaluation.as_legacy_tuple() == (
        "raw output",
        ("coqc", "x.v"),
        7,
        1.25,
        42.0,
    )
    assert set(EvaluationStatus.ALL) == {
        "success",
        "command_error",
        "parse_error",
        "timeout",
        "out_of_memory",
        "crash",
        "internal_error",
    }


def test_target_policy_preserves_legacy_nonzero_no_error_quirks():
    crash_without_generic_error = _evaluation("plain stderr", returncode=3)
    assert LegacyTargetPolicy(True, None).primary_preserves(
        crash_without_generic_error
    )
    assert LegacyTargetPolicy(False, "target").passing_succeeds(
        crash_without_generic_error
    )


def test_coqc_context_materialization_timeout_table_and_memory(monkeypatch):
    evaluator = CoqcEvaluator(lambda *args, **kwargs: None)
    executable = ("coqc",)
    diagnose_error.reset_timeout()
    diagnose_error.reset_memory_usage()
    diagnose_error.set_memory_usage((executable, "rss"), 100)
    diagnose_error.set_memory_usage((executable, "as"), 200)
    spec = EvaluationContextSpec(
        executable,
        environment={"PATH": os.environ.get("PATH", "")},
        resource_request=ResourceRequest(
            -1,
            10,
            20,
            1.5,
            2.0,
            "prlimit",
            None,
            executable,
        ),
    )
    first = evaluator.materialize_context(spec)
    assert first.resource_policy.timeout_policy == TimeoutPolicy(-1, None, True)
    assert first.resource_policy.memory_plan.compiler_max_mem_rss == 150
    assert first.resource_policy.memory_plan.compiler_max_mem_as == 400
    diagnose_error.TIMEOUT[executable] = 9
    second = evaluator.materialize_context(spec)
    assert second.resource_policy.timeout_policy == TimeoutPolicy(-1, 9, False)
    zero = evaluator.materialize_context(
        EvaluationContextSpec(
            executable,
            environment={"PATH": ""},
            resource_request=ResourceRequest(0, memory_usage_key=executable),
        )
    )
    assert zero.resource_policy.timeout_policy == TimeoutPolicy(0, None, False)


@pytest.mark.parametrize(
    "output,returncode,expected",
    (
        (
            'File "x.v", line 1, characters 0-1:\nError:\nboom\n'
            + diagnose_error.TIMEOUT_POSTFIX,
            1,
            EvaluationStatus.TIMEOUT,
        ),
        (
            'File "x.v", line 1, characters 0-1:\nError:\nboom\n'
            + diagnose_error.MEMORY_LIMIT_POSTFIXES[0],
            1,
            EvaluationStatus.OUT_OF_MEMORY,
        ),
        (
            'File "x.v", line 1, characters 0-1:\nError:\nboom\n',
            0,
            EvaluationStatus.COMMAND_ERROR,
        ),
        ("ok", 0, EvaluationStatus.SUCCESS),
        ("plain failure", 7, EvaluationStatus.CRASH),
        ("plain failure", None, EvaluationStatus.CRASH),
    ),
)
def test_coqc_status_precedence(output, returncode, expected):
    assert CoqcEvaluator._status(output, returncode) == expected


def test_coqc_adapter_maps_status_precedence_and_preserves_metadata(monkeypatch):
    stage = diagnose_error.SubprocessStageResult(
        "compiler", ("coqc",), None, None, None, 1, 1.0, 2.0
    )
    result = diagnose_error.CoqOutputResult(
        "File \"x.v\", line 1, characters 0-1:\nError:\nboom\n"
        + diagnose_error.TIMEOUT_POSTFIX,
        ("coqc", "x.v"),
        1,
        1.0,
        2.0,
        (stage,),
    )
    calls = []

    def fake_executor(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(diagnose_error, "_get_coq_output_result", fake_executor)
    monkeypatch.setattr(diagnose_error, "default_retry_with_debug_when", lambda x: False)
    evaluator = CoqcEvaluator(lambda *args, **kwargs: None)
    context = evaluator.materialize_context(_spec())
    trial = evaluator.begin(context, "Check nat.")
    assert trial.evaluation.status == EvaluationStatus.TIMEOUT
    assert trial.evaluation.as_legacy_tuple() == result.as_legacy_tuple()
    assert calls[0][1]["use_cache"] is False
    assert calls[0][1]["automatic_debug_retry"] is False
    assert calls[0][1]["process_environment"] == context.environment.as_dict()
    assert calls[0][1]["memory_plan"] == context.resource_policy.memory_plan


@pytest.mark.parametrize(
    "requested,expected_retry_deadline",
    ((-1, 12), (0, None), (5, 5), (None, None)),
)
def test_candidate_debug_retry_rematerializes_timeout_policy(
    monkeypatch, requested, expected_retry_deadline
):
    executable = ("coqc",)
    diagnose_error.reset_timeout()
    calls = []
    preliminary = diagnose_error.CoqOutputResult(
        "is not a compiled interface for this version of OCaml",
        ("coqc", "first.v"),
        1,
        1.0,
        1.0,
    )
    final = diagnose_error.CoqOutputResult(
        "final", ("coqc", "second.v"), 0, 2.0, 2.0
    )

    def fake_executor(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if requested is not None:
                diagnose_error.TIMEOUT[executable] = 12
            return preliminary
        return final

    debug_helper_calls = []

    def fake_debug_args(executable, **kwargs):
        debug_helper_calls.append((executable, kwargs))
        return ["-debug"]

    monkeypatch.setattr(diagnose_error, "_get_coq_output_result", fake_executor)
    monkeypatch.setattr(
        diagnose_error,
        "get_coq_debug_native_compiler_args",
        fake_debug_args,
    )
    evaluator = CoqcEvaluator(lambda *args, **kwargs: None)
    spec = EvaluationContextSpec(
        executable,
        ("-q",),
        environment={"PATH": ""},
        resource_request=ResourceRequest(requested, memory_usage_key=executable),
    )
    context = evaluator.materialize_context(spec)
    trial = evaluator.begin(context, "source")
    assert len(calls) == 2
    assert calls[1]["effective_timeout"] == expected_retry_deadline
    assert trial.evaluation.as_legacy_tuple() == final.as_legacy_tuple()
    assert calls[1]["use_cache"] is False
    assert debug_helper_calls[0][1]["executable_identity"] == context.executable_identity


def test_candidate_debug_retry_resolves_memory_from_latest_peak(monkeypatch):
    executable = ("coqc",)
    diagnose_error.reset_timeout()
    diagnose_error.reset_memory_usage()
    calls = []

    def fake_executor(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            diagnose_error.set_memory_usage((executable, "rss"), 300)
            diagnose_error.set_memory_usage((executable, "as"), 400)
            return diagnose_error.CoqOutputResult(
                "is not a compiled interface for this version of OCaml",
                ("coqc", "first.v"),
                1,
                1.0,
                1.0,
            )
        return diagnose_error.CoqOutputResult(
            "final", ("coqc", "second.v"), 0, 2.0, 2.0
        )

    monkeypatch.setattr(diagnose_error, "_get_coq_output_result", fake_executor)
    monkeypatch.setattr(
        diagnose_error,
        "get_coq_debug_native_compiler_args",
        lambda executable, **kwargs: ["-debug"],
    )
    request = ResourceRequest(
        None,
        max_mem_rss_multiplier=2.0,
        max_mem_as_multiplier=3.0,
        memory_usage_key=executable,
    )
    evaluator = CoqcEvaluator(lambda *args, **kwargs: None)
    context = evaluator.materialize_context(
        EvaluationContextSpec(
            executable,
            environment={"PATH": ""},
            resource_request=request,
        )
    )
    evaluator.begin(context, "source")
    retry_plan = calls[1]["memory_plan"]
    assert retry_plan.compiler_max_mem_rss == 600
    assert retry_plan.compiler_max_mem_as == 1200


def test_reporting_hook_failure_does_not_change_candidate_decision():
    evaluator = RecordingEvaluator([_evaluation("TARGET")])

    def broken_reporting_hook(*args, **kwargs):
        raise RuntimeError("reporting failed")

    evaluator.record_target_decision = broken_reporting_hook
    coordinator = CandidateCheckCoordinator(evaluator)
    verdict = coordinator.begin_candidate(
        "candidate",
        _spec(),
        None,
        LegacyTargetPolicy(False, "TARGET"),
    )
    assert verdict.result_type == CHANGE_SUCCESS
    coordinator.discard_candidate(verdict.attempt)


def test_coordinator_primary_passing_cache_bypass_and_reverse_discard():
    target = 'File "x.v", line 1, characters 0-1:\nError:\ntarget\n'
    primary = _evaluation(target, returncode=1)
    passing = _evaluation("all good")
    evaluator = RecordingEvaluator([primary, passing, primary, passing])
    coordinator = CandidateCheckCoordinator(evaluator)
    policy = LegacyTargetPolicy(False, "target")
    primary_spec = _spec("bad")
    passing_spec = EvaluationContextSpec(
        ("good",),
        ("-q",),
        environment={"PATH": os.environ.get("PATH", "")},
        logical_file="candidate.v",
        resource_request=ResourceRequest(
            None, memory_usage_key=("good",)
        ),
        role="passing",
    )

    first = coordinator.begin_candidate(
        "source", primary_spec, passing_spec, policy
    )
    assert first.result_type == CHANGE_SUCCESS
    assert len(evaluator.materialized) == 2
    coordinator.discard_candidate(first.attempt)
    assert evaluator.finished == [(1, False), (0, False)]

    cached = coordinator.begin_candidate(
        "source", primary_spec, passing_spec, policy
    )
    assert cached.result_type == CHANGE_SUCCESS
    assert len(evaluator.begun) == 2
    coordinator.commit_candidate(cached.attempt, "serialized", "out.v")
    assert coordinator.last_checkpoint.raw_source == "source"
    assert coordinator.last_checkpoint.serialized_contents == "serialized"

    bypass = coordinator.begin_candidate(
        "source", primary_spec, passing_spec, policy, bypass_cache=True
    )
    assert len(evaluator.begun) == 4
    coordinator.discard_candidate(bypass.attempt)


def test_coordinator_separates_target_decisions_and_stateful_cached_acceptance():
    target_output = 'File "x.v", line 1, characters 0-1:\nError:\ntarget\n'
    target_observation = _evaluation(target_output, returncode=1)
    evaluator = RecordingEvaluator([target_observation, target_observation])
    coordinator = CandidateCheckCoordinator(evaluator)
    failure_mode = coordinator.begin_candidate(
        "source", _spec(), None, LegacyTargetPolicy(False, "target")
    )
    assert failure_mode.result_type == CHANGE_SUCCESS
    coordinator.discard_candidate(failure_mode.attempt)
    success_mode = coordinator.begin_candidate(
        "source", _spec(), None, LegacyTargetPolicy(True, None)
    )
    assert success_mode.result_type == CHANGE_FAILURE
    assert len(evaluator.begun) == 2
    coordinator.discard_candidate(success_mode.attempt)
    assert len(coordinator.decision_cache) == 2

    success = _evaluation("ok")
    stateful = RecordingEvaluator(
        [success, success], requires_materialization=True
    )
    coordinator = CandidateCheckCoordinator(stateful)
    first = coordinator.begin_candidate(
        "source", _spec(), None, LegacyTargetPolicy(True, None)
    )
    coordinator.discard_candidate(first.attempt)
    second = coordinator.begin_candidate(
        "source", _spec(), None, LegacyTargetPolicy(True, None)
    )
    assert second.result_type == CHANGE_SUCCESS
    assert len(stateful.begun) == 2
    coordinator.discard_candidate(second.attempt)


def test_coordinator_conditional_passing_and_partial_failure_cleanup():
    no_target = _evaluation("different output", returncode=1)
    evaluator = RecordingEvaluator([no_target])
    coordinator = CandidateCheckCoordinator(evaluator)
    verdict = coordinator.begin_candidate(
        "source", _spec("bad"), _spec("good"), LegacyTargetPolicy(False, "target")
    )
    assert verdict.result_type == CHANGE_FAILURE
    assert len(evaluator.materialized) == 1
    coordinator.discard_candidate(verdict.attempt)

    target = _evaluation(
        'File "x.v", line 1, characters 0-1:\nError:\ntarget\n',
        returncode=1,
    )

    class PassingFailureEvaluator(RecordingEvaluator):
        def materialize_context(self, spec):
            if spec.executable == ("good",):
                raise RuntimeError("passing materialization failed")
            return super(PassingFailureEvaluator, self).materialize_context(spec)

    failing = PassingFailureEvaluator([target])
    coordinator = CandidateCheckCoordinator(failing)
    with pytest.raises(RuntimeError, match="passing materialization"):
        coordinator.begin_candidate(
            "source",
            _spec("bad"),
            _spec("good", role="passing"),
            LegacyTargetPolicy(False, "target"),
        )
    assert failing.finished == [(0, False)]


def test_reset_precedes_materialization_and_resource_side_effects_skip_cache():
    success = _evaluation("ok")
    evaluator = RecordingEvaluator([success])
    coordinator = CandidateCheckCoordinator(evaluator)
    spec = _spec(timeout=0)
    first = coordinator.begin_candidate(
        "source", spec, None, LegacyTargetPolicy(True, None), reset_calibration=True
    )
    coordinator.discard_candidate(first.attempt)
    second = coordinator.begin_candidate(
        "source", spec, None, LegacyTargetPolicy(True, None)
    )
    coordinator.discard_candidate(second.attempt)
    assert evaluator.reset_count == 1
    # Fake materialization marks no calibration side effect, so this primarily
    # proves reset orchestration; the production materializer is tested above.
    assert len(evaluator.materialized) == 2


def test_double_resolution_is_rejected():
    evaluator = RecordingEvaluator([_evaluation("ok")])
    coordinator = CandidateCheckCoordinator(evaluator)
    verdict = coordinator.begin_candidate(
        "source", _spec(), None, LegacyTargetPolicy(True, None)
    )
    coordinator.discard_candidate(verdict.attempt)
    with pytest.raises(CandidateLifecycleError):
        coordinator.discard_candidate(verdict.attempt)
    assert evaluator.finished == [(0, False)]


def test_exactly_once_and_accepted_finalization_failure_checkpoint():
    primary = _evaluation(
        'File "x.v", line 1, characters 0-1:\nError:\ntarget\n', 1
    )
    passing = _evaluation("ok")
    evaluator = RecordingEvaluator([primary, passing], fail_accept_at=1)
    coordinator = CandidateCheckCoordinator(evaluator)
    verdict = coordinator.begin_candidate(
        "raw",
        _spec("bad"),
        _spec("good", role="passing"),
        LegacyTargetPolicy(False, "target"),
    )
    with pytest.raises(CandidateFinalizationError) as excinfo:
        coordinator.commit_candidate(verdict.attempt, "serialized", "actual.v")
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert coordinator.last_checkpoint.raw_source == "raw"
    assert coordinator.last_checkpoint.output_file_name == "actual.v"
    assert not coordinator.healthy
    with pytest.raises(CandidateCoordinatorUnhealthy):
        coordinator.discard_candidate(verdict.attempt)


def test_fresh_observation_invalidates_decisions_for_every_policy():
    ok = _evaluation("ok")
    target = _evaluation(
        'File "x.v", line 1, characters 0-1:\nError:\ntarget\n', 1
    )
    evaluator = RecordingEvaluator([ok, ok, target, target])
    coordinator = CandidateCheckCoordinator(evaluator)
    spec = _spec()
    target_policy = LegacyTargetPolicy(False, "target")
    success_policy = LegacyTargetPolicy(True, None)

    initial_target = coordinator.begin_candidate("source", spec, None, target_policy)
    assert initial_target.result_type == CHANGE_FAILURE
    coordinator.discard_candidate(initial_target.attempt)
    initial_success = coordinator.begin_candidate("source", spec, None, success_policy)
    assert initial_success.result_type == CHANGE_SUCCESS
    coordinator.discard_candidate(initial_success.attempt)

    refreshed = coordinator.begin_candidate(
        "source", spec, None, target_policy, bypass_cache=True
    )
    assert refreshed.result_type == CHANGE_SUCCESS
    coordinator.discard_candidate(refreshed.attempt)
    rematched = coordinator.begin_candidate("source", spec, None, success_policy)
    assert rematched.result_type == CHANGE_SUCCESS
    coordinator.discard_candidate(rematched.attempt)
    assert len(evaluator.begun) == 3


def test_resource_forced_fresh_observation_recomputes_decision():
    target = _evaluation(
        'File "x.v", line 1, characters 0-1:\nError:\ntarget\n', 1
    )
    other = _evaluation("other", 1)
    evaluator = RecordingEvaluator([target, other])
    coordinator = CandidateCheckCoordinator(evaluator)
    spec = _spec(multiplier=2.0)
    policy = LegacyTargetPolicy(False, "target")
    first = coordinator.begin_candidate("source", spec, None, policy)
    assert first.result_type == CHANGE_SUCCESS
    coordinator.discard_candidate(first.attempt)
    second = coordinator.begin_candidate("source", spec, None, policy)
    assert second.result_type == CHANGE_FAILURE
    coordinator.discard_candidate(second.attempt)
    assert len(evaluator.begun) == 2


def test_relative_path_fingerprint_tracks_cwd_executable_and_checker(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    compiler = bindir / "coqc"
    checker = bindir / "coqchk"
    for path, contents in ((compiler, "one"), (checker, "check-one")):
        path.write_text(contents)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    environment = EnvironmentSnapshot({"PATH": "bin"})
    request = ResourceRequest(None, memory_usage_key=("coqc",))
    spec = EvaluationContextSpec(
        ("coqc",),
        cwd=str(tmp_path),
        environment=environment,
        checker_executable=("coqchk",),
        resource_request=request,
    )
    evaluator = CoqcEvaluator(lambda *args, **kwargs: None)
    first = evaluator.materialize_context(spec)
    assert first.executable_identity[0] == str(compiler)
    assert first.checker_executable_identity[0] == str(checker)
    compiler.write_text("compiler replacement with a different size")
    checker.write_text("checker replacement with a different size")
    second = evaluator.materialize_context(spec)
    assert first != second
    assert first.executable_identity != second.executable_identity
    assert first.checker_executable_identity != second.checker_executable_identity

    cwd_compiler = tmp_path / "cwd-coqc"
    cwd_compiler.write_text("cwd executable")
    cwd_compiler.chmod(cwd_compiler.stat().st_mode | stat.S_IEXEC)
    empty_path_context = evaluator.materialize_context(
        EvaluationContextSpec(
            ("cwd-coqc",),
            cwd=str(tmp_path),
            environment={"PATH": ""},
            resource_request=ResourceRequest(
                None, memory_usage_key=("cwd-coqc",)
            ),
        )
    )
    assert empty_path_context.executable_identity[0] == str(cwd_compiler)


def test_resource_contracts_defensively_normalize_nested_mutable_inputs():
    request_data = {
        "requested_timeout": None,
        "cgroup": ["group"],
        "memory_usage_key": ["coqc"],
    }
    source_arguments = [["-q"]]
    spec = EvaluationContextSpec(
        ("coqc",),
        source_arguments,
        environment={"PATH": ""},
        resource_request=request_data,
    )
    spec_hash = hash(spec)
    source_arguments[0].append("changed")
    request_data["cgroup"].append("changed")
    request_data["memory_usage_key"].append("changed")
    assert spec.arguments == (("-q",),)
    assert spec.resource_request.cgroup == ("group",)
    assert spec.resource_request.memory_usage_key == ("coqc",)
    assert hash(spec) == spec_hash

    policy_data = {
        "request": spec.resource_request,
        "timeout_policy": {
            "requested_timeout": None,
            "effective_timeout": None,
            "should_calibrate_timeout": False,
        },
        "memory_plan": {
            "request": spec.resource_request,
            "initial_usage_rss": [1],
            "initial_usage_as": [2],
            "compiler_max_mem_rss": None,
            "compiler_max_mem_as": None,
            "checker_resolution_mode": ["fixed"],
        },
    }
    context = EvaluationContext(
        "fake",
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
        policy_data,
    )
    context_hash = hash(context)
    policy_data["memory_plan"]["initial_usage_rss"].append(3)
    assert context.resource_policy.memory_plan.initial_usage_rss == (1,)
    assert hash(context) == context_hash


def test_explicit_none_memory_key_ignores_global_none_baselines():
    diagnose_error.reset_memory_usage()
    diagnose_error.set_memory_usage((None, "rss"), 100)
    diagnose_error.set_memory_usage((None, "as"), 200)
    evaluator = CoqcEvaluator(lambda *args, **kwargs: None)
    context = evaluator.materialize_context(
        EvaluationContextSpec(
            ("coqc",),
            environment={"PATH": ""},
            resource_request=ResourceRequest(
                None,
                max_mem_rss_multiplier=2.0,
                max_mem_as_multiplier=3.0,
                memory_usage_key=None,
            ),
        )
    )
    assert context.resource_policy.memory_plan.initial_usage_rss is None
    assert context.resource_policy.memory_plan.initial_usage_as is None
    assert context.resource_policy.memory_plan.compiler_max_mem_rss is None
    assert context.resource_policy.memory_plan.compiler_max_mem_as is None


def test_finalization_failure_resolves_other_attempts_before_recovery_close():
    class StrictEvaluator(RecordingEvaluator):
        def finish(self, trial, accepted):
            if self.closed:
                raise AssertionError("finish called after close")
            return super(StrictEvaluator, self).finish(trial, accepted)

    evaluator = StrictEvaluator(
        [_evaluation("ok"), _evaluation("ok")], fail_accept_at=0
    )
    coordinator = CandidateCheckCoordinator(evaluator)
    policy = LegacyTargetPolicy(True, None)
    first = coordinator.begin_candidate("first", _spec(), None, policy)
    second = coordinator.begin_candidate("second", _spec(), None, policy)
    with pytest.raises(CandidateFinalizationError):
        coordinator.commit_candidate(first.attempt, "serialized", "out.v")
    assert evaluator.finished == [(0, True), (1, False)]
    assert second.attempt.resolved
    assert evaluator.closed
    with pytest.raises(CandidateCoordinatorUnhealthy):
        coordinator.discard_candidate(second.attempt)


def test_candidate_debug_probe_uses_frozen_relative_path_and_cwd(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    compiler = bindir / "coqc"
    compiler.write_text(
        "#!/bin/sh\n"
        "case \" $* \" in\n"
        "  *\" -d native-compiler \"*) echo final ;;\n"
        "  *) echo 'is not a compiled interface for this version of OCaml' ;;\n"
        "esac\n"
    )
    compiler.chmod(compiler.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(
        diagnose_error, "get_filepath_of_coq_args", lambda *args, **kwargs: (None, None)
    )
    evaluator = CoqcEvaluator(lambda *args, **kwargs: None)
    spec = EvaluationContextSpec(
        ("coqc",),
        cwd=str(tmp_path),
        environment={"PATH": "bin"},
        resource_request=ResourceRequest(None, memory_usage_key=("coqc",)),
    )
    trial = evaluator.begin(evaluator.materialize_context(spec), "Check nat.")
    assert trial.evaluation.output.strip() == "final"
    assert trial.evaluation.status == EvaluationStatus.SUCCESS
    assert not list(tmp_path.glob("*.v"))


def test_strict_hybrid_policy_requires_status_and_zero_returncode():
    policy = StrictHybridTargetPolicy(True, None)
    assert policy.primary_preserves(_evaluation("", 0))
    assert not policy.primary_preserves(_evaluation("", 1))
    assert not policy.primary_preserves(
        Evaluation(EvaluationStatus.TIMEOUT, "Timeout!", (), 0)
    )
    target = StrictHybridTargetPolicy(False, "target")
    assert target.primary_preserves(
        Evaluation(EvaluationStatus.COMMAND_ERROR, "Error: target", (), 1)
    )
    assert not target.primary_preserves(
        Evaluation(EvaluationStatus.TIMEOUT, "Error: target", (), 1)
    )


def test_target_policy_identity_participates_in_observation_cache():
    evaluator = RecordingEvaluator([_evaluation("ok"), _evaluation("ok")])
    coordinator = CandidateCheckCoordinator(evaluator)
    spec = _spec()
    first = coordinator.begin_candidate(
        "source", spec, None, LegacyTargetPolicy(True, None)
    )
    coordinator.discard_candidate(first.attempt)
    second = coordinator.begin_candidate(
        "source", spec, None, StrictHybridTargetPolicy(True, None)
    )
    coordinator.discard_candidate(second.attempt)
    assert len(evaluator.begun) == 2


def test_acceptance_can_invalidate_baseline_dependent_caches():
    class BaselineEvaluator(RecordingEvaluator):
        @property
        def invalidates_cache_after_accept(self):
            return True

    evaluator = BaselineEvaluator([_evaluation("ok"), _evaluation("ok")])
    coordinator = CandidateCheckCoordinator(evaluator)
    policy = LegacyTargetPolicy(True, None)
    first = coordinator.begin_candidate("source", _spec(), None, policy)
    coordinator.commit_candidate(first.attempt, "serialized", "out.v")
    assert coordinator.observation_cache == {}
    second = coordinator.begin_candidate("source", _spec(), None, policy)
    coordinator.discard_candidate(second.attempt)
    assert len(evaluator.begun) == 2


def test_explicit_acceptance_authorization_blocks_document_only_success():
    class RejectingAuthority(RecordingEvaluator):
        def acceptance_authorized(self, trial, target_policy, role):
            return False

    evaluator = RejectingAuthority([_evaluation("ok")])
    coordinator = CandidateCheckCoordinator(evaluator)
    with pytest.raises(CandidateEvaluatorError, match="reference oracle"):
        coordinator.begin_candidate(
            "source", _spec(), None, LegacyTargetPolicy(True, None)
        )
    assert evaluator.finished == [(0, False)]
