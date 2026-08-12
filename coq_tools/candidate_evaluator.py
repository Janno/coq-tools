"""Candidate execution contracts for the bug minimizer.

This module deliberately contains no minimizer transformations.  It provides a
small immutable execution model, the compiler-backed adapter, and the
run-scoped coordinator which owns candidate memoization and trial lifecycles.
The implementation is compatible with Python 3.6 and therefore does not use
``dataclasses``.
"""
from __future__ import print_function

import hashlib
import math
import os
import shutil
from collections import namedtuple

from . import diagnose_error


CONTENTS_UNCHANGED = "contents_unchanged"
CHANGE_SUCCESS = "change_success"
CHANGE_FAILURE = "change_failure"


class CandidateEvaluatorError(Exception):
    """Base class for candidate-evaluator infrastructure failures."""


class CandidateLifecycleError(CandidateEvaluatorError):
    """Raised when a trial is resolved more than once or after closure."""


class CandidateFinalizationError(CandidateEvaluatorError):
    """Raised when accepted evaluator state cannot be finalized."""


class CandidateCoordinatorUnhealthy(CandidateEvaluatorError):
    """Raised when work is attempted after an accepted-finalization failure."""


class _TupleValue(tuple):
    """Small named immutable value base with a redaction-friendly repr hook."""

    __slots__ = ()
    _fields = ()

    def __new__(cls, *values, **named_values):
        if len(values) > len(cls._fields):
            raise TypeError(
                "%s takes %d arguments (%d given)"
                % (cls.__name__, len(cls._fields), len(values))
            )
        normalized = list(values)
        for field in cls._fields[len(values) :]:
            if field not in named_values:
                raise TypeError("Missing required argument %r" % field)
            normalized.append(named_values.pop(field))
        if named_values:
            raise TypeError(
                "Unexpected argument%s: %s"
                % (
                    "s" if len(named_values) != 1 else "",
                    ", ".join(sorted(named_values)),
                )
            )
        return tuple.__new__(cls, tuple(normalized))

    def __getattr__(self, name):
        try:
            return self[self._fields.index(name)]
        except ValueError:
            raise AttributeError(name)

    def __repr__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join(
                "%s=%r" % (field, value)
                for field, value in zip(self._fields, tuple(self))
            ),
        )


def _tuple_value(name, fields):
    return type(name, (_TupleValue,), {"__slots__": (), "_fields": tuple(fields)})


def _freeze(value):
    """Defensively convert common mutable containers to immutable values."""
    if isinstance(value, EnvironmentSnapshot):
        return value
    if isinstance(value, dict):
        return tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(item) for item in value))
    return value


_ResourceRequestBase = _tuple_value(
    "_ResourceRequestBase",
    (
        "requested_timeout",
        "max_mem_rss",
        "max_mem_as",
        "max_mem_rss_multiplier",
        "max_mem_as_multiplier",
        "memory_limit_method",
        "cgroup",
        "memory_usage_key",
    ),
)


class ResourceRequest(_ResourceRequestBase):
    __slots__ = ()

    def __new__(
        cls,
        requested_timeout=None,
        max_mem_rss=None,
        max_mem_as=None,
        max_mem_rss_multiplier=None,
        max_mem_as_multiplier=None,
        memory_limit_method="prlimit",
        cgroup=None,
        memory_usage_key=None,
    ):
        return _ResourceRequestBase.__new__(
            cls,
            _freeze(requested_timeout),
            _freeze(max_mem_rss),
            _freeze(max_mem_as),
            _freeze(max_mem_rss_multiplier),
            _freeze(max_mem_as_multiplier),
            _freeze(memory_limit_method),
            _freeze(cgroup),
            _freeze(memory_usage_key),
        )


def _coerce_value(value, cls, description):
    if isinstance(value, cls):
        return value
    try:
        if isinstance(value, dict):
            return cls(**dict(value))
        return cls(*tuple(value))
    except (TypeError, ValueError):
        raise TypeError("%s must be a %s-compatible immutable value" % (description, cls.__name__))


def _coerce_resource_request(value):
    return _coerce_value(value, ResourceRequest, "resource request")


_TimeoutPolicyBase = _tuple_value(
    "_TimeoutPolicyBase",
    ("requested_timeout", "effective_timeout", "should_calibrate_timeout"),
)


class TimeoutPolicy(_TimeoutPolicyBase):
    __slots__ = ()

    def __new__(cls, requested_timeout, effective_timeout, should_calibrate_timeout):
        return _TimeoutPolicyBase.__new__(
            cls,
            _freeze(requested_timeout),
            _freeze(effective_timeout),
            bool(should_calibrate_timeout),
        )


_MemoryLimitPlanBase = _tuple_value(
    "_MemoryLimitPlanBase",
    (
        "request",
        "initial_usage_rss",
        "initial_usage_as",
        "compiler_max_mem_rss",
        "compiler_max_mem_as",
        "checker_resolution_mode",
    ),
)


class MemoryLimitPlan(_MemoryLimitPlanBase):
    __slots__ = ()

    def __new__(
        cls,
        request,
        initial_usage_rss,
        initial_usage_as,
        compiler_max_mem_rss,
        compiler_max_mem_as,
        checker_resolution_mode,
    ):
        return _MemoryLimitPlanBase.__new__(
            cls,
            _coerce_resource_request(request),
            _freeze(initial_usage_rss),
            _freeze(initial_usage_as),
            _freeze(compiler_max_mem_rss),
            _freeze(compiler_max_mem_as),
            _freeze(checker_resolution_mode),
        )


_ResourcePolicyBase = _tuple_value(
    "_ResourcePolicyBase", ("request", "timeout_policy", "memory_plan")
)


class ResourcePolicy(_ResourcePolicyBase):
    __slots__ = ()

    def __new__(cls, request, timeout_policy, memory_plan):
        request = _coerce_resource_request(request)
        timeout_policy = _coerce_value(timeout_policy, TimeoutPolicy, "timeout policy")
        memory_plan = _coerce_value(memory_plan, MemoryLimitPlan, "memory plan")
        if memory_plan.request != request:
            raise ValueError("MemoryLimitPlan request must match ResourcePolicy request")
        return _ResourcePolicyBase.__new__(cls, request, timeout_policy, memory_plan)


class EnvironmentSnapshot(tuple):
    """Hashable copy of the exact process environment, with a redacted repr."""

    __slots__ = ()

    def __new__(cls, environment=None, ocamlpath=None):
        source = dict(os.environ if environment is None else environment)
        if ocamlpath is not None:
            source["OCAMLPATH"] = ocamlpath
        items = tuple(sorted((str(key), str(value)) for key, value in source.items()))
        return tuple.__new__(cls, items)

    @property
    def items(self):
        return tuple(self)

    @property
    def entry_count(self):
        return len(self)

    @property
    def digest(self):
        digest = hashlib.sha256()
        for key, value in self:
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def as_dict(self):
        return dict(self)

    def __repr__(self):
        return "EnvironmentSnapshot(digest=%r, entry_count=%d)" % (
            self.digest,
            self.entry_count,
        )


def _normalize_cwd(cwd):
    return os.path.abspath(os.getcwd() if cwd is None else cwd)


def _normalize_logical_file(logical_file, cwd=None):
    """Normalize display identity independently from the execution cwd."""
    if logical_file is None:
        return None
    return os.path.abspath(logical_file)


def _derive_top_name(arguments, logical_file):
    arguments = tuple(arguments)
    for index, argument in enumerate(arguments[:-1]):
        if argument == "-top":
            return arguments[index + 1]
    if logical_file:
        return os.path.splitext(os.path.basename(logical_file))[0]
    return None


def _resolved_search_path(environment, cwd):
    path = environment.as_dict().get("PATH")
    if path is None:
        return None
    return os.pathsep.join(
        entry if os.path.isabs(entry) else os.path.abspath(os.path.join(cwd, entry or os.curdir))
        for entry in path.split(os.pathsep)
    )


def _resolve_executable_fingerprint(executable, environment, cwd):
    command = executable[0] if executable else ""
    if os.path.isabs(command):
        resolved = command
    elif os.path.dirname(command):
        resolved = os.path.abspath(os.path.join(cwd, command))
    else:
        resolved = shutil.which(command, path=_resolved_search_path(environment, cwd))
    if resolved is None:
        return (None, command)
    resolved = os.path.abspath(resolved)
    try:
        stat_result = os.stat(resolved)
    except OSError:
        return (resolved, None)
    mtime_ns = getattr(
        stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1000000000)
    )
    return (
        resolved,
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        mtime_ns,
    )


_EvaluationContextSpecBase = _tuple_value(
    "_EvaluationContextSpecBase",
    (
        "executable",
        "arguments",
        "cwd",
        "environment",
        "logical_file",
        "top_name",
        "is_toplevel",
        "pass_on_stdin",
        "checker_executable",
        "checker_arguments",
        "resource_request",
        "role",
    ),
)


class EvaluationContextSpec(_EvaluationContextSpecBase):
    __slots__ = ()

    def __new__(
        cls,
        executable,
        arguments=(),
        cwd=None,
        environment=None,
        logical_file=None,
        top_name=None,
        is_toplevel=False,
        pass_on_stdin=False,
        checker_executable=None,
        checker_arguments=(),
        resource_request=None,
        role="primary",
    ):
        normalized_cwd = _normalize_cwd(cwd)
        executable = tuple(_freeze(item) for item in executable)
        arguments = tuple(_freeze(item) for item in arguments)
        if not isinstance(environment, EnvironmentSnapshot):
            environment = EnvironmentSnapshot(environment)
        logical_file = _normalize_logical_file(logical_file, normalized_cwd)
        checker_executable = (
            None
            if checker_executable is None
            else tuple(_freeze(item) for item in checker_executable)
        )
        checker_arguments = tuple(_freeze(item) for item in checker_arguments)
        if resource_request is None:
            resource_request = ResourceRequest(memory_usage_key=executable)
        else:
            resource_request = _coerce_resource_request(resource_request)
        if top_name is None:
            top_name = _derive_top_name(arguments, logical_file)
        if role not in ("primary", "passing"):
            raise ValueError("Unknown evaluation context role %r" % (role,))
        return _EvaluationContextSpecBase.__new__(
            cls,
            executable,
            arguments,
            normalized_cwd,
            environment,
            logical_file,
            top_name,
            bool(is_toplevel),
            bool(pass_on_stdin),
            checker_executable,
            checker_arguments,
            resource_request,
            _freeze(role),
        )


_EvaluationContextBase = _tuple_value(
    "_EvaluationContextBase",
    (
        "evaluator_identity",
        "executable",
        "executable_identity",
        "arguments",
        "cwd",
        "environment",
        "logical_file",
        "top_name",
        "is_toplevel",
        "pass_on_stdin",
        "checker_executable",
        "checker_executable_identity",
        "checker_arguments",
        "resource_policy",
        "role",
    ),
)


class EvaluationContext(_EvaluationContextBase):
    __slots__ = ()

    def __new__(
        cls,
        evaluator_identity,
        executable,
        arguments,
        cwd,
        environment,
        logical_file,
        top_name,
        is_toplevel,
        pass_on_stdin,
        checker_executable,
        checker_arguments,
        resource_policy,
        executable_identity=None,
        checker_executable_identity=None,
        role="primary",
    ):
        executable = tuple(_freeze(item) for item in executable)
        arguments = tuple(_freeze(item) for item in arguments)
        if not isinstance(environment, EnvironmentSnapshot):
            environment = EnvironmentSnapshot(environment)
        cwd = _normalize_cwd(cwd)
        logical_file = _normalize_logical_file(logical_file, cwd)
        if executable_identity is None:
            executable_identity = _resolve_executable_fingerprint(
                executable, environment, cwd
            )
        resource_policy = _coerce_value(
            resource_policy, ResourcePolicy, "resource policy"
        )
        normalized_checker = (
            None
            if checker_executable is None
            else tuple(_freeze(item) for item in checker_executable)
        )
        if role not in ("primary", "passing"):
            raise ValueError("Unknown evaluation context role %r" % (role,))
        if (
            normalized_checker is not None
            and checker_executable_identity is None
        ):
            checker_executable_identity = _resolve_executable_fingerprint(
                normalized_checker, environment, cwd
            )
        return _EvaluationContextBase.__new__(
            cls,
            _freeze(evaluator_identity),
            executable,
            _freeze(executable_identity),
            arguments,
            cwd,
            environment,
            logical_file,
            top_name,
            bool(is_toplevel),
            bool(pass_on_stdin),
            normalized_checker,
            _freeze(checker_executable_identity),
            tuple(_freeze(item) for item in checker_arguments),
            resource_policy,
            _freeze(role),
        )


class EvaluationStatus(object):
    SUCCESS = "success"
    COMMAND_ERROR = "command_error"
    PARSE_ERROR = "parse_error"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    CRASH = "crash"
    INTERNAL_ERROR = "internal_error"

    ALL = (
        SUCCESS,
        COMMAND_ERROR,
        PARSE_ERROR,
        TIMEOUT,
        OUT_OF_MEMORY,
        CRASH,
        INTERNAL_ERROR,
    )


_EvaluationBase = _tuple_value(
    "_EvaluationBase",
    (
        "status",
        "output",
        "commands",
        "returncode",
        "runtime",
        "peak_rss_kb",
        "diagnostics",
        "details",
    ),
)


class Evaluation(_EvaluationBase):
    __slots__ = ()

    def __new__(
        cls,
        status,
        output,
        commands=(),
        returncode=None,
        runtime=None,
        peak_rss_kb=None,
        diagnostics=(),
        details=(),
    ):
        if status not in EvaluationStatus.ALL:
            raise ValueError("Unknown evaluation status %r" % (status,))
        return _EvaluationBase.__new__(
            cls,
            status,
            output,
            tuple(_freeze(item) for item in commands),
            returncode,
            runtime,
            peak_rss_kb,
            tuple(_freeze(item) for item in diagnostics),
            _freeze(details),
        )

    def as_legacy_tuple(self):
        return (
            self.output,
            self.commands,
            self.returncode,
            self.runtime,
            self.peak_rss_kb,
        )


EvaluationTrial = _tuple_value(
    "EvaluationTrial", ("evaluation", "token", "promotable", "from_cache")
)


_TargetPolicyBase = _tuple_value(
    "_TargetPolicyBase", ("should_succeed", "error_reg_string")
)


class TargetPolicy(_TargetPolicyBase):
    """The intentionally permissive legacy text-based target policy."""

    __slots__ = ()

    def __new__(cls, should_succeed=False, error_reg_string=None):
        return _TargetPolicyBase.__new__(
            cls, bool(should_succeed), error_reg_string
        )

    @property
    def identity(self):
        return (type(self).__name__,) + tuple(self)

    def primary_preserves(self, evaluation):
        if self.should_succeed:
            return not diagnose_error.has_error(evaluation.output)
        return diagnose_error.has_error(evaluation.output, self.error_reg_string)

    def passing_succeeds(self, evaluation):
        return not (
            diagnose_error.has_error(evaluation.output)
            or diagnose_error.is_timeout(evaluation.output)
        )


LegacyTargetPolicy = TargetPolicy


class StrictHybridTargetPolicy(TargetPolicy):
    """Fail-closed target policy for decision-capable hybrid execution."""

    __slots__ = ()

    def primary_preserves(self, evaluation):
        if self.should_succeed:
            return self.passing_succeeds(evaluation)
        if evaluation.status != EvaluationStatus.COMMAND_ERROR:
            return False
        return diagnose_error.has_error(evaluation.output, self.error_reg_string)

    def passing_succeeds(self, evaluation):
        return (
            evaluation.status == EvaluationStatus.SUCCESS
            and evaluation.returncode == 0
            and not diagnose_error.has_error(evaluation.output)
            and not diagnose_error.is_timeout(evaluation.output)
            and not diagnose_error.is_memory_limit(evaluation.output)
        )


class CandidateAttempt(object):
    """One unresolved candidate transaction.

    Observation data is immutable.  Only the private resolution guard changes.
    """

    __slots__ = (
        "_evaluated_source",
        "_primary_context",
        "_primary_trial",
        "_primary_evaluation",
        "_passing_context",
        "_passing_trial",
        "_passing_evaluation",
        "_target_policy",
        "_cache_provenance",
        "_resolved",
    )

    def __init__(
        self,
        evaluated_source,
        primary_context,
        primary_trial,
        primary_evaluation,
        passing_context=None,
        passing_trial=None,
        passing_evaluation=None,
        target_policy=None,
        cache_provenance=(),
    ):
        object.__setattr__(self, "_evaluated_source", evaluated_source)
        object.__setattr__(self, "_primary_context", primary_context)
        object.__setattr__(self, "_primary_trial", primary_trial)
        object.__setattr__(self, "_primary_evaluation", primary_evaluation)
        object.__setattr__(self, "_passing_context", passing_context)
        object.__setattr__(self, "_passing_trial", passing_trial)
        object.__setattr__(self, "_passing_evaluation", passing_evaluation)
        object.__setattr__(self, "_target_policy", target_policy)
        object.__setattr__(self, "_cache_provenance", _freeze(cache_provenance))
        object.__setattr__(self, "_resolved", False)

    def __setattr__(self, name, value):
        raise AttributeError("CandidateAttempt fields are read-only")

    evaluated_source = property(lambda self: self._evaluated_source)
    primary_context = property(lambda self: self._primary_context)
    primary_trial = property(lambda self: self._primary_trial)
    primary_evaluation = property(lambda self: self._primary_evaluation)
    passing_context = property(lambda self: self._passing_context)
    passing_trial = property(lambda self: self._passing_trial)
    passing_evaluation = property(lambda self: self._passing_evaluation)
    target_policy = property(lambda self: self._target_policy)
    cache_provenance = property(lambda self: self._cache_provenance)
    resolved = property(lambda self: self._resolved)

    def _mark_resolved(self):
        if self._resolved:
            raise CandidateLifecycleError("Candidate attempt was already resolved")
        object.__setattr__(self, "_resolved", True)


EvaluationVerdict = _tuple_value(
    "EvaluationVerdict",
    (
        "result_type",
        "evaluations",
        "bad_output_index",
        "description",
        "runtime",
        "peak_rss_kb",
        "verbose_descriptions",
        "attempt",
    ),
)


_CandidateDecisionBase = _tuple_value(
    "_CandidateDecisionBase", ("evaluated_source", "serialized_contents", "verdict")
)


class CandidateDecision(_CandidateDecisionBase):
    __slots__ = ()

    result_type = property(lambda self: self.verdict.result_type)
    evaluations = property(lambda self: self.verdict.evaluations)
    outputs = property(lambda self: tuple(item.output for item in self.evaluations))
    bad_output_index = property(lambda self: self.verdict.bad_output_index)
    description = property(lambda self: self.verdict.description)
    runtime = property(lambda self: self.verdict.runtime)
    peak_rss_kb = property(lambda self: self.verdict.peak_rss_kb)
    verbose_descriptions = property(
        lambda self: self.verdict.verbose_descriptions
    )
    attempt = property(lambda self: self.verdict.attempt)

    def as_legacy_tuple(self):
        return (
            self.result_type,
            self.serialized_contents,
            self.outputs,
            self.bad_output_index,
            self.description,
            self.runtime,
            list(self.verbose_descriptions),
        )


CandidateCheckpoint = _tuple_value(
    "CandidateCheckpoint",
    (
        "raw_source",
        "serialized_contents",
        "output_file_name",
        "primary_context",
        "passing_context",
        "target_policy",
    ),
)


class CandidateEvaluator(object):
    """Lifecycle-oriented candidate execution interface."""

    @property
    def identity(self):
        raise NotImplementedError

    @property
    def requires_materialization_for_accept(self):
        return False

    @property
    def invalidates_cache_after_accept(self):
        return False

    def materialize_context(self, spec):
        raise NotImplementedError

    def begin(self, context, source, target_policy=None):
        raise NotImplementedError

    def acceptance_authorized(self, trial, target_policy, role):
        """Return whether this exact live trial may authorize acceptance."""
        if role == "passing":
            return target_policy.passing_succeeds(trial.evaluation)
        return target_policy.primary_preserves(trial.evaluation)

    def finish(self, trial, accepted):
        raise NotImplementedError

    def reset_calibration(self, context=None):
        raise NotImplementedError

    def record_target_decision(self, trial, target_policy, role):
        """Optional reporting hook; it must not influence the decision."""
        return None

    def close(self):
        raise NotImplementedError


def _resolve_limit(fixed_limit, multiplier, usage):
    if multiplier is None or multiplier <= 0:
        return fixed_limit
    if usage is None:
        return None
    resolved = int(math.ceil(usage * multiplier))
    return resolved if resolved > 0 else None


def _timeout_policy(requested_timeout, calibrated_timeout):
    if requested_timeout is None:
        return TimeoutPolicy(requested_timeout, None, False)
    should_calibrate = calibrated_timeout is None
    if requested_timeout < 0 and calibrated_timeout is not None:
        effective_timeout = calibrated_timeout
    elif requested_timeout > 0:
        effective_timeout = requested_timeout
    else:
        effective_timeout = None
    return TimeoutPolicy(requested_timeout, effective_timeout, should_calibrate)


class CoqcEvaluator(CandidateEvaluator):
    """Adapter from immutable candidate contexts to the legacy Coq executor."""

    def __init__(self, log, verbose_base=2):
        self._log = log
        self._verbose_base = verbose_base
        self._closed = False
        self._identity = ("coqc-evaluator", 1)

    @property
    def identity(self):
        return self._identity

    @property
    def requires_materialization_for_accept(self):
        return False

    def _ensure_open(self):
        if self._closed:
            raise CandidateLifecycleError("Evaluator is closed")

    def materialize_context(self, spec):
        self._ensure_open()
        request = spec.resource_request
        calibrated_timeout = diagnose_error.get_timeout_snapshot(spec.executable)
        timeout_policy = _timeout_policy(
            request.requested_timeout, calibrated_timeout
        )
        if request.memory_usage_key is None:
            # An explicit None disables dynamic-memory baselines, matching the
            # legacy process resolver even if unrelated (None, kind) globals exist.
            usage_rss = None
            usage_as = None
        else:
            usage_rss = diagnose_error.get_memory_usage_snapshot(
                (request.memory_usage_key, "rss")
            )
            usage_as = diagnose_error.get_memory_usage_snapshot(
                (request.memory_usage_key, "as")
            )
        compiler_rss = _resolve_limit(
            request.max_mem_rss, request.max_mem_rss_multiplier, usage_rss
        )
        compiler_as = _resolve_limit(
            request.max_mem_as, request.max_mem_as_multiplier, usage_as
        )
        checker_mode = (
            "after_compiler"
            if (
                (request.max_mem_rss_multiplier is not None and request.max_mem_rss_multiplier > 0)
                or (request.max_mem_as_multiplier is not None and request.max_mem_as_multiplier > 0)
            )
            else "fixed"
        )
        memory_plan = MemoryLimitPlan(
            request,
            usage_rss,
            usage_as,
            compiler_rss,
            compiler_as,
            checker_mode,
        )
        policy = ResourcePolicy(request, timeout_policy, memory_plan)
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

    @staticmethod
    def _status(output, returncode):
        if diagnose_error.is_timeout(output):
            return EvaluationStatus.TIMEOUT
        if diagnose_error.is_memory_limit(output):
            return EvaluationStatus.OUT_OF_MEMORY
        if diagnose_error.has_error(output):
            return EvaluationStatus.COMMAND_ERROR
        if returncode == 0:
            return EvaluationStatus.SUCCESS
        return EvaluationStatus.CRASH

    def _run(self, context, source):
        policy = context.resource_policy
        result = diagnose_error._get_coq_output_result(
            context.executable,
            context.arguments,
            source,
            policy.request.requested_timeout,
            cwd=context.cwd,
            is_coqtop=context.is_toplevel,
            pass_on_stdin=context.pass_on_stdin,
            verbose_base=self._verbose_base,
            retry_with_debug_when=(lambda output: False),
            coqchk_prog=context.checker_executable,
            coqchk_prog_args=context.checker_arguments,
            log=self._log,
            use_cache=False,
            process_environment=context.environment.as_dict(),
            log_cwd=(
                None
                if context.cwd == os.path.abspath(os.getcwd())
                else context.cwd
            ),
            effective_timeout=policy.timeout_policy.effective_timeout,
            should_calibrate_timeout=policy.timeout_policy.should_calibrate_timeout,
            memory_plan=policy.memory_plan,
            automatic_debug_retry=False,
        )
        return result

    def begin(self, context, source, target_policy=None):
        self._ensure_open()
        result = self._run(context, source)
        metadata = (("stages", result.stages),)
        final_context = context
        if diagnose_error.default_retry_with_debug_when(result.output):
            debug_args = diagnose_error.get_coq_debug_native_compiler_args(
                context.executable,
                cwd=context.cwd,
                process_environment=context.environment.as_dict(),
                log=self._log,
                executable_identity=context.executable_identity,
            )
            self._log(
                "Retrying with %s..." % " ".join(debug_args),
                level=self._verbose_base - 1,
            )
            retry_spec = EvaluationContextSpec(
                context.executable,
                tuple(debug_args) + context.arguments,
                cwd=context.cwd,
                environment=context.environment,
                logical_file=context.logical_file,
                top_name=context.top_name,
                is_toplevel=context.is_toplevel,
                pass_on_stdin=context.pass_on_stdin,
                checker_executable=context.checker_executable,
                checker_arguments=context.checker_arguments,
                resource_request=context.resource_policy.request,
                role=context.role,
            )
            final_context = self.materialize_context(retry_spec)
            retry_result = self._run(final_context, source)
            metadata = (
                ("preliminary_stages", result.stages),
                ("debug_stages", retry_result.stages),
                ("debug_context", final_context),
            )
            result = retry_result
        evaluation = Evaluation(
            self._status(result.output, result.returncode),
            result.output,
            result.commands,
            result.returncode,
            result.runtime,
            result.peak_rss_kb,
            (),
            metadata,
        )
        return EvaluationTrial(evaluation, None, False, False)

    def finish(self, trial, accepted):
        # The compiler adapter has no promotable state.
        return None

    def reset_calibration(self, context=None):
        self._ensure_open()
        diagnose_error.reset_timeout()

    def close(self):
        self._closed = True


_DecisionSummary = namedtuple(
    "_DecisionSummary",
    "result_type bad_output_index description runtime peak_rss_kb verbose_descriptions",
)


class CandidateCheckCoordinator(object):
    """Run-owned candidate orchestration, memoization, and trial resolution."""

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self._observations = {}
        self._decisions = {}
        self._outstanding = set()
        self._closed = False
        self._healthy = True
        self.last_checkpoint = None

    @property
    def observation_cache(self):
        return dict(self._observations)

    @property
    def decision_cache(self):
        return dict(self._decisions)

    @property
    def healthy(self):
        return self._healthy

    def _ensure_usable(self):
        if self._closed:
            raise CandidateLifecycleError("Candidate coordinator is closed")
        if not self._healthy:
            raise CandidateCoordinatorUnhealthy(
                "Candidate coordinator is unhealthy after finalization failure"
            )

    @staticmethod
    def _requires_resource_execution(context):
        policy = context.resource_policy
        request = policy.request
        return bool(
            policy.timeout_policy.should_calibrate_timeout
            or (
                request.max_mem_rss_multiplier is not None
                and request.max_mem_rss_multiplier > 0
            )
            or (
                request.max_mem_as_multiplier is not None
                and request.max_mem_as_multiplier > 0
            )
        )

    def _invalidate_decisions_for_observation(self, observation_key):
        for decision_key in tuple(self._decisions):
            if decision_key[0] == observation_key or decision_key[1] == observation_key:
                del self._decisions[decision_key]

    def invalidate_observations(self):
        """Drop run-level observations after stateful baseline/context change."""
        self._ensure_usable()
        if self._outstanding:
            raise CandidateLifecycleError(
                "Cannot invalidate observations with unresolved candidates"
            )
        self._observations.clear()
        self._decisions.clear()

    def _observation(self, context, source, target_policy, bypass_cache):
        key = (self.evaluator.identity, context, source, target_policy.identity)
        may_read = not bypass_cache and not self._requires_resource_execution(context)
        if may_read and key in self._observations:
            evaluation = self._observations[key]
            return key, EvaluationTrial(evaluation, None, False, True), True
        trial = self.evaluator.begin(context, source, target_policy)
        if not isinstance(trial, EvaluationTrial):
            raise CandidateEvaluatorError("Evaluator.begin did not return EvaluationTrial")
        if key in self._observations:
            # Replacing a resource-forced or bypassed observation invalidates
            # summaries for every target policy that depended on its old value.
            self._invalidate_decisions_for_observation(key)
        self._observations[key] = trial.evaluation
        return key, trial, False

    def begin_candidate(
        self,
        source,
        primary_spec,
        passing_spec,
        target_policy,
        bypass_cache=False,
        reset_calibration=False,
    ):
        self._ensure_usable()
        if reset_calibration:
            self.evaluator.reset_calibration()
            # Context policies change across calibration reset.  Clearing both
            # layers prevents a later post-calibration context from reviving a
            # stale pre-reset observation with the same immediate deadline.
            self._observations.clear()
            self._decisions.clear()
        primary_context = self.evaluator.materialize_context(primary_spec)
        primary_key, primary_trial, primary_cached = self._observation(
            primary_context, source, target_policy, bypass_cache
        )
        primary_evaluation = primary_trial.evaluation
        passing_context = None
        passing_trial = None
        passing_evaluation = None
        passing_key = None
        passing_cached = False
        try:
            primary_preserves = target_policy.primary_preserves(primary_evaluation)
            if not primary_cached:
                try:
                    self.evaluator.record_target_decision(
                        primary_trial, target_policy, primary_context.role
                    )
                except BaseException:
                    # A reporting-only hook must never become decision authority.
                    pass
            if (
                primary_preserves
                and not target_policy.should_succeed
                and passing_spec is not None
            ):
                passing_context = self.evaluator.materialize_context(passing_spec)
                passing_key, passing_trial, passing_cached = self._observation(
                    passing_context, source, target_policy, bypass_cache
                )
                passing_evaluation = passing_trial.evaluation
                passing_succeeds = target_policy.passing_succeeds(
                    passing_evaluation
                )
                if not passing_cached:
                    try:
                        self.evaluator.record_target_decision(
                            passing_trial, target_policy, passing_context.role
                        )
                    except BaseException:
                        pass
            else:
                passing_succeeds = True
        except BaseException:
            # Preserve the infrastructure/policy exception while resolving all
            # trials that were created before it, newest first.
            created_trials = []
            if primary_trial is not None and not primary_trial.from_cache:
                created_trials.append(primary_trial)
            if passing_trial is not None and not passing_trial.from_cache:
                created_trials.append(passing_trial)
            for created_trial in reversed(created_trials):
                try:
                    self.evaluator.finish(created_trial, False)
                except BaseException:
                    pass
            raise

        evaluations = (
            (primary_evaluation, passing_evaluation)
            if passing_evaluation is not None
            else (primary_evaluation,)
        )
        decision_key = (primary_key, passing_key, target_policy.identity)
        summary = None if bypass_cache else self._decisions.get(decision_key)
        if summary is None:
            if not primary_preserves:
                summary = _DecisionSummary(
                    CHANGE_FAILURE,
                    0,
                    "",
                    primary_evaluation.runtime,
                    primary_evaluation.peak_rss_kb,
                    ((2, "The error was:\n%s\n" % primary_evaluation.output),),
                )
            elif passing_evaluation is not None and not passing_succeeds:
                summary = _DecisionSummary(
                    CHANGE_FAILURE,
                    1,
                    "The alternate coqc (%s) was supposed to pass, but instead emitted an error.  "
                    % passing_context.executable,
                    primary_evaluation.runtime,
                    primary_evaluation.peak_rss_kb,
                    (),
                )
            else:
                selected = (
                    passing_evaluation
                    if passing_evaluation is not None
                    else primary_evaluation
                )
                summary = _DecisionSummary(
                    CHANGE_SUCCESS,
                    None,
                    "Change successful.  ",
                    selected.runtime,
                    selected.peak_rss_kb,
                    (),
                )
            self._decisions[decision_key] = summary

        if summary.result_type == CHANGE_SUCCESS:
            authorization = (
                (primary_trial, primary_context.role),
            ) + (
                ((passing_trial, passing_context.role),)
                if passing_trial is not None
                else ()
            )
            cached_unauthorized = any(
                trial.from_cache for trial, role in authorization
            )
            live_unauthorized = any(
                not trial.from_cache
                and not self.evaluator.acceptance_authorized(
                    trial, target_policy, role
                )
                for trial, role in authorization
            )
        else:
            cached_unauthorized = False
            live_unauthorized = False

        # Stateful evaluators must recreate live accepted state instead of
        # accepting solely from an immutable observation cache.  The explicit
        # authorization check also prevents a rejection-only backend route
        # from ever becoming acceptance authority.
        if summary.result_type == CHANGE_SUCCESS and live_unauthorized:
            for trial in reversed(
                tuple(
                    item
                    for item in (passing_trial, primary_trial)
                    if item is not None and not item.from_cache
                )
            ):
                try:
                    self.evaluator.finish(trial, False)
                except BaseException:
                    pass
            raise CandidateEvaluatorError(
                "Candidate acceptance was not authorized by the reference oracle"
            )
        if (
            summary.result_type == CHANGE_SUCCESS
            and cached_unauthorized
            and self.evaluator.requires_materialization_for_accept
        ):
            if passing_trial is not None and not passing_trial.from_cache:
                self.evaluator.finish(passing_trial, False)
            if not primary_trial.from_cache:
                self.evaluator.finish(primary_trial, False)
            return self.begin_candidate(
                source,
                primary_spec,
                passing_spec,
                target_policy,
                bypass_cache=True,
                reset_calibration=False,
            )

        attempt = CandidateAttempt(
            source,
            primary_context,
            primary_trial,
            primary_evaluation,
            passing_context,
            passing_trial,
            passing_evaluation,
            target_policy,
            (primary_cached, passing_cached),
        )
        self._outstanding.add(attempt)
        verdict = EvaluationVerdict(
            summary.result_type,
            evaluations,
            summary.bad_output_index,
            summary.description,
            summary.runtime,
            summary.peak_rss_kb,
            tuple(summary.verbose_descriptions),
            attempt,
        )
        return verdict

    @staticmethod
    def _live_trials(attempt):
        trials = []
        if attempt.primary_trial is not None and not attempt.primary_trial.from_cache:
            trials.append(attempt.primary_trial)
        if attempt.passing_trial is not None and not attempt.passing_trial.from_cache:
            trials.append(attempt.passing_trial)
        return trials

    def _discard_candidate(self, attempt):
        if attempt is None:
            return
        if attempt.resolved:
            raise CandidateLifecycleError("Candidate attempt was already resolved")
        first_error = None
        for trial in reversed(self._live_trials(attempt)):
            try:
                self.evaluator.finish(trial, False)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        attempt._mark_resolved()
        self._outstanding.discard(attempt)
        if first_error is not None:
            raise first_error

    def discard_candidate(self, attempt):
        self._ensure_usable()
        return self._discard_candidate(attempt)

    def commit_candidate(
        self, attempt, serialized_contents, output_file_name
    ):
        self._ensure_usable()
        if attempt.resolved:
            raise CandidateLifecycleError("Candidate attempt was already resolved")
        self.last_checkpoint = CandidateCheckpoint(
            attempt.evaluated_source,
            serialized_contents,
            output_file_name,
            attempt.primary_context,
            attempt.passing_context,
            attempt.target_policy,
        )
        trials = self._live_trials(attempt)
        for index, trial in enumerate(trials):
            try:
                self.evaluator.finish(trial, True)
            except BaseException as exc:
                # Do not reject the trial whose accepting finalizer threw.  It
                # may already have promoted partial state.  Reject only sibling
                # trials whose accepted finalization was not attempted.
                for remaining in reversed(trials[index + 1 :]):
                    try:
                        self.evaluator.finish(remaining, False)
                    except BaseException:
                        pass
                attempt._mark_resolved()
                self._outstanding.discard(attempt)
                self._healthy = False
                # No other unresolved trial may be sent to an evaluator after
                # its recovery close.  Best-effort reject them first, marking
                # every attempt resolved even if a rejection finalizer fails.
                for other_attempt in tuple(self._outstanding):
                    try:
                        self._discard_candidate(other_attempt)
                    except BaseException:
                        pass
                try:
                    self.evaluator.close()
                except BaseException:
                    pass
                raise CandidateFinalizationError(
                    "Accepted candidate finalization failed"
                ) from exc
        attempt._mark_resolved()
        self._outstanding.discard(attempt)
        if self.evaluator.invalidates_cache_after_accept:
            self._observations.clear()
            self._decisions.clear()

    def close(self):
        if self._closed:
            return
        first_error = None
        for attempt in tuple(self._outstanding):
            try:
                self._discard_candidate(attempt)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._observations.clear()
        self._decisions.clear()
        try:
            self.evaluator.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error
