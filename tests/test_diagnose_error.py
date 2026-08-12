"""Tests for coq_tools.diagnose_error.make_reg_string."""

import os
import re
import tempfile

import pytest

from coq_tools import diagnose_error
from coq_tools.candidate_evaluator import MemoryLimitPlan, ResourceRequest
from coq_tools.diagnose_error import make_reg_string


def _make_output(error_body):
    """Wrap an error body in the standard Coq error output format."""
    return 'File "./test.v", line 3, characters 14-15:\n' + error_body + "\n"


class TestMakeRegStringUniverseInconsistency:
    """Tests for universe inconsistency error regex generation."""

    def test_named_universe_matches_original(self):
        output = _make_output(
            'Error:\nThe term "P" has type "SProp" while it is expected to have type \n'
            '"Type" (universe inconsistency: Cannot enforce SProp <= foo.foo.u0).'
        )
        reg = make_reg_string(output)
        assert re.search(reg, output)

    def test_named_universe_matches_different_qualified_name(self):
        output = _make_output(
            'Error:\nThe term "P" has type "SProp" while it is expected to have type \n'
            '"Type" (universe inconsistency: Cannot enforce SProp <= foo.foo.u0).'
        )
        reg = make_reg_string(output)
        output2 = output.replace("foo.foo.u0", "bar.baz.u1")
        assert re.search(reg, output2)

    def test_alphanumeric_components(self):
        """Handles mixed alphanumeric components like fo5a.xy1b1.u5."""
        output = _make_output(
            'Error:\nThe term "P" has type "SProp" while it is expected to have type \n'
            '"Type" (universe inconsistency: Cannot enforce SProp <= fo5a.xy1b1.u5).'
        )
        reg = make_reg_string(output)
        assert re.search(reg, output)
        output2 = output.replace("fo5a.xy1b1.u5", "ab1c.de2f.u9")
        assert re.search(reg, output2)

    def test_long_qualified_name(self):
        """Handles long multi-component qualified universe names."""
        output = _make_output(
            'Error: Illegal application: \n'
            'The term "args.imported_QC" of type\n "Type -> Type"\n'
            'cannot be applied to the term\n "P" : "SProp"\n'
            "This term has type \"SProp\" which should be a subtype of \n"
            '"Type". (universe inconsistency: Cannot enforce SProp <=\n'
            "Checker.Parametricity.playground.Interface.imported_QC.u0)"
        )
        reg = make_reg_string(output)
        assert re.search(reg, output)
        output2 = output.replace(
            "Checker.Parametricity.playground.Interface.imported_QC.u0",
            "Different.Module.Sub.u0",
        )
        assert re.search(reg, output2)

    def test_scoping_preserves_names_before_inconsistency(self):
        """Qualified names before 'universe inconsistency' are not generalized."""
        output = _make_output(
            'Error:\nThe term "Foo.Bar.baz" has type "SProp" while it is expected to have type \n'
            '"Type" (universe inconsistency: Cannot enforce SProp <= Module.Sub.u0).'
        )
        reg = make_reg_string(output)
        # Foo.Bar.baz should appear literally in the regex (escaped)
        assert "Foo" in reg
        # Module.Sub.u0 after 'universe inconsistency' should be generalized
        assert "Module" not in reg
        assert re.search(reg, output)

    def test_numeric_universe_still_works(self):
        """Existing numeric-suffix universe handling is preserved."""
        output = _make_output(
            'Error:\nThe term "A.foo" has type "Type" while it is expected to have type \n'
            '"Set" (universe inconsistency).'
        )
        reg = make_reg_string(output)
        assert re.search(reg, output)

    def test_because_clause_still_works(self):
        """The 'because' clause truncation still works."""
        output = _make_output(
            'Error:\nThe term "A.foo" has type "Type" while it is expected to have type \n'
            '"Set" (universe inconsistency: Type != Set because blah blah).'
        )
        reg = make_reg_string(output)
        assert re.search(reg, output)
        # 'because' clause should be replaced with .*
        assert "blah" not in reg
        output2 = output.replace("blah blah", "different reason here")
        assert re.search(reg, output2)


@pytest.fixture
def isolated_executor(monkeypatch):
    diagnose_error.COQ_OUTPUT.clear()
    diagnose_error.reset_timeout()
    diagnose_error.reset_memory_usage()
    monkeypatch.setattr(
        diagnose_error, "get_filepath_of_coq_args", lambda *args, **kwargs: (None, None)
    )
    yield
    diagnose_error.COQ_OUTPUT.clear()
    diagnose_error.reset_timeout()
    diagnose_error.reset_memory_usage()


def _executor_kwargs():
    return {
        "log": lambda *args, **kwargs: None,
        "is_coqtop": False,
        "pass_on_stdin": False,
        "verbose_base": 1,
    }


def test_public_result_and_cache_entry_shapes_are_unchanged(
    isolated_executor, monkeypatch
):
    calls = []

    def fake_process(log, command, **kwargs):
        calls.append((tuple(command), kwargs))
        return (("stdout", ""), 0, 1024)

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", fake_process
    )
    first = diagnose_error.get_coq_output(
        ("coqc",), (), "Check nat.", None, **_executor_kwargs()
    )
    second = diagnose_error.get_coq_output(
        ("coqc",), (), "Check nat.", None, **_executor_kwargs()
    )
    assert first == second
    assert len(first) == 5
    assert len(calls) == 1
    assert len(diagnose_error.COQ_OUTPUT) == 1
    file_name, cached = next(iter(diagnose_error.COQ_OUTPUT.values()))
    assert isinstance(file_name, str)
    assert cached == first
    assert len(cached) == 5


def test_cache_bypass_runs_fresh_does_not_store_and_forwards_environment(
    isolated_executor, monkeypatch
):
    calls = []
    source_names = []

    def fake_process(log, command, **kwargs):
        calls.append(kwargs)
        source_names.append(command[-2])
        assert os.path.exists(command[-2])
        return (("ok", ""), 0, 0)

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", fake_process
    )
    explicit_environment = {"PATH": "/frozen/path", "ONLY_THIS": "yes"}
    for _ in range(2):
        result = diagnose_error.get_coq_output(
            ("coqc",),
            (),
            "Check nat.",
            None,
            use_cache=False,
            process_environment=explicit_environment,
            **_executor_kwargs(),
        )
        assert len(result) == 5
    assert len(calls) == 2
    assert source_names[0] != source_names[1]
    assert calls[0]["env"] == explicit_environment
    assert calls[1]["env"] == explicit_environment
    assert diagnose_error.COQ_OUTPUT == {}
    assert all(not os.path.exists(name) for name in source_names)


def test_existing_source_path_is_compiled_in_place_without_cleanup(
    isolated_executor, monkeypatch, tmp_path
):
    source = tmp_path / "Delivered.v"
    source.write_text("Check nat.\n")
    commands = []

    def fake_process(log, command, **kwargs):
        commands.append(tuple(command))
        assert command[-2] == str(source)
        assert source.read_text() == "Check nat.\n"
        return (("ok", ""), 0, 0)

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", fake_process
    )
    diagnose_error.get_coq_output(
        ("coqc",),
        (),
        "Check nat.\n",
        None,
        source_file_name=str(source),
        use_cache=False,
        **_executor_kwargs(),
    )
    assert commands == [("coqc", str(source), "-q")]
    assert source.exists()


def test_structured_checker_metadata_and_staged_multiplier_resolution(
    isolated_executor, monkeypatch
):
    executable = ("coqc",)
    responses = [("compiler", 0, 100), ("Fatal Error: bad", 1, 300)]
    process_kwargs = []

    def fake_process(log, command, **kwargs):
        process_kwargs.append(kwargs)
        output, returncode, peak = responses.pop(0)
        key = kwargs.get("memory_usage_key")
        if key is not None:
            diagnose_error.set_memory_usage((key, "rss"), peak)
            diagnose_error.set_memory_usage((key, "as"), peak)
        return ((output, ""), returncode, peak)

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", fake_process
    )
    request = ResourceRequest(
        None,
        10,
        20,
        2.0,
        3.0,
        "prlimit",
        None,
        executable,
    )
    plan = MemoryLimitPlan(request, None, None, None, None, "after_compiler")
    result = diagnose_error._get_coq_output_result(
        executable,
        (),
        "Check nat.",
        None,
        coqchk_prog=("coqchk",),
        coqchk_prog_args=("-silent",),
        use_cache=False,
        process_environment={"PATH": ""},
        effective_timeout=None,
        should_calibrate_timeout=False,
        memory_plan=plan,
        automatic_debug_retry=False,
        **_executor_kwargs(),
    )
    assert len(result.stages) == 2
    compiler, checker = result.stages
    assert compiler.role == "compiler"
    assert compiler.max_mem_rss is None
    assert checker.role == "checker"
    assert checker.max_mem_rss == 200
    assert checker.max_mem_as == 300
    assert "Fatal Error: bad" in result.output
    assert result.returncode == 1
    assert result.runtime >= 0
    assert result.peak_rss_kb == 300 / 1024
    assert "max_mem_rss_multiplier" not in process_kwargs[0]
    assert "max_mem_rss_multiplier" not in process_kwargs[1]


def test_executor_cleans_source_on_process_and_normalization_failures(
    isolated_executor, monkeypatch
):
    captured = []

    def failing_process(log, command, **kwargs):
        captured.append(command[-2])
        raise RuntimeError("process failed")

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", failing_process
    )
    with pytest.raises(RuntimeError, match="process failed"):
        diagnose_error.get_coq_output(
            ("coqc",), (), "Check nat.", None, use_cache=False, **_executor_kwargs()
        )
    assert not os.path.exists(captured[-1])

    def successful_process(log, command, **kwargs):
        captured.append(command[-2])
        return (("ok", ""), 0, 0)

    monkeypatch.setattr(
        diagnose_error,
        "memory_robust_timeout_Popen_communicate",
        successful_process,
    )
    monkeypatch.setattr(
        diagnose_error,
        "clean_output",
        lambda output: (_ for _ in ()).throw(RuntimeError("normalize failed")),
    )
    with pytest.raises(RuntimeError, match="normalize failed"):
        diagnose_error.get_coq_output(
            ("coqc",), (), "Check nat.", None, use_cache=False, **_executor_kwargs()
        )
    assert not os.path.exists(captured[-1])


@pytest.mark.parametrize(
    "requested,second_deadline_is_none", ((-1, False), (0, True))
)
def test_public_debug_retry_preserves_negative_and_zero_deadlines(
    isolated_executor, monkeypatch, requested, second_deadline_is_none
):
    deadlines = []
    outputs = [
        "is not a compiled interface for this version of OCaml",
        "final",
    ]

    def fake_process(log, command, **kwargs):
        deadlines.append(kwargs["timeout"])
        return ((outputs.pop(0), ""), 0, 0)

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", fake_process
    )
    monkeypatch.setattr(
        diagnose_error,
        "get_coq_debug_native_compiler_args",
        lambda executable, **kwargs: ["-debug"],
    )
    result = diagnose_error.get_coq_output(
        ("coqc",),
        (),
        "Check nat.",
        requested,
        use_cache=False,
        **_executor_kwargs(),
    )
    assert result[0] == "final"
    assert len(deadlines) == 2
    assert deadlines[0] is None
    assert (deadlines[1] is None) is second_deadline_is_none
    if requested < 0:
        assert deadlines[1] == diagnose_error.get_timeout(("coqc",))


def test_preparation_cleans_source_if_logging_raises(
    isolated_executor, monkeypatch, tmp_path
):
    original_named_temporary_file = tempfile.NamedTemporaryFile

    def local_named_temporary_file(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        return original_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(
        diagnose_error.tempfile, "NamedTemporaryFile", local_named_temporary_file
    )

    def failing_log(message, **kwargs):
        if "Running command" in message:
            raise RuntimeError("log failed")

    kwargs = _executor_kwargs()
    kwargs["log"] = failing_log
    with pytest.raises(RuntimeError, match="log failed"):
        diagnose_error.prepare_cmds_for_coq_output(
            ("coqc",), (), "Check nat.", use_cache=False, **kwargs
        )
    assert list(tmp_path.iterdir()) == []


def test_explicit_environments_are_distinct_legacy_cache_entries(
    isolated_executor, monkeypatch
):
    calls = []

    def fake_process(log, command, **kwargs):
        calls.append(kwargs["env"])
        return (("ok", ""), 0, 0)

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", fake_process
    )
    for environment in ({"PATH": "/one"}, {"PATH": "/two"}):
        diagnose_error.get_coq_output(
            ("coqc",),
            (),
            "Check nat.",
            None,
            process_environment=environment,
            **_executor_kwargs(),
        )
    assert calls == [{"PATH": "/one"}, {"PATH": "/two"}]
    assert len(diagnose_error.COQ_OUTPUT) == 2


def test_explicit_none_memory_key_disables_checker_multiplier_baseline(
    isolated_executor, monkeypatch
):
    diagnose_error.set_memory_usage((None, "rss"), 100)
    diagnose_error.set_memory_usage((None, "as"), 200)
    responses = [("compiler", 0, 300), ("checker", 0, 400)]

    def fake_process(log, command, **kwargs):
        output, returncode, peak = responses.pop(0)
        return ((output, ""), returncode, peak)

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", fake_process
    )
    request = ResourceRequest(
        None,
        max_mem_rss_multiplier=2.0,
        max_mem_as_multiplier=3.0,
        memory_usage_key=None,
    )
    plan = MemoryLimitPlan(request, None, None, None, None, "after_compiler")
    result = diagnose_error._get_coq_output_result(
        ("coqc",),
        (),
        "Check nat.",
        None,
        coqchk_prog=("coqchk",),
        use_cache=False,
        memory_plan=plan,
        automatic_debug_retry=False,
        **_executor_kwargs(),
    )
    assert result.stages[0].max_mem_rss is None
    assert result.stages[1].max_mem_rss is None
    assert result.stages[1].max_mem_as is None


def test_source_write_and_checker_failures_clean_owned_source(
    isolated_executor, monkeypatch, tmp_path
):
    original_named_temporary_file = tempfile.NamedTemporaryFile

    class FailingWriteFile(object):
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.name = wrapped.name

        def write(self, contents):
            raise RuntimeError("source write failed")

        def close(self):
            return self._wrapped.close()

    def failing_named_temporary_file(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        return FailingWriteFile(original_named_temporary_file(*args, **kwargs))

    monkeypatch.setattr(
        diagnose_error.tempfile,
        "NamedTemporaryFile",
        failing_named_temporary_file,
    )
    with pytest.raises(RuntimeError, match="source write failed"):
        diagnose_error.prepare_cmds_for_coq_output(
            ("coqc",), (), "Check nat.", use_cache=False, **_executor_kwargs()
        )
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(
        diagnose_error.tempfile, "NamedTemporaryFile", original_named_temporary_file
    )
    captured = []

    def checker_failure(log, command, **kwargs):
        captured.append(command[-2] if len(captured) == 0 else command[-1])
        if len(captured) == 1:
            return (("compiled", ""), 0, 0)
        raise RuntimeError("checker failed")

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", checker_failure
    )
    with pytest.raises(RuntimeError, match="checker failed"):
        diagnose_error.get_coq_output(
            ("coqc",),
            (),
            "Check nat.",
            None,
            coqchk_prog=("coqchk",),
            use_cache=False,
            **_executor_kwargs(),
        )
    assert not os.path.exists(captured[0])


def test_cleanup_failure_does_not_mask_active_executor_error(
    isolated_executor, monkeypatch
):
    captured = []

    def failing_process(log, command, **kwargs):
        captured.append(command[-2])
        raise RuntimeError("compiler failed")

    def failing_cleaner(file_name):
        raise OSError("cleanup failed")

    monkeypatch.setattr(
        diagnose_error, "memory_robust_timeout_Popen_communicate", failing_process
    )
    monkeypatch.setattr(diagnose_error, "clean_v_file", failing_cleaner)
    try:
        with pytest.raises(RuntimeError, match="compiler failed"):
            diagnose_error.get_coq_output(
                ("coqc",), (), "Check nat.", None, use_cache=False, **_executor_kwargs()
            )
    finally:
        if captured and os.path.exists(captured[0]):
            os.remove(captured[0])


def test_native_debug_probe_uses_context_and_cleans_on_interruption(
    monkeypatch, tmp_path
):
    created = []
    process_events = []
    original_named_temporary_file = tempfile.NamedTemporaryFile

    def local_named_temporary_file(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        result = original_named_temporary_file(*args, **kwargs)
        created.append(result.name)
        return result

    class FailingProcess(object):
        def communicate(self):
            process_events.append("communicate")
            raise KeyboardInterrupt()

        def terminate(self):
            process_events.append("terminate")

        def wait(self, timeout=None):
            process_events.append(("wait", timeout))
            return 0

        def kill(self):
            process_events.append("kill")

    monkeypatch.setattr(
        diagnose_error.tempfile, "NamedTemporaryFile", local_named_temporary_file
    )
    monkeypatch.setattr(
        diagnose_error.subprocess, "Popen", lambda *args, **kwargs: FailingProcess()
    )
    with pytest.raises(KeyboardInterrupt):
        diagnose_error.get_coq_accepts_fine_grained_debug(
            ("context-only-coqc",),
            "unique-debug-kind",
            cwd=str(tmp_path),
            process_environment={"PATH": "relative-bin"},
            log=lambda *args, **kwargs: None,
            executable_identity=("interrupted-probe",),
        )
    assert process_events == ["communicate", "terminate", ("wait", 1)]
    assert created
    assert all(not os.path.exists(path) for path in created)


def test_native_debug_probe_cache_includes_executable_identity(tmp_path):
    executable = tmp_path / "fake-coqc"

    def write_executable(output):
        executable.write_text(
            "#!/bin/sh\nprintf '%s' %s\n" % ("%s", repr(output))
        )
        executable.chmod(0o755)

    environment = dict(os.environ)
    write_executable("")
    first = diagnose_error.get_coq_debug_native_compiler_args(
        (str(executable),),
        cwd=str(tmp_path),
        process_environment=environment,
        executable_identity=("fake-coqc", 1),
    )
    write_executable("Unknown option -d")
    second = diagnose_error.get_coq_debug_native_compiler_args(
        (str(executable),),
        cwd=str(tmp_path),
        process_environment=environment,
        executable_identity=("fake-coqc", 2),
    )
    # A reporting-only callback is deliberately absent from the cache key.
    write_executable("")
    cached_second = diagnose_error.get_coq_debug_native_compiler_args(
        (str(executable),),
        cwd=str(tmp_path),
        process_environment=environment,
        log=lambda *args, **kwargs: None,
        executable_identity=("fake-coqc", 2),
    )
    assert first == ["-d", "native-compiler"]
    assert second == ["-debug"]
    assert cached_second == second


def test_ocaml_minor_heap_oom_is_a_memory_limit():
    output = "Fatal error: Not enough heap memory to reserve minor heaps"
    assert diagnose_error.is_memory_limit(output)

    adjusted = diagnose_error.adjust_error_message_for_selected_errors(
        output, file_name="oom.v", line_number=7, characters="1-2"
    )
    assert 'File "oom.v", line 7, characters 1-2:' in adjusted
    assert "Error:\nFatal error: Not enough heap memory" in adjusted
