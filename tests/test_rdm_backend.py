import io
import json
import os
import shutil
import stat
import sys
from collections import namedtuple
import textwrap

import pytest

from coq_tools import rdm_backend
from coq_tools.candidate_evaluator import (
    CandidateChange,
    EnvironmentSnapshot,
    Evaluation,
    EvaluationStatus,
    EvaluationTrial,
    LegacyTargetPolicy,
    StrictHybridTargetPolicy,
)


def _candidate(source, base="accepted"):
    return CandidateChange.from_sources(base, source)


def _frame(value):
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n"
    return b"Content-Length: %d\r\n\r\n" % len(payload) + payload


class ChunkedBytesIO(object):
    def __init__(self, data, chunk_size):
        self._stream = io.BytesIO(data)
        self._chunk_size = chunk_size

    def readline(self, size=-1):
        return self._stream.readline(size)

    def read(self, size=-1):
        if size < 0:
            size = self._chunk_size
        return self._stream.read(min(size, self._chunk_size))


def _make_fake_server(tmp_path):
    path = tmp_path / "fake_rdm.py"
    path.write_text(
        textwrap.dedent(
            r'''
            import json
            import os
            import sys
            import time

            scenario = sys.argv[1]

            def frame(value):
                payload = json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n"
                return b"Content-Length: %d\r\n\r\n" % len(payload) + payload

            def send(value):
                sys.stdout.buffer.write(frame(value))
                sys.stdout.buffer.flush()

            def receive():
                header = sys.stdin.buffer.readline()
                blank = sys.stdin.buffer.readline()
                if not header.startswith(b"Content-Length: ") or blank != b"\r\n":
                    raise RuntimeError("bad request frame")
                length = int(header[len(b"Content-Length: "):].strip())
                return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))

            if scenario == "exit-startup":
                sys.stderr.write("startup exploded\n")
                sys.stderr.flush()
                sys.exit(17)
            if scenario == "bad-startup":
                send({"jsonrpc": "2.0", "id": 4, "result": None})
                sys.exit(0)
            if scenario == "hang-startup":
                time.sleep(60)

            send({"jsonrpc": "2.0", "method": "boot-note", "params": [1]})
            send({"jsonrpc": "2.0", "method": "ready_seq"})
            request = receive()

            if scenario == "hang-request":
                time.sleep(60)
            elif scenario == "normal":
                send({"jsonrpc": "2.0", "method": "progress", "params": {"step": 1}})
                send({"jsonrpc": "2.0", "id": request["id"], "result": {"ok": "😊"}})
            elif scenario == "recoverable":
                send({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {
                        "code": -32803,
                        "message": "candidate failed",
                        "data": {"nb_processed": 2},
                    },
                })
            elif scenario == "stale-id":
                send({"jsonrpc": "2.0", "id": request["id"] + 1, "result": None})
            elif scenario == "unexpected-error":
                send({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32601, "message": "missing method"},
                })
            elif scenario == "ambiguous-response":
                send({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": None,
                    "error": {"code": -32803, "message": "also failed"},
                })
            elif scenario == "malformed-json":
                sys.stdout.buffer.write(b"Content-Length: 5\r\n\r\n{oops")
                sys.stdout.buffer.flush()
            elif scenario == "exit-request":
                sys.stderr.write("request exploded\n")
                sys.stderr.flush()
                sys.exit(23)
            else:
                raise RuntimeError("unknown scenario")
            '''
        )
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _process(tmp_path, scenario):
    server = _make_fake_server(tmp_path)
    return rdm_backend.JsonRpcProcess((sys.executable, server, scenario))


def test_frame_round_trip_uses_utf8_byte_length_and_split_reads():
    value = {"jsonrpc": "2.0", "result": "😊"}
    encoded = rdm_backend.encode_json_rpc_frame(value)
    header, payload = encoded.split(b"\r\n\r\n", 1)
    assert int(header.split(b":", 1)[1]) == len(payload)
    assert rdm_backend.read_json_rpc_frame(ChunkedBytesIO(encoded, 2)) == value


def test_frame_rejects_bad_header_length_json_and_eof():
    with pytest.raises(rdm_backend.JsonRpcFramingError):
        rdm_backend.read_json_rpc_frame(io.BytesIO(b"Length: 2\r\n\r\n{}"))
    with pytest.raises(rdm_backend.JsonRpcFramingError):
        rdm_backend.read_json_rpc_frame(
            io.BytesIO(b"Content-Length: nope\r\n\r\n{}")
        )
    with pytest.raises(rdm_backend.JsonRpcFramingError):
        rdm_backend.read_json_rpc_frame(
            io.BytesIO(b"Content-Length: 20\r\n\r\n{}")
        )
    with pytest.raises(rdm_backend.JsonRpcProtocolError):
        rdm_backend.read_json_rpc_frame(
            io.BytesIO(b"Content-Length: 5\r\n\r\n{oops")
        )


def test_frame_rejects_oversized_payload():
    encoded = _frame({"large": "x" * 100})
    with pytest.raises(rdm_backend.JsonRpcFramingError):
        rdm_backend.read_json_rpc_frame(io.BytesIO(encoded), max_frame_size=10)


def test_process_accepts_startup_and_interleaved_notifications(tmp_path):
    process = _process(tmp_path, "normal")
    try:
        assert [item.method for item in process.notifications] == ["boot-note"]
        assert process.request("echo", ["😊"]) == {"ok": "😊"}
        assert [item.method for item in process.notifications] == [
            "boot-note",
            "progress",
        ]
    finally:
        process.close()
    assert process.returncode == 0


def test_process_exposes_recoverable_error_data(tmp_path):
    process = _process(tmp_path, "recoverable")
    try:
        with pytest.raises(rdm_backend.JsonRpcRequestError) as exc_info:
            process.request("run_steps", [0, 4])
        assert exc_info.value.code == -32803
        assert exc_info.value.method == "run_steps"
        assert exc_info.value.data == {"nb_processed": 2}
    finally:
        process.close()


def test_process_rejects_stale_response_id_and_becomes_unusable(tmp_path):
    process = _process(tmp_path, "stale-id")
    try:
        with pytest.raises(rdm_backend.JsonRpcProtocolError):
            process.request("echo", [])
        with pytest.raises(rdm_backend.JsonRpcError):
            process.request("echo", [])
    finally:
        process.close()


@pytest.mark.parametrize("scenario", ("unexpected-error", "ambiguous-response"))
def test_process_rejects_unexpected_error_and_ambiguous_response(tmp_path, scenario):
    process = _process(tmp_path, scenario)
    try:
        with pytest.raises(rdm_backend.JsonRpcProtocolError):
            process.request("echo", [])
    finally:
        process.close()


def test_process_reports_malformed_response_and_stderr_exit(tmp_path):
    malformed = _process(tmp_path, "malformed-json")
    try:
        with pytest.raises(rdm_backend.JsonRpcProtocolError):
            malformed.request("echo", [])
    finally:
        malformed.close()

    exited = _process(tmp_path, "exit-request")
    try:
        with pytest.raises(rdm_backend.JsonRpcProcessError) as exc_info:
            exited.request("echo", [])
        assert exc_info.value.returncode == 23
        assert "request exploded" in exc_info.value.stderr_tail
    finally:
        exited.close()


def test_startup_failures_capture_shape_and_stderr(tmp_path):
    server = _make_fake_server(tmp_path)
    with pytest.raises(rdm_backend.JsonRpcProtocolError):
        rdm_backend.JsonRpcProcess((sys.executable, server, "bad-startup"))
    with pytest.raises(rdm_backend.JsonRpcProcessError) as exc_info:
        rdm_backend.JsonRpcProcess((sys.executable, server, "exit-startup"))
    assert exc_info.value.returncode == 17
    assert "startup exploded" in exc_info.value.stderr_tail


def test_startup_and_request_deadlines_reap_process(tmp_path):
    server = _make_fake_server(tmp_path)
    with pytest.raises(rdm_backend.JsonRpcDeadlineExceeded):
        rdm_backend.JsonRpcProcess(
            (sys.executable, server, "hang-startup"), request_timeout=0.05
        )
    process = rdm_backend.JsonRpcProcess(
        (sys.executable, server, "hang-request"), request_timeout=0.05
    )
    with pytest.raises(rdm_backend.JsonRpcDeadlineExceeded):
        process.request("ping", [])
    child = process._process
    process.close()
    assert child.poll() is not None


def test_close_is_idempotent(tmp_path):
    process = _process(tmp_path, "normal")
    process.close()
    process.close()


class RecordingTransport(object):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def request(self, method, params):
        self.calls.append((method, list(params)))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self):
        self.closed = True


def test_document_client_uses_exact_positional_rpc_shapes():
    transport = RecordingTransport(
        [
            None,
            7,
            None,
            None,
            None,
            [],
            [],
            [],
            None,
            "Check I.\n",
            None,
        ]
    )
    client = rdm_backend.RdmClient(transport)
    assert client.load_file(0) is None
    assert client.clone(0) == 7
    assert client.go_to(7, 2) is None
    assert client.revert_before(7, False, 1) is None
    assert client.clear_suffix(7, 2) is None
    assert client.replace_suffix(7, "Check I.\n", None) == ()
    assert client.doc_prefix(7) == ()
    assert client.doc_suffix(7) == ()
    assert client.run_steps(7, 0) is None
    assert client.contents(7, False, True) == "Check I.\n"
    assert client.dispose(7) is None
    assert transport.calls == [
        ("load_file", [0]),
        ("clone", [0]),
        ("go_to", [7, 2]),
        ("revert_before", [7, False, 1]),
        ("clear_suffix", [7, 2]),
        ("replace_suffix", [7, "Check I.\n", None]),
        ("doc_prefix", [7]),
        ("doc_suffix", [7]),
        ("run_steps", [7, 0]),
        ("contents", [7, False, True]),
        ("dispose", [7]),
    ]


def test_document_client_validates_items_and_preserves_data():
    transport = RecordingTransport(
        [
            [
                {
                    "kind": "command",
                    "offset": 0,
                    "text": "Check I.",
                    "data": {"kind": "CheckMayEval", "pure": True},
                },
                {"kind": "blanks", "offset": 8, "text": "\n"},
            ],
            [{"kind": "command", "text": "Check I.", "data": None}],
        ]
    )
    client = rdm_backend.RdmClient(transport)
    prefix = client.doc_prefix(0)
    suffix = client.doc_suffix(0)
    assert prefix[0].kind == "command"
    assert prefix[0].offset == 0
    assert prefix[0].data == (("kind", "CheckMayEval"), ("pure", True))
    assert prefix[1].kind == "blanks"
    assert suffix[0].offset is None


def test_document_client_rejects_malformed_never_fail_results():
    client = rdm_backend.RdmClient(
        RecordingTransport([[{"kind": "invalid", "text": "x"}]])
    )
    with pytest.raises(rdm_backend.JsonRpcProtocolError):
        client.doc_suffix(0)

    client = rdm_backend.RdmClient(RecordingTransport([True]))
    with pytest.raises(rdm_backend.JsonRpcProtocolError):
        client.clone(0)


def test_document_error_parsers_validate_recoverable_payloads():
    error = rdm_backend.JsonRpcRequestError(
        "run_steps",
        3,
        -32803,
        "missing",
        {
            "nb_processed": 2,
            "cmd_error": {
                "error_loc": {"bp": 28, "ep": 32},
                "feedback_messages": [
                    {"level": "error", "text": "missing", "loc": None}
                ],
            },
        },
    )
    parsed = rdm_backend.parse_steps_error(error)
    assert parsed.nb_processed == 2
    assert parsed.command_error.error_loc.bp == 28
    assert parsed.command_error.feedback_messages[0].text == "missing"

    malformed = rdm_backend.JsonRpcRequestError(
        "run_steps", 4, -32803, "bad", {"nb_processed": -1}
    )
    with pytest.raises(rdm_backend.JsonRpcProtocolError):
        rdm_backend.parse_steps_error(malformed)


def test_capability_probe_parses_required_methods(tmp_path):
    server = tmp_path / "manager"
    methods = sorted(rdm_backend.REQUIRED_METHODS)
    server.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --api-docs ]; then\n"
        + "printf '%s\\n' "
        + " ".join("'### `%s`'" % method for method in methods)
        + "\nexit 0\nfi\nexit 2\n"
    )
    server.chmod(server.stat().st_mode | stat.S_IXUSR)
    probe = rdm_backend.probe_rdm((str(server),))
    assert set(probe.methods) == set(methods)
    assert probe.sha256
    assert probe.missing_methods == ()


def test_item_boundary_rolls_edits_inside_commands_back_to_command_start():
    items = (
        rdm_backend.DocumentItem("command", "Definition x := 0.", (), 0),
        rdm_backend.DocumentItem("blanks", "\n", (), 18),
        rdm_backend.DocumentItem("command", "Check x.", (), 19),
        rdm_backend.DocumentItem("blanks", "\n", (), 27),
    )
    old = "".join(item.text for item in items)
    assert rdm_backend.find_item_boundary(items, old, old) == (4, len(old))
    assert rdm_backend.find_item_boundary(
        items, old, old.replace("0", "1")
    ) == (0, 0)
    assert rdm_backend.find_item_boundary(
        items, old, old.replace("Check x.", "Check I.")
    ) == (2, len("Definition x := 0.\n"))


def _item(kind, text):
    return rdm_backend.DocumentItem(kind, text, (), None)


def test_item_splice_plan_uses_counted_clear_for_whole_items():
    base = "Definition a := 0.\n\nDefinition b := 1.\n"
    items = (
        _item("command", "Definition a := 0."),
        _item("blanks", "\n\n"),
        _item("command", "Definition b := 1."),
        _item("blanks", "\n"),
    )
    candidate = CandidateChange.delete(base, 0, len("Definition a := 0.\n\n"))
    plan = rdm_backend.plan_item_splice(items, candidate)
    assert plan.strategy == "clear"
    assert (plan.start_item, plan.end_item) == (0, 2)
    assert plan.replacement == ""


def test_item_splice_plan_expands_partial_command_and_preserves_tail():
    base = "Definition a := 0.\nDefinition b := 1.\nCheck b.\n"
    items = (
        _item("command", "Definition a := 0."),
        _item("blanks", "\n"),
        _item("command", "Definition b := 1."),
        _item("blanks", "\n"),
        _item("command", "Check b."),
        _item("blanks", "\n"),
    )
    candidate = CandidateChange.from_sources(
        base, base.replace("b := 1", "b := 2")
    )
    plan = rdm_backend.plan_item_splice(items, candidate)
    assert plan.strategy == "replace"
    assert (plan.start_item, plan.end_item) == (1, 3)
    assert plan.replacement == "\nDefinition b := 2."
    assert "Check b." not in plan.replacement


def test_item_splice_plan_clamps_edits_after_canonical_error():
    base = "Definition a := 0.\nCheck missing.\nCheck later.\n"
    items = (
        _item("command", "Definition a := 0."),
        _item("blanks", "\n"),
        _item("command", "Check missing."),
        _item("blanks", "\n"),
        _item("command", "Check later."),
        _item("blanks", "\n"),
    )
    candidate = CandidateChange.from_sources(
        base, base.replace("Check later.", "Check I.")
    )
    plan = rdm_backend.plan_item_splice(
        items, candidate, maximum_start_item=2
    )
    assert plan.start_item == 1
    assert plan.end_item == 5
    assert plan.replacement == "\nCheck missing.\nCheck I."


def test_unicode_diagnostic_offsets_are_derived_from_candidate_bytes():
    source = "Definition α := True.\nCheck nope.\n"
    source_bytes = source.encode("utf-8")
    bp = source_bytes.index(b"nope")
    location = rdm_backend.RocqLocation(bp, bp + 4, None, None, None, None, None)
    output = rdm_backend.render_diagnostic(
        "/tmp/candidate.v", source, "missing", location=location
    )
    assert 'File "candidate.v", line 2, characters 6-10:' in output
    assert output.endswith("Error:\nmissing\n")


ShadowDocumentTrial = namedtuple(
    "ShadowDocumentTrial", "observation promotable"
)


def _document_observation(status, output):
    return rdm_backend.RdmObservation(
        status,
        output,
        (),
        0.2,
        "end" if status == "success" else status,
        2,
        3,
        1,
        1,
        0.05,
        0.15,
        "replace",
        "inferred",
        1,
        9,
        1,
        (),
    )


class FakeCompilerEvaluator(object):
    identity = ("fake-compiler",)
    requires_materialization_for_accept = False

    def __init__(self, evaluation, begin_error=None):
        self.evaluation = evaluation
        self.begin_error = begin_error
        self.begun = []
        self.finished = []
        self.closed = False
        self.reset_count = 0

    def materialize_context(self, spec):
        return spec

    def begin(self, context, candidate, target_policy=None):
        assert isinstance(candidate, CandidateChange)
        self.begun.append((context, candidate.source))
        if self.begin_error is not None:
            raise self.begin_error
        return EvaluationTrial(self.evaluation, "compiler", False, False)

    def finish(self, trial, accepted):
        self.finished.append((trial.token, accepted))

    def reset_calibration(self, context=None):
        self.reset_count += 1

    def close(self):
        self.closed = True


class FakeDocumentEvaluator(object):
    identity = ("fake-document",)

    def __init__(
        self, observation=None, reason=None, begin_error=None, finish_error=None
    ):
        self.observation = observation
        self.reason = reason
        self.begin_error = begin_error
        self.finish_error = finish_error
        self.begun = []
        self.finished = []
        self.closed = False
        self.reset_count = 0
        self.disabled = []
        self.advanced = []
        self.accepted_source = "accepted"

    def unsupported_reason(self, context):
        return self.reason

    def begin(self, context, candidate):
        assert isinstance(candidate, CandidateChange)
        self.begun.append((context, candidate.source))
        if self.begin_error is not None:
            raise self.begin_error
        return ShadowDocumentTrial(self.observation, True)

    def finish(self, trial, accepted):
        self.finished.append((trial, accepted))
        if self.finish_error is not None:
            raise self.finish_error

    def disable(self, context, reason):
        self.disabled.append((context, reason))

    def advance_accepted_source(self, source):
        self.advanced.append(source)

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


def _document_only(context_evaluator, document, logs):
    evaluator = object.__new__(rdm_backend.RdmOnlyEvaluator)
    evaluator.context_evaluator = context_evaluator
    evaluator.document_evaluator = document
    evaluator.target_policy = StrictHybridTargetPolicy(False, "TARGET")
    evaluator._log = lambda message, **kwargs: logs.append(message)
    evaluator._jsonl_path = None
    evaluator._closed = False
    return evaluator


def test_document_only_positive_is_authoritative_without_compiler_execution():
    context_evaluator = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.CRASH, "must not run", (), 2)
    )
    document = FakeDocumentEvaluator(
        _document_observation("command_error", "Error:\nTARGET\n")
    )
    logs = []
    evaluator = _document_only(context_evaluator, document, logs)
    Context = namedtuple("Context", "role")
    context = Context("primary")
    policy = StrictHybridTargetPolicy(False, "TARGET")

    trial = evaluator.begin(context, _candidate("candidate"), policy)
    assert trial.evaluation.status == EvaluationStatus.COMMAND_ERROR
    assert trial.promotable
    assert evaluator.acceptance_authorized(trial, policy, "primary")
    assert context_evaluator.begun == []

    evaluator.record_target_decision(trial, policy, "primary")
    assert '"document_verdict":true' in logs[-1]
    evaluator.finish(trial, True)
    assert document.finished[0][1] is True
    assert context_evaluator.finished == []


def test_document_only_rejection_is_not_authorized_or_promoted():
    context_evaluator = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1)
    )
    document = FakeDocumentEvaluator(_document_observation("success", ""))
    evaluator = _document_only(context_evaluator, document, [])
    Context = namedtuple("Context", "role")
    context = Context("primary")
    policy = StrictHybridTargetPolicy(False, "TARGET")

    trial = evaluator.begin(context, _candidate("candidate"), policy)
    assert not trial.promotable
    assert not evaluator.acceptance_authorized(trial, policy, "primary")
    evaluator.finish(trial, False)
    assert document.finished[0][1] is False
    assert context_evaluator.begun == []


def test_document_only_unavailable_context_aborts_without_compiler_fallback():
    context_evaluator = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1)
    )
    document = FakeDocumentEvaluator(reason="unsupported context")
    evaluator = _document_only(context_evaluator, document, [])
    Context = namedtuple("Context", "role")

    with pytest.raises(
        rdm_backend.RdmUnsupported,
        match="rdm-only cannot evaluate.*unsupported context",
    ):
        evaluator.begin(
            Context("primary"),
            _candidate("candidate"),
            StrictHybridTargetPolicy(False, "TARGET"),
        )
    assert not document.begun
    assert not context_evaluator.begun


def test_document_only_transport_failure_does_not_fall_back_to_compiler():
    context_evaluator = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1)
    )
    document = FakeDocumentEvaluator(
        begin_error=rdm_backend.RdmUnavailable("manager exited")
    )
    evaluator = _document_only(context_evaluator, document, [])
    Context = namedtuple("Context", "role")

    with pytest.raises(rdm_backend.RdmUnavailable, match="manager exited"):
        evaluator.begin(
            Context("primary"),
            _candidate("candidate"),
            StrictHybridTargetPolicy(False, "TARGET"),
        )
    assert not context_evaluator.begun


def test_document_only_jsonl_records_document_metrics(tmp_path):
    context_evaluator = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.CRASH, "must not run", (), 2)
    )
    document = FakeDocumentEvaluator(
        _document_observation("command_error", "Error:\nTARGET\n")
    )
    evaluator = _document_only(context_evaluator, document, [])
    path = tmp_path / "document-only.jsonl"
    evaluator._jsonl_path = str(path)
    Context = namedtuple("Context", "role")
    policy = StrictHybridTargetPolicy(False, "TARGET")
    trial = evaluator.begin(Context("primary"), _candidate("candidate"), policy)
    evaluator.record_target_decision(trial, policy, "primary")

    record = json.loads(path.read_text())
    assert record["mode"] == "rdm-only"
    assert record["document_verdict"] is True
    assert record["document_split_runtime"] == 0.05
    assert record["document_execution_runtime"] == 0.15
    assert record["edit_strategy"] == "replace"
    assert "compiler_runtime" not in record
    evaluator.finish(trial, False)


def test_document_only_reset_and_close_cover_both_components():
    context_evaluator = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.SUCCESS, "", (), 0)
    )
    document = FakeDocumentEvaluator(_document_observation("success", ""))
    evaluator = _document_only(context_evaluator, document, [])
    evaluator.reset_calibration()
    assert context_evaluator.reset_count == 1
    assert document.reset_count == 1
    evaluator.close()
    assert context_evaluator.closed
    assert document.closed


def _shadow(compiler, document, logs):
    shadow = object.__new__(rdm_backend.RdmShadowEvaluator)
    shadow.compiler_evaluator = compiler
    shadow.document_evaluator = document
    shadow._log = lambda message, **kwargs: logs.append(message)
    shadow._closed = False
    return shadow


def test_shadow_evaluator_preserves_compiler_authority_and_logs_disagreement():
    compiler_evaluation = Evaluation(
        EvaluationStatus.COMMAND_ERROR,
        'File "x.v", line 1, characters 0-1:\nError:\nTARGET\n',
        ("coqc", "x.v"),
        1,
        1.25,
        42.0,
        details=(("compiler", "metadata"),),
    )
    compiler = FakeCompilerEvaluator(compiler_evaluation)
    document = FakeDocumentEvaluator(_document_observation("success", ""))
    logs = []
    shadow = _shadow(compiler, document, logs)
    Context = namedtuple("Context", "role")
    context = Context("primary")

    trial = shadow.begin(context, _candidate("candidate"))
    assert trial.evaluation.as_legacy_tuple() == compiler_evaluation.as_legacy_tuple()
    assert trial.evaluation.status == EvaluationStatus.COMMAND_ERROR
    assert trial.evaluation.details[0] == ("compiler", "metadata")
    assert trial.evaluation.details[-1][0] == "rdm_shadow"

    policy = LegacyTargetPolicy(False, "TARGET")
    shadow.record_target_decision(trial, policy, "primary")
    assert '"agreement":false' in logs[-1]
    assert '"compiler_verdict":true' in logs[-1]
    assert '"document_verdict":false' in logs[-1]

    shadow.finish(trial, True)
    assert compiler.finished == [("compiler", True)]
    assert document.finished[0][1] is True


def test_shadow_unavailability_never_skips_compiler_or_changes_output():
    compiler_evaluation = Evaluation(
        EvaluationStatus.SUCCESS, "compiler output", ("coqc",), 0, 0.1, 10
    )
    compiler = FakeCompilerEvaluator(compiler_evaluation)
    document = FakeDocumentEvaluator(reason="unsupported context")
    shadow = _shadow(compiler, document, [])
    Context = namedtuple("Context", "role")
    trial = shadow.begin(Context("primary"), _candidate("candidate"))
    assert compiler.begun
    assert not document.begun
    assert trial.evaluation.output == "compiler output"
    shadow.finish(trial, False)
    assert compiler.finished == [("compiler", False)]


def test_shadow_promotion_failure_does_not_undo_compiler_acceptance():
    compiler = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.SUCCESS, "", (), 0, 0.1, 1)
    )
    document = FakeDocumentEvaluator(
        _document_observation("success", ""),
        finish_error=RuntimeError("promotion failed"),
    )
    logs = []
    shadow = _shadow(compiler, document, logs)
    Context = namedtuple("Context", "role")
    trial = shadow.begin(Context("primary"), _candidate("candidate"))
    shadow.finish(trial, True)
    assert compiler.finished == [("compiler", True)]
    assert "state discarded after finalization failure" in logs[-1]


def test_compiler_infrastructure_failure_discards_live_shadow_trial():
    compiler = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.SUCCESS, "", (), 0, 0.1, 1),
        begin_error=RuntimeError("compiler failed"),
    )
    document = FakeDocumentEvaluator(_document_observation("success", ""))
    shadow = _shadow(compiler, document, [])
    Context = namedtuple("Context", "role")
    with pytest.raises(RuntimeError, match="compiler failed"):
        shadow.begin(Context("primary"), _candidate("candidate"))
    assert document.finished[0][1] is False


def _hybrid(compiler, document, logs, check_rejected_every=0):
    hybrid = object.__new__(rdm_backend.RdmHybridEvaluator)
    hybrid.compiler_evaluator = compiler
    hybrid.document_evaluator = document
    hybrid.target_policy = LegacyTargetPolicy(False, "TARGET")
    hybrid._log = lambda message, **kwargs: logs.append(message)
    hybrid._check_rejected_every = check_rejected_every
    hybrid._reject_counts = {}

    class Confirmed(dict):
        def get(self, key, default=None):
            return document.accepted_source

    hybrid._baseline_confirmed = Confirmed()
    hybrid._closed = False
    return hybrid


def test_hybrid_reference_baseline_is_compiler_confirmed_before_fast_reject():
    baseline = Evaluation(
        EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1
    )
    compiler = FakeCompilerEvaluator(baseline)
    document = FakeDocumentEvaluator(_document_observation("success", ""))
    hybrid = _hybrid(compiler, document, [])
    hybrid._baseline_confirmed = {}
    Context = namedtuple("Context", "role")
    context = Context("primary")
    policy = LegacyTargetPolicy(False, "TARGET")
    trial = hybrid.begin(context, _candidate("candidate"), policy)
    assert compiler.begun == [(context, "accepted")]
    assert trial.token.route == "fast_reject"
    hybrid.finish(trial, False)


def test_hybrid_fast_reject_skips_compiler_and_cannot_authorize_acceptance():
    compiler = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1)
    )
    document = FakeDocumentEvaluator(_document_observation("success", ""))
    hybrid = _hybrid(compiler, document, [])
    Context = namedtuple("Context", "role")
    context = Context("primary")
    policy = LegacyTargetPolicy(False, "TARGET")
    trial = hybrid.begin(context, _candidate("candidate"), policy)
    assert trial.token.route == "fast_reject"
    assert not compiler.begun
    assert not hybrid.acceptance_authorized(trial, policy, "primary")
    hybrid.finish(trial, False)
    assert document.finished[0][1] is False


def test_hybrid_document_positive_requires_compiler_confirmation():
    compiler_evaluation = Evaluation(
        EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1
    )
    compiler = FakeCompilerEvaluator(compiler_evaluation)
    document = FakeDocumentEvaluator(
        _document_observation("command_error", "Error: TARGET")
    )
    hybrid = _hybrid(compiler, document, [])
    Context = namedtuple("Context", "role")
    policy = LegacyTargetPolicy(False, "TARGET")
    trial = hybrid.begin(Context("primary"), _candidate("candidate"), policy)
    assert trial.token.route == "reference_confirm"
    assert compiler.begun
    assert hybrid.acceptance_authorized(trial, policy, "primary")
    hybrid.finish(trial, True)
    assert compiler.finished == [("compiler", True)]
    assert document.finished[0][1] is True


def test_hybrid_comparison_jsonl_retains_route_context_and_metrics(tmp_path):
    compiler = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1)
    )
    document = FakeDocumentEvaluator(
        _document_observation("command_error", "Error: TARGET")
    )
    hybrid = _hybrid(compiler, document, [])
    path = tmp_path / "comparisons.jsonl"
    hybrid._jsonl_path = str(path)
    Context = namedtuple("Context", "role")
    policy = LegacyTargetPolicy(False, "TARGET")
    trial = hybrid.begin(Context("primary"), _candidate("candidate"), policy)
    hybrid.record_target_decision(trial, policy, "primary")
    record = json.loads(path.read_text())
    assert record["event"] == "candidate-comparison"
    assert record["mode"] == "rdm-hybrid"
    assert record["context"] == {"role": "primary"}
    assert record["source_sha256"]
    assert record["document_split_runtime"] == 0.05
    assert record["edit_strategy"] == "replace"
    assert record["edit_kind"] == "inferred"
    assert record["replaced_items"] == 1
    hybrid.finish(trial, False)


def test_hybrid_compiler_fallback_acceptance_advances_raw_source():
    compiler = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1)
    )
    document = FakeDocumentEvaluator(reason="unsupported context")
    hybrid = _hybrid(compiler, document, [])
    Context = namedtuple("Context", "role")
    policy = LegacyTargetPolicy(False, "TARGET")
    trial = hybrid.begin(Context("primary"), _candidate("new raw"), policy)
    assert trial.token.route == "compiler_fallback"
    assert hybrid.acceptance_authorized(trial, policy, "primary")
    hybrid.finish(trial, True)
    assert document.advanced == ["new raw"]


def test_hybrid_passing_requires_strict_compiler_success():
    compiler = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.SUCCESS, "", (), 0)
    )
    document = FakeDocumentEvaluator(_document_observation("success", ""))
    hybrid = _hybrid(compiler, document, [])
    Context = namedtuple("Context", "role")
    policy = LegacyTargetPolicy(False, "TARGET")
    trial = hybrid.begin(Context("passing"), _candidate("candidate"), policy)
    assert trial.token.document_verdict is True
    assert trial.token.compiler_verdict is True
    assert hybrid.acceptance_authorized(trial, policy, "passing")
    hybrid.finish(trial, True)


def test_hybrid_audit_detects_false_negative_and_disables_context():
    compiler = FakeCompilerEvaluator(
        Evaluation(EvaluationStatus.COMMAND_ERROR, "Error: TARGET", (), 1)
    )
    document = FakeDocumentEvaluator(_document_observation("success", ""))
    logs = []
    hybrid = _hybrid(compiler, document, logs, check_rejected_every=1)
    Context = namedtuple("Context", "role")
    context = Context("primary")
    policy = LegacyTargetPolicy(False, "TARGET")
    trial = hybrid.begin(context, _candidate("candidate"), policy)
    assert trial.token.route == "audited_reject"
    assert trial.token.reason == "document-false-negative"
    assert document.disabled
    assert hybrid.acceptance_authorized(trial, policy, "primary")
    hybrid.finish(trial, True)
    assert document.finished[0][1] is False


def test_session_pool_restarts_once_from_accepted_source_after_process_failure(
    monkeypatch,
):
    created = []
    failures = [True]
    baseline = _document_observation(
        "command_error",
        'File "bug.v", line 1, characters 0-1:\nError:\nTARGET\n',
    )

    class FakeSession(object):
        def __init__(
            self,
            manager_command,
            context,
            accepted_source,
            generation=1,
            client_factory=None,
            log=None,
            request_timeout=rdm_backend.DEFAULT_REQUEST_TIMEOUT,
        ):
            self.context = context
            self.accepted_source = accepted_source
            self.generation = generation
            self.canonical_observation = baseline
            self.active_trial_count = 0
            self.closed = False
            created.append(self)

        def begin(self, candidate):
            assert isinstance(candidate, CandidateChange)
            if failures:
                failures.pop()
                raise rdm_backend.JsonRpcProcessError("server exited", 9, "boom")
            self.active_trial_count = 1
            return rdm_backend.RdmSessionTrial(
                self,
                self.generation,
                5,
                candidate.source,
                baseline,
                True,
            )

        def finish(self, trial, accepted):
            self.active_trial_count = 0
            if accepted:
                self.accepted_source = trial.source

        def close(self):
            self.closed = True

    monkeypatch.setattr(rdm_backend, "RdmSession", FakeSession)
    Context = namedtuple("Context", "role")
    context = Context("primary")
    pool = rdm_backend.RdmSessionPool(
        ("manager",),
        "accepted",
        LegacyTargetPolicy(False, "TARGET"),
    )
    trial = pool.begin(context, _candidate("candidate"))
    assert len(created) == 2
    assert all(session.accepted_source == "accepted" for session in created)
    assert created[0].closed
    assert pool.restart_count == 1
    pool.finish(trial, False)
    pool.close()


def test_ten_thousand_trials_keep_one_live_session_with_restart_cap(monkeypatch):
    created = []
    baseline = _document_observation(
        "command_error", "Error: TARGET"
    )

    class Session(object):
        def __init__(
            self,
            manager_command,
            context,
            accepted_source,
            generation=1,
            client_factory=None,
            log=None,
            request_timeout=rdm_backend.DEFAULT_REQUEST_TIMEOUT,
        ):
            self.context = context
            self.accepted_source = accepted_source
            self.generation = generation
            self.canonical_observation = baseline
            self.active_trial_count = 0
            created.append(self)

        def begin(self, candidate):
            assert isinstance(candidate, CandidateChange)
            self.active_trial_count = 1
            return rdm_backend.RdmSessionTrial(
                self, self.generation, 1, candidate.source, baseline, True
            )

        def finish(self, trial, accepted):
            self.active_trial_count = 0

        def close(self):
            self.active_trial_count = 0

    monkeypatch.setattr(rdm_backend, "RdmSession", Session)
    Context = namedtuple("Context", "role")
    pool = rdm_backend.RdmSessionPool(
        ("manager",),
        "accepted",
        LegacyTargetPolicy(False, "TARGET"),
        restart_every=17,
    )
    for _ in range(10000):
        trial = pool.begin(Context("primary"), _candidate("candidate"))
        pool.finish(trial, False)
        assert pool.session_count == 1
    assert pool.restart_count == 588
    assert len(created) == 589
    pool.close()
    assert pool.session_count == 0


def test_real_document_session_promotes_raw_source_and_discards_rejection(tmp_path):
    manager = shutil.which("rocq-doc-manager")
    if manager is None:
        pytest.skip("rocq-doc-manager is not installed")
    Context = namedtuple(
        "Context", "cwd logical_file arguments environment role"
    )
    context = Context(
        str(tmp_path),
        str(tmp_path / "bug.v"),
        (),
        EnvironmentSnapshot(os.environ),
        "primary",
    )
    source = "Definition x := True.\nCheck nope.\n"
    policy = LegacyTargetPolicy(False, "The reference nope was not found")
    pool = rdm_backend.RdmSessionPool(
        (manager,), source, policy, restart_every=1
    )
    try:
        preserving = pool.begin(
            context, _candidate("Check nope.\n", source)
        )
        assert preserving.observation.status == "command_error"
        assert preserving.observation.edit_strategy == "clear"
        assert preserving.observation.replaced_items == 2
        pool.finish(preserving, True)
        assert pool.accepted_source == "Check nope.\n"

        rejected = pool.begin(
            context, _candidate("Check I.\n", "Check nope.\n")
        )
        assert rejected.observation.status == "success"
        assert rejected.observation.edit_strategy == "replace"
        assert rejected.observation.replaced_items == 1
        pool.finish(rejected, False)
        assert pool.accepted_source == "Check nope.\n"

        parse_error = pool.begin(
            context, _candidate("Check (\n", "Check nope.\n")
        )
        assert parse_error.observation.status == "parse_error"
        assert "Error:" in parse_error.observation.output
        pool.finish(parse_error, False)
        assert pool.accepted_source == "Check nope.\n"
        assert pool.restart_count >= 2
    finally:
        pool.close()
    assert not list(tmp_path.glob(".coq-tools-rdm-*"))
