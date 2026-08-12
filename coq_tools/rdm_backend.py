"""Dependency-free client and evaluator support for ``rocq-doc-manager``.

The module is intentionally compatible with Python 3.6.  The low-level JSON-RPC
transport is kept separate from the document and candidate-evaluator layers so
it can be tested without Rocq.
"""
from __future__ import print_function

import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import namedtuple

from .candidate_evaluator import (
    CandidateChange,
    CandidateEvaluator,
    CandidateLifecycleError,
    CoqcEvaluator,
    Evaluation,
    EvaluationStatus,
    EvaluationTrial,
)


DEFAULT_MAX_FRAME_SIZE = 64 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_HYBRID_RESTART_EVERY = 500
PROCESS_TERM_GRACE = 0.5
TRUSTED_MANAGER_SHA256 = frozenset(
    ("8ef1e2e9a4637b48375ebf5ffabfe72a07df4e3511df794d9b4a5a605777b2c7",)
)


class RdmError(Exception):
    """Base class for rocq-doc-manager backend failures."""


class RdmUnavailable(RdmError):
    """The requested document-manager backend is not available."""


class RdmUnsupported(RdmError):
    """The current execution context cannot be represented safely."""


class JsonRpcError(RdmError):
    """Base class for framed JSON-RPC failures."""


class JsonRpcFramingError(JsonRpcError):
    """A Content-Length frame was missing or malformed."""


class JsonRpcProtocolError(JsonRpcError):
    """A frame contained invalid JSON-RPC data."""


class JsonRpcDeadlineExceeded(JsonRpcError):
    """A startup or request exceeded its absolute wall-clock deadline."""

    def __init__(self, phase, timeout, method=None, stderr_tail=""):
        message = "JSON-RPC %s exceeded %.3f seconds" % (phase, timeout)
        if method is not None:
            message += " while calling %s" % method
        super(JsonRpcDeadlineExceeded, self).__init__(message)
        self.phase = phase
        self.timeout = timeout
        self.method = method
        self.stderr_tail = stderr_tail


class JsonRpcProcessError(JsonRpcError):
    """The JSON-RPC child failed or exited unexpectedly."""

    def __init__(self, message, returncode=None, stderr_tail=""):
        super(JsonRpcProcessError, self).__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class JsonRpcRequestError(JsonRpcError):
    """A method reported a recoverable JSON-RPC error."""

    def __init__(self, method, request_id, code, message, data=None):
        super(JsonRpcRequestError, self).__init__(message)
        self.method = method
        self.request_id = request_id
        self.code = code
        self.data = data


JsonRpcNotification = namedtuple("JsonRpcNotification", "method params")
DocumentItem = namedtuple("DocumentItem", "kind text data offset")
RocqLocation = namedtuple(
    "RocqLocation",
    "bp ep line_nb bol_pos line_nb_last bol_pos_last fname",
)
FeedbackMessage = namedtuple(
    "FeedbackMessage", "level text location quickfixes"
)
CommandError = namedtuple(
    "CommandError", "message error_loc feedback_messages"
)
StepsError = namedtuple(
    "StepsError", "message nb_processed command_error"
)
SentenceSplitError = namedtuple(
    "SentenceSplitError", "message sentences rest"
)
RdmProbe = namedtuple(
    "RdmProbe",
    "command resolved_executable fingerprint sha256 methods missing_methods stderr",
)
RdmDiagnostic = namedtuple(
    "RdmDiagnostic", "phase message location feedback_messages"
)
RdmObservation = namedtuple(
    "RdmObservation",
    (
        "status output diagnostics runtime completion processed_items "
        "candidate_items replay_item reused_items split_runtime execution_runtime "
        "edit_strategy edit_kind replaced_items cursor generation notifications"
    ),
)
RdmSessionTrial = namedtuple(
    "RdmSessionTrial", "session generation cursor source observation promotable"
)
ItemSplicePlan = namedtuple(
    "ItemSplicePlan",
    "start_item end_item source_offset replacement strategy",
)
ShadowTrialToken = namedtuple(
    "ShadowTrialToken",
    (
        "source context compiler_trial document_trial document_observation "
        "unsupported_reason"
    ),
)
HybridTrialToken = namedtuple(
    "HybridTrialToken",
    (
        "route source context role policy_identity compiler_trial document_trial "
        "document_observation document_verdict compiler_verdict reason audited"
    ),
)
DocumentOnlyTrialToken = namedtuple(
    "DocumentOnlyTrialToken",
    (
        "source context role policy_identity document_trial "
        "document_observation document_verdict"
    ),
)

REQUIRED_METHODS = frozenset(
    (
        "load_file",
        "doc_prefix",
        "doc_suffix",
        "clone",
        "go_to",
        "revert_before",
        "clear_suffix",
        "replace_suffix",
        "run_step",
        "run_steps",
        "contents",
        "dispose",
    )
)


def _freeze_json(value):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_json(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _context_record(context):
    if not hasattr(context, "resource_policy"):
        return {"role": getattr(context, "role", None)}
    request = context.resource_policy.request
    return {
        "evaluator_identity": context.evaluator_identity,
        "executable": context.executable,
        "executable_identity": context.executable_identity,
        "arguments": context.arguments,
        "cwd": context.cwd,
        "environment_digest": context.environment.digest,
        "environment_entry_count": context.environment.entry_count,
        "logical_file": context.logical_file,
        "top_name": context.top_name,
        "is_toplevel": context.is_toplevel,
        "pass_on_stdin": context.pass_on_stdin,
        "checker_executable": context.checker_executable,
        "checker_executable_identity": context.checker_executable_identity,
        "checker_arguments": context.checker_arguments,
        "resource_request": tuple(request),
        "role": context.role,
    }


def _normalize_output(value):
    value = re.sub(r'File "[^"]+", line [0-9-]+, characters [0-9-]+:', "", value)
    return " ".join(value.split())


def _append_jsonl(path, record):
    if not path:
        return
    value = dict(record)
    value.setdefault("timestamp", time.time())
    case_id = os.environ.get("COQ_TOOLS_RDM_CASE_ID")
    run_id = os.environ.get("COQ_TOOLS_RDM_RUN_ID")
    if case_id:
        value.setdefault("case_id", case_id)
    if run_id:
        value.setdefault("run_id", run_id)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def _require_int(value, description, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise JsonRpcProtocolError("%s must be an integer >= %d" % (description, minimum))
    return value


def _require_text(value, description):
    if not isinstance(value, str):
        raise JsonRpcProtocolError("%s must be text" % description)
    return value


def _optional_int(mapping, key):
    value = mapping.get(key)
    if value is None:
        return None
    return _require_int(value, key)


def _validate_location(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise JsonRpcProtocolError("Rocq location must be an object or null")
    bp = _require_int(value.get("bp"), "location bp")
    ep = _require_int(value.get("ep"), "location ep")
    if ep < bp:
        raise JsonRpcProtocolError("location ep precedes bp")
    return RocqLocation(
        bp,
        ep,
        _optional_int(value, "line_nb"),
        _optional_int(value, "bol_pos"),
        _optional_int(value, "line_nb_last"),
        _optional_int(value, "bol_pos_last"),
        _freeze_json(value.get("fname")),
    )


def _validate_feedback(value):
    if not isinstance(value, dict):
        raise JsonRpcProtocolError("Feedback message must be an object")
    level = _require_text(value.get("level"), "feedback level")
    if level not in ("debug", "info", "notice", "warning", "error"):
        raise JsonRpcProtocolError("Unknown feedback level %r" % level)
    text = _require_text(value.get("text"), "feedback text")
    quickfixes = value.get("quickfix", ())
    if not isinstance(quickfixes, (list, tuple)):
        raise JsonRpcProtocolError("feedback quickfix must be a list")
    return FeedbackMessage(
        level,
        text,
        _validate_location(value.get("loc")),
        _freeze_json(quickfixes),
    )


def _validate_command_error(value, message=""):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise JsonRpcProtocolError("Command error must be an object")
    feedback = value.get("feedback_messages", ())
    if not isinstance(feedback, (list, tuple)):
        raise JsonRpcProtocolError("feedback_messages must be a list")
    return CommandError(
        message,
        _validate_location(value.get("error_loc")),
        tuple(_validate_feedback(item) for item in feedback),
    )


def parse_steps_error(error):
    """Validate the structured data of a recoverable ``run_steps`` error."""
    if not isinstance(error, JsonRpcRequestError) or error.method != "run_steps":
        raise TypeError("Expected a run_steps JsonRpcRequestError")
    data = error.data
    if not isinstance(data, dict):
        raise JsonRpcProtocolError("Steps error data must be an object")
    nb_processed = _require_int(data.get("nb_processed"), "nb_processed")
    command_error = _validate_command_error(data.get("cmd_error"), error.args[0])
    return StepsError(error.args[0], nb_processed, command_error)


def parse_sentence_split_error(error):
    if not isinstance(error, JsonRpcRequestError) or error.method not in (
        "replace_suffix",
        "split_sentences",
    ):
        raise TypeError("Expected a sentence-splitting JsonRpcRequestError")
    data = error.data
    if not isinstance(data, dict):
        raise JsonRpcProtocolError("Sentence split error data must be an object")
    sentences = _validate_items(data.get("sentences", ()), prefix=False, sentence=True)
    rest = _require_text(data.get("rest"), "sentence split rest")
    return SentenceSplitError(error.args[0], sentences, rest)


def _validate_item(value, prefix=False, sentence=False):
    if not isinstance(value, dict):
        raise JsonRpcProtocolError("Document item must be an object")
    allowed_kinds = ("blanks", "command") if sentence else ("blanks", "command", "ghost")
    kind = _require_text(value.get("kind"), "item kind")
    if kind not in allowed_kinds:
        raise JsonRpcProtocolError("Unknown document item kind %r" % kind)
    text = _require_text(value.get("text"), "item text")
    offset = None
    if prefix:
        offset = _require_int(value.get("offset"), "prefix item offset")
    return DocumentItem(kind, text, _freeze_json(value.get("data")), offset)


def _validate_items(value, prefix=False, sentence=False):
    if not isinstance(value, (list, tuple)):
        raise JsonRpcProtocolError("Document items must be a list")
    return tuple(
        _validate_item(item, prefix=prefix, sentence=sentence) for item in value
    )


def encode_json_rpc_frame(value):
    """Encode one JSON value using jsonrpc-tp Content-Length framing."""
    try:
        payload = json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise JsonRpcProtocolError("Unable to encode JSON-RPC packet: %s" % exc)
    return b"Content-Length: %d\r\n\r\n" % len(payload) + payload


def _read_exactly(stream, count):
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            received = count - remaining
            raise JsonRpcFramingError(
                "Unexpected EOF after %d of %d frame bytes" % (received, count)
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_json_rpc_frame(stream, max_frame_size=DEFAULT_MAX_FRAME_SIZE):
    """Read and decode one strict Content-Length framed JSON value."""
    header = stream.readline()
    if not header:
        raise JsonRpcFramingError("Unexpected EOF before JSON-RPC header")
    prefix = b"Content-Length: "
    if not header.startswith(prefix) or not header.endswith(b"\r\n"):
        raise JsonRpcFramingError("Invalid JSON-RPC header %r" % (header,))
    length_bytes = header[len(prefix) : -2]
    if not length_bytes or not length_bytes.isdigit():
        raise JsonRpcFramingError("Invalid Content-Length %r" % (length_bytes,))
    length = int(length_bytes)
    if length < 0 or length > max_frame_size:
        raise JsonRpcFramingError(
            "Content-Length %d exceeds allowed range 0..%d"
            % (length, max_frame_size)
        )
    blank = stream.readline()
    if blank != b"\r\n":
        raise JsonRpcFramingError(
            "Expected blank line after Content-Length, got %r" % (blank,)
        )
    payload = _read_exactly(stream, length)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JsonRpcProtocolError("Response was not valid UTF-8: %s" % exc)
    try:
        return json.loads(text)
    except (TypeError, ValueError) as exc:
        raise JsonRpcProtocolError("Response was not valid JSON: %s" % exc)


def _validate_packet(packet):
    if not isinstance(packet, dict):
        raise JsonRpcProtocolError("JSON-RPC packet must be an object")
    if packet.get("jsonrpc") != "2.0":
        raise JsonRpcProtocolError("JSON-RPC packet has an invalid version")
    return packet


class JsonRpcProcess(object):
    """Synchronous single-request JSON-RPC process connection."""

    def __init__(
        self,
        command,
        cwd=None,
        environment=None,
        max_frame_size=DEFAULT_MAX_FRAME_SIZE,
        startup_method="ready_seq",
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
    ):
        self.command = tuple(command)
        self.cwd = cwd
        self._environment = None if environment is None else dict(environment)
        self._max_frame_size = max_frame_size
        self._request_timeout = float(request_timeout)
        if self._request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self._process = None
        self._process_group = None
        self._read_buffer = bytearray()
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        self._notifications = []
        self._next_id = 0
        self._in_request = False
        self._absolute_deadline = None
        self._writer_thread = None
        self._usable = True
        self._closed = False
        self.returncode = None
        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self._environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                start_new_session=(os.name == "posix"),
            )
            if os.name == "posix":
                self._process_group = self._process.pid
            self._wait_for_startup(startup_method)
        except BaseException:
            self._cleanup_after_startup_failure()
            raise

    @property
    def notifications(self):
        return tuple(self._notifications)

    def _stderr_tail(self, limit=8192):
        try:
            self._stderr.flush()
            self._stderr.seek(0, 2)
            size = self._stderr.tell()
            self._stderr.seek(max(0, size - limit))
            return self._stderr.read().decode("utf-8", "replace")
        except BaseException:
            return ""

    def _poll_after_eof(self):
        if self._process is None:
            return self.returncode
        code = self._process.poll()
        if code is None:
            try:
                code = self._process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                code = None
        if code is not None:
            self.returncode = code
        return code

    def _read_until(self, marker, deadline, phase, method=None):
        if self._process is None or self._process.stdout is None:
            raise JsonRpcError("JSON-RPC process is not running")
        descriptor = self._process.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while marker not in self._read_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise JsonRpcDeadlineExceeded(
                        phase,
                        self._request_timeout,
                        method=method,
                        stderr_tail=self._stderr_tail(),
                    )
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    raise JsonRpcFramingError("Unexpected EOF while reading frame")
                self._read_buffer.extend(chunk)

    def _read_exact_buffered(self, length, deadline, phase, method=None):
        if self._process is None or self._process.stdout is None:
            raise JsonRpcError("JSON-RPC process is not running")
        descriptor = self._process.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while len(self._read_buffer) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise JsonRpcDeadlineExceeded(
                        phase,
                        self._request_timeout,
                        method=method,
                        stderr_tail=self._stderr_tail(),
                    )
                chunk = os.read(descriptor, max(65536, length - len(self._read_buffer)))
                if not chunk:
                    raise JsonRpcFramingError("Unexpected EOF while reading frame")
                self._read_buffer.extend(chunk)
        value = bytes(self._read_buffer[:length])
        del self._read_buffer[:length]
        return value

    def _receive(self, deadline=None, phase="request", method=None):
        if deadline is None:
            deadline = time.monotonic() + self._request_timeout
        try:
            self._read_until(b"\r\n\r\n", deadline, phase, method)
            end = self._read_buffer.index(b"\r\n\r\n")
            header = bytes(self._read_buffer[:end])
            del self._read_buffer[: end + 4]
            lines = header.split(b"\r\n")
            lengths = []
            for line in lines:
                if b":" not in line:
                    raise JsonRpcFramingError("Malformed JSON-RPC header")
                key, value = line.split(b":", 1)
                if key.strip().lower() == b"content-length":
                    try:
                        lengths.append(int(value.strip()))
                    except ValueError:
                        raise JsonRpcFramingError("Invalid Content-Length")
            if len(lengths) != 1 or lengths[0] < 0:
                raise JsonRpcFramingError("Expected one Content-Length header")
            if lengths[0] > self._max_frame_size:
                raise JsonRpcFramingError("JSON-RPC payload is too large")
            payload = self._read_exact_buffered(
                lengths[0], deadline, phase, method
            )
            try:
                packet = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise JsonRpcProtocolError("Invalid JSON-RPC payload: %s" % exc)
            return _validate_packet(packet)
        except JsonRpcProtocolError:
            self._usable = False
            raise
        except (JsonRpcFramingError, JsonRpcDeadlineExceeded):
            code = self._poll_after_eof()
            self._usable = False
            if code is not None:
                raise JsonRpcProcessError(
                    "JSON-RPC process exited with return code %s" % code,
                    returncode=code,
                    stderr_tail=self._stderr_tail(),
                )
            raise

    def _wait_for_startup(self, startup_method):
        deadline = time.monotonic() + self._request_timeout
        while True:
            packet = self._receive(deadline, "startup")
            if "id" in packet or "result" in packet or "error" in packet:
                self._usable = False
                raise JsonRpcProtocolError(
                    "Received a response before %s notification" % startup_method
                )
            method = packet.get("method")
            if not isinstance(method, str):
                self._usable = False
                raise JsonRpcProtocolError("Startup notification has no method")
            params = packet.get("params")
            if method == startup_method:
                return
            self._notifications.append(JsonRpcNotification(method, params))

    def _ensure_usable(self):
        if self._closed or self._process is None:
            raise JsonRpcError("JSON-RPC process is closed")
        if not self._usable:
            raise JsonRpcError("JSON-RPC process is unusable")
        code = self._process.poll()
        if code is not None:
            self.returncode = code
            self._usable = False
            raise JsonRpcProcessError(
                "JSON-RPC process exited with return code %s" % code,
                returncode=code,
                stderr_tail=self._stderr_tail(),
            )

    def _write_deadline_error(self, request_timeout, method):
        return JsonRpcDeadlineExceeded(
            "request write",
            request_timeout,
            method=method,
            stderr_tail=self._stderr_tail(),
        )

    def _write_frame_posix(
        self, frame, deadline, request_timeout, method
    ):
        """Write one frame with nonblocking POSIX pipe readiness."""
        descriptor = self._process.stdin.fileno()
        get_blocking = getattr(os, "get_blocking", None)
        set_blocking = getattr(os, "set_blocking", None)
        if get_blocking is None or set_blocking is None:
            return self._write_frame_threaded(
                frame, deadline, request_timeout, method
            )
        was_blocking = get_blocking(descriptor)
        try:
            set_blocking(descriptor, False)
            offset = 0
            view = memoryview(frame)
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_WRITE)
                while offset < len(frame):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise self._write_deadline_error(
                            request_timeout, method
                        )
                    try:
                        written = os.write(
                            descriptor, view[offset : offset + 65536]
                        )
                    except BlockingIOError:
                        continue
                    if written <= 0:
                        raise OSError("JSON-RPC pipe accepted no request bytes")
                    offset += written
        finally:
            try:
                set_blocking(descriptor, was_blocking)
            except OSError:
                pass

    def _write_frame_threaded(
        self, frame, deadline, request_timeout, method
    ):
        """Fallback for platforms whose selectors cannot monitor pipes.

        Terminating the child closes the read end and unblocks ordinary
        anonymous-pipe writes on the platforms supported by this backend.
        """
        descriptor = self._process.stdin.fileno()
        completed = threading.Event()
        errors = []

        def write_frame():
            try:
                offset = 0
                view = memoryview(frame)
                while offset < len(frame):
                    written = os.write(
                        descriptor, view[offset : offset + 65536]
                    )
                    if written <= 0:
                        raise OSError(
                            "JSON-RPC pipe accepted no request bytes"
                        )
                    offset += written
            except (IOError, OSError) as exc:
                errors.append(exc)
            finally:
                completed.set()

        writer = threading.Thread(target=write_frame)
        writer.daemon = True
        self._writer_thread = writer
        writer.start()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not completed.wait(remaining):
            raise self._write_deadline_error(request_timeout, method)
        writer.join()
        self._writer_thread = None
        if errors:
            raise errors[0]

    def _send(self, packet, deadline, request_timeout, method):
        self._ensure_usable()
        if self._process.stdin is None:
            raise JsonRpcError("JSON-RPC stdin is unavailable")
        try:
            frame = encode_json_rpc_frame(packet)
            if deadline - time.monotonic() <= 0:
                raise self._write_deadline_error(request_timeout, method)
            if os.name == "posix":
                self._write_frame_posix(
                    frame, deadline, request_timeout, method
                )
            else:
                self._write_frame_threaded(
                    frame, deadline, request_timeout, method
                )
        except JsonRpcDeadlineExceeded:
            self._usable = False
            self._terminate_process_group()
            writer = self._writer_thread
            if writer is not None:
                writer.join(PROCESS_TERM_GRACE)
                if not writer.is_alive():
                    self._writer_thread = None
            raise
        except (IOError, OSError) as exc:
            self._usable = False
            code = self._poll_after_eof()
            raise JsonRpcProcessError(
                "Failed to write JSON-RPC request: %s" % exc,
                returncode=code,
                stderr_tail=self._stderr_tail(),
            )

    def set_deadline(self, deadline):
        self._absolute_deadline = deadline

    def request(self, method, params, timeout=None):
        self._ensure_usable()
        if self._in_request:
            raise JsonRpcProtocolError("Concurrent or reentrant request is unsupported")
        if not isinstance(method, str) or not isinstance(params, (list, tuple)):
            raise TypeError("JSON-RPC method must be text and params must be a sequence")
        request_id = self._next_id
        self._next_id += 1
        request_timeout = self._request_timeout if timeout is None else float(timeout)
        if request_timeout <= 0:
            raise ValueError("request timeout must be positive")
        request_started = time.monotonic()
        deadline = request_started + request_timeout
        if self._absolute_deadline is not None:
            deadline = min(deadline, self._absolute_deadline)
        effective_request_timeout = max(0.0, deadline - request_started)
        self._in_request = True
        try:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": list(params),
                },
                deadline,
                effective_request_timeout,
                method,
            )
            while True:
                packet = self._receive(deadline, "request", method)
                if "method" in packet and "id" not in packet:
                    notification_method = packet.get("method")
                    if not isinstance(notification_method, str):
                        self._usable = False
                        raise JsonRpcProtocolError(
                            "Notification method must be text"
                        )
                    self._notifications.append(
                        JsonRpcNotification(
                            notification_method, packet.get("params")
                        )
                    )
                    continue
                if packet.get("id") != request_id:
                    self._usable = False
                    raise JsonRpcProtocolError(
                        "Expected response id %r, got %r"
                        % (request_id, packet.get("id"))
                    )
                has_result = "result" in packet
                has_error = "error" in packet
                if has_result == has_error:
                    self._usable = False
                    raise JsonRpcProtocolError(
                        "Response must contain exactly one of result or error"
                    )
                if has_result:
                    return packet["result"]
                error = packet["error"]
                if not isinstance(error, dict):
                    self._usable = False
                    raise JsonRpcProtocolError("JSON-RPC error must be an object")
                code = error.get("code")
                message = error.get("message")
                if isinstance(code, bool) or not isinstance(code, int):
                    self._usable = False
                    raise JsonRpcProtocolError("JSON-RPC error code must be an integer")
                if not isinstance(message, str):
                    self._usable = False
                    raise JsonRpcProtocolError("JSON-RPC error message must be text")
                if code == -32803:
                    raise JsonRpcRequestError(
                        method,
                        request_id,
                        code,
                        message,
                        error.get("data"),
                    )
                self._usable = False
                raise JsonRpcProtocolError(
                    "Unexpected JSON-RPC error %d for %s: %s"
                    % (code, method, message)
                )
        finally:
            self._in_request = False

    def _terminate_process_group(self):
        process = self._process
        if process is None:
            return
        try:
            if self._process_group is not None and os.name == "posix":
                # Signal the group even if the direct child already exited;
                # splitter descendants may still be alive.
                os.killpg(self._process_group, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
            if process.poll() is None:
                process.wait(timeout=PROCESS_TERM_GRACE)
            elif self._process_group is None:
                return
        except (subprocess.TimeoutExpired, OSError):
            pass
        try:
            if self._process_group is not None and os.name == "posix":
                os.killpg(self._process_group, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=PROCESS_TERM_GRACE)
        except BaseException:
            pass

    def _cleanup_after_startup_failure(self):
        process = self._process
        self._terminate_process_group()
        if process is not None:
            self.returncode = process.poll()
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except BaseException:
                    pass
        self._process = None
        self._usable = False

    def close(self):
        if self._closed:
            return
        process = self._process
        first_error = None
        writer = self._writer_thread
        if writer is not None and writer.is_alive():
            try:
                self._terminate_process_group()
                writer.join(PROCESS_TERM_GRACE)
            except BaseException as exc:
                first_error = exc
        if writer is not None and not writer.is_alive():
            self._writer_thread = None
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except BaseException as exc:
                first_error = exc
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    self._terminate_process_group()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            try:
                self._terminate_process_group()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            self.returncode = process.poll()
            for stream in (process.stdout,):
                try:
                    if stream is not None:
                        stream.close()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        try:
            self._stderr.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self._process = None
        self._usable = False
        self._closed = True
        if first_error is not None:
            raise first_error

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass


class RdmClient(object):
    """Validated positional wrapper for the document-manager RPC methods."""

    def __init__(self, transport):
        self.transport = transport

    def _null(self, method, params):
        result = self.transport.request(method, params)
        if result is not None:
            raise JsonRpcProtocolError("%s returned non-null result" % method)
        return None

    def load_file(self, cursor):
        return self._null("load_file", [_require_int(cursor, "cursor")])

    def clone(self, cursor):
        result = self.transport.request("clone", [_require_int(cursor, "cursor")])
        return _require_int(result, "clone cursor")

    def go_to(self, cursor, index):
        return self._null(
            "go_to",
            [_require_int(cursor, "cursor"), _require_int(index, "index")],
        )

    def revert_before(self, cursor, erase, index):
        if not isinstance(erase, bool):
            raise TypeError("erase must be a boolean")
        return self._null(
            "revert_before",
            [
                _require_int(cursor, "cursor"),
                erase,
                _require_int(index, "index"),
            ],
        )

    def clear_suffix(self, cursor, count=None):
        if count is not None:
            count = _require_int(count, "count")
        return self._null(
            "clear_suffix", [_require_int(cursor, "cursor"), count]
        )

    def replace_suffix(self, cursor, text, count=None):
        if count is not None:
            count = _require_int(count, "count")
        result = self.transport.request(
            "replace_suffix",
            [_require_int(cursor, "cursor"), _require_text(text, "text"), count],
        )
        return _validate_items(result, prefix=False, sentence=True)

    def run_step(self, cursor):
        result = self.transport.request("run_step", [_require_int(cursor, "cursor")])
        if result is not None and not isinstance(result, dict):
            raise JsonRpcProtocolError("run_step result must be an object or null")
        return _freeze_json(result)

    def run_steps(self, cursor, count):
        return self._null(
            "run_steps",
            [_require_int(cursor, "cursor"), _require_int(count, "count")],
        )

    def contents(self, cursor, include_ghost=False, include_suffix=True):
        if not isinstance(include_ghost, bool) or not isinstance(include_suffix, bool):
            raise TypeError("contents flags must be booleans")
        result = self.transport.request(
            "contents",
            [_require_int(cursor, "cursor"), include_ghost, include_suffix],
        )
        return _require_text(result, "contents result")

    def dispose(self, cursor):
        return self._null("dispose", [_require_int(cursor, "cursor")])

    def doc_prefix(self, cursor):
        result = self.transport.request(
            "doc_prefix", [_require_int(cursor, "cursor")]
        )
        return _validate_items(result, prefix=True)

    def doc_suffix(self, cursor):
        result = self.transport.request(
            "doc_suffix", [_require_int(cursor, "cursor")]
        )
        return _validate_items(result, prefix=False)

    def cursor_index(self, cursor):
        result = self.transport.request(
            "cursor_index", [_require_int(cursor, "cursor")]
        )
        return _require_int(result, "cursor index")

    def has_suffix(self, cursor):
        result = self.transport.request(
            "has_suffix", [_require_int(cursor, "cursor")]
        )
        if not isinstance(result, bool):
            raise JsonRpcProtocolError("has_suffix result must be a boolean")
        return result

    def split_sentences(self, cursor, text):
        result = self.transport.request(
            "split_sentences",
            [_require_int(cursor, "cursor"), _require_text(text, "text")],
        )
        return _validate_items(result, prefix=False, sentence=True)

    def set_deadline(self, deadline):
        setter = getattr(self.transport, "set_deadline", None)
        if setter is not None:
            setter(deadline)

    def close(self):
        return self.transport.close()


def _resolved_executable(command, cwd, environment):
    executable = command[0]
    if os.path.isabs(executable):
        return os.path.abspath(executable)
    if os.path.dirname(executable):
        return os.path.abspath(os.path.join(cwd or os.getcwd(), executable))
    path = None if environment is None else environment.get("PATH")
    return shutil.which(executable, path=path)


def _file_fingerprint(path):
    if path is None:
        return (None,)
    try:
        stat_result = os.stat(path)
    except OSError:
        return (path, None)
    mtime_ns = getattr(
        stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1000000000)
    )
    return (
        path,
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        mtime_ns,
    )


def _sha256_file(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def probe_rdm(command, cwd=None, environment=None, timeout=10):
    """Probe ``--api-docs`` and return immutable executable capabilities."""
    command = tuple(command)
    if not command:
        raise RdmUnavailable("Empty rocq-doc-manager command")
    cwd = os.path.abspath(cwd or os.getcwd())
    environment_dict = None if environment is None else dict(environment)
    resolved = _resolved_executable(command, cwd, environment_dict)
    if resolved is None:
        raise RdmUnavailable(
            "Unable to resolve rocq-doc-manager executable %r" % (command[0],)
        )
    try:
        completed = subprocess.run(
            list(command) + ["--api-docs"],
            cwd=cwd,
            env=environment_dict,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RdmUnavailable("rocq-doc-manager capability probe failed: %s" % exc)
    stdout = completed.stdout.decode("utf-8", "replace")
    stderr = completed.stderr.decode("utf-8", "replace")
    if completed.returncode != 0:
        raise RdmUnavailable(
            "rocq-doc-manager --api-docs exited with %d: %s"
            % (completed.returncode, stderr[-2000:])
        )
    methods = tuple(sorted(set(re.findall(r"^### `([^`]+)`$", stdout, re.MULTILINE))))
    missing = tuple(sorted(REQUIRED_METHODS.difference(methods)))
    return RdmProbe(
        command,
        resolved,
        _file_fingerprint(resolved),
        _sha256_file(resolved),
        methods,
        missing,
        stderr,
    )


def find_item_boundary(items, old_source, new_source):
    """Return ``(item_index, character_offset)`` before the first edit."""
    old_from_items = "".join(item.text for item in items)
    if old_from_items != old_source:
        raise RdmUnsupported(
            "Document items do not reconstruct the accepted raw source"
        )
    common = 0
    limit = min(len(old_source), len(new_source))
    while common < limit and old_source[common] == new_source[common]:
        common += 1
    if len(old_source) == len(new_source) and common == len(old_source):
        return len(items), len(old_source)
    item_index = 0
    offset = 0
    for index, item in enumerate(items):
        next_offset = offset + len(item.text)
        if next_offset > common:
            break
        item_index = index + 1
        offset = next_offset
    return item_index, offset


def plan_item_splice(items, candidate, maximum_start_item=None):
    """Plan a bounded item replacement for one validated source edit."""
    if not isinstance(candidate, CandidateChange):
        raise TypeError("Item splice planning requires a CandidateChange")
    item_text = "".join(item.text for item in items)
    if item_text != candidate.base_source:
        raise RdmUnsupported(
            "Document items do not reconstruct the candidate base source"
        )
    if len(candidate.edits) > 1:
        return None
    if candidate.edits:
        edit = candidate.edits[0]
        edit_start = edit.start
        edit_end = edit.end
        inserted = edit.replacement
    else:
        edit_start = len(candidate.base_source)
        edit_end = edit_start
        inserted = ""

    boundaries = [0]
    for item in items:
        boundaries.append(boundaries[-1] + len(item.text))
    start_item = 0
    for index, offset in enumerate(boundaries):
        if offset > edit_start:
            break
        start_item = index
    end_item = len(items)
    for index, offset in enumerate(boundaries):
        if offset >= edit_end:
            end_item = index
            break
    if maximum_start_item is not None and start_item > maximum_start_item:
        start_item = maximum_start_item
    if end_item < start_item:
        end_item = start_item
    start_offset = boundaries[start_item]
    end_offset = boundaries[end_item]
    replacement = (
        candidate.base_source[start_offset:edit_start]
        + inserted
        + candidate.base_source[edit_end:end_offset]
    )
    # Sentence splitting is context-sensitive: after a processed command the
    # manager requires inserted text to start with blanks.  Pull an immediately
    # preceding blanks item into the bounded splice rather than falling back to
    # replacing the complete suffix.
    if (
        replacement
        and not replacement[0].isspace()
        and start_item > 0
        and items[start_item - 1].kind == "blanks"
    ):
        start_item -= 1
        leading = items[start_item].text
        start_offset -= len(leading)
        replacement = leading + replacement
    reconstructed = (
        candidate.base_source[:start_offset]
        + replacement
        + candidate.base_source[end_offset:]
    )
    if reconstructed != candidate.source:
        return None
    strategy = "clear" if replacement == "" else "replace"
    return ItemSplicePlan(
        start_item,
        end_item,
        start_offset,
        replacement,
        strategy,
    )


def offset_to_line_characters(source, bp, ep):
    """Convert absolute UTF-8 byte offsets to Coq line/character fields."""
    source_bytes = source.encode("utf-8")
    bp = _require_int(bp, "diagnostic bp")
    ep = _require_int(ep, "diagnostic ep")
    if ep < bp or ep > len(source_bytes):
        raise RdmUnsupported("Diagnostic location is outside candidate source")
    line = source_bytes.count(b"\n", 0, bp) + 1
    bol = source_bytes.rfind(b"\n", 0, bp)
    bol = 0 if bol < 0 else bol + 1
    return line, bp - bol, ep - bol


def render_diagnostic(logical_file, source, message, location=None, fallback_offset=None):
    """Render one structured diagnostic for legacy target-regex comparison."""
    if location is not None:
        bp, ep = location.bp, location.ep
    elif fallback_offset is not None:
        bp = len(source[:fallback_offset].encode("utf-8"))
        ep = bp
    else:
        raise RdmUnsupported("No safe diagnostic location is available")
    line, start, end = offset_to_line_characters(source, bp, ep)
    file_name = os.path.basename(logical_file or "candidate.v")
    return 'File "%s", line %d, characters %d-%d:\nError:\n%s\n' % (
        file_name,
        line,
        start,
        end,
        message,
    )


def _select_command_message(parsed_error):
    if parsed_error.command_error is None:
        return parsed_error.message, None, ()
    command_error = parsed_error.command_error
    errors = [
        item for item in command_error.feedback_messages if item.level == "error"
    ]
    selected = errors[0] if errors else (
        command_error.feedback_messages[0]
        if command_error.feedback_messages
        else None
    )
    message = selected.text if selected is not None else parsed_error.message
    location = command_error.error_loc
    if location is None and selected is not None:
        location = selected.location
    return message, location, command_error.feedback_messages


def _default_client_factory(
    command, cwd, environment, request_timeout=DEFAULT_REQUEST_TIMEOUT
):
    return RdmClient(
        JsonRpcProcess(
            command,
            cwd=cwd,
            environment=environment,
            request_timeout=request_timeout,
        )
    )


class RdmSession(object):
    """One canonical document cursor for an exact immutable context."""

    def __init__(
        self,
        manager_command,
        context,
        accepted_source,
        generation=1,
        client_factory=None,
        log=None,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
    ):
        self.manager_command = tuple(manager_command)
        self.context = context
        self.accepted_source = accepted_source
        self.generation = generation
        self._client_factory = client_factory or _default_client_factory
        self._request_timeout = float(request_timeout)
        self._log = log or (lambda *args, **kwargs: None)
        self.client = None
        self.canonical_cursor = None
        self.canonical_observation = None
        self._active = {}
        self._closed = False
        self._private_dir = None
        self._private_file = None
        try:
            self._start()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def _make_private_file(self):
        cwd = self.context.cwd
        try:
            self._private_dir = tempfile.mkdtemp(
                prefix=".coq-tools-rdm-", dir=cwd
            )
        except OSError:
            self._private_dir = tempfile.mkdtemp(prefix="coq-tools-rdm-")
        file_name = os.path.basename(self.context.logical_file or "candidate.v")
        if not file_name.endswith(".v"):
            file_name += ".v"
        self._private_file = os.path.join(self._private_dir, file_name)
        with open(self._private_file, "wb") as output:
            output.write(self.accepted_source.encode("utf-8"))

    def _start(self):
        self._make_private_file()
        command = list(self.manager_command) + [self._private_file]
        if self.context.arguments:
            command += ["--"] + list(self.context.arguments)
        try:
            self.client = self._client_factory(
                tuple(command),
                self.context.cwd,
                self.context.environment.as_dict(),
                self._request_timeout,
            )
        except TypeError:
            self.client = self._client_factory(
                tuple(command),
                self.context.cwd,
                self.context.environment.as_dict(),
            )
        self.client.set_deadline(time.monotonic() + self._request_timeout)
        try:
            try:
                self.client.load_file(0)
            except JsonRpcRequestError as exc:
                raise RdmUnsupported(
                    "Accepted baseline could not be loaded: %s" % exc
                )
            if self.client.contents(0, False, True) != self.accepted_source:
                raise RdmUnsupported(
                    "Loaded document does not match accepted raw source"
                )
            self.canonical_cursor = 0
            self.canonical_observation = self._run_cursor(
                0, self.accepted_source, replay_item=0, split_runtime=0.0
            )
        finally:
            self.client.set_deadline(None)

    @property
    def private_file(self):
        return self._private_file

    @property
    def active_trial_count(self):
        return len(self._active)

    def _notifications(self):
        transport = getattr(self.client, "transport", None)
        return tuple(getattr(transport, "notifications", ()))

    def _run_cursor(
        self,
        cursor,
        source,
        replay_item,
        split_runtime,
        edit_strategy="load",
        edit_kind=None,
        replaced_items=None,
    ):
        before_prefix = self.client.doc_prefix(cursor)
        suffix = self.client.doc_suffix(cursor)
        start = time.time()
        processed = len(suffix)
        parsed_error = None
        if suffix:
            try:
                self.client.run_steps(cursor, len(suffix))
            except JsonRpcRequestError as exc:
                parsed_error = parse_steps_error(exc)
                processed = parsed_error.nb_processed
        execution_runtime = time.time() - start
        if parsed_error is None:
            status = "success"
            output = ""
            diagnostics = ()
            completion = "end"
        else:
            status = "command_error"
            message, location, feedback = _select_command_message(parsed_error)
            fallback_index = min(
                replay_item + parsed_error.nb_processed,
                len(before_prefix) + len(suffix),
            )
            all_items = before_prefix + suffix
            fallback_offset = sum(
                len(item.text) for item in all_items[:fallback_index]
            )
            output = render_diagnostic(
                self.context.logical_file,
                source,
                message,
                location=location,
                fallback_offset=fallback_offset,
            )
            diagnostics = (
                RdmDiagnostic(
                    "execution", message, location, tuple(feedback)
                ),
            )
            completion = "command_error"
        after_prefix = self.client.doc_prefix(cursor)
        after_suffix = self.client.doc_suffix(cursor)
        candidate_items = len(after_prefix) + len(after_suffix)
        return RdmObservation(
            status,
            output,
            diagnostics,
            split_runtime + execution_runtime,
            completion,
            processed,
            candidate_items,
            replay_item,
            replay_item,
            split_runtime,
            execution_runtime,
            edit_strategy,
            edit_kind,
            replaced_items,
            cursor,
            self.generation,
            self._notifications(),
        )

    def _canonical_items(self):
        return self.client.doc_prefix(self.canonical_cursor) + self.client.doc_suffix(
            self.canonical_cursor
        )

    def begin(self, candidate):
        if self._closed:
            raise RdmUnavailable("Document session is closed")
        if not isinstance(candidate, CandidateChange):
            raise TypeError("Document session requires a CandidateChange")
        if candidate.base_source != self.accepted_source:
            raise RdmUnsupported("Candidate base does not match accepted document")
        source = candidate.source
        edit_kind = (
            candidate.edits[0].kind
            if len(candidate.edits) == 1
            else ("none" if not candidate.edits else "multiple")
        )
        self.client.set_deadline(time.monotonic() + self._request_timeout)
        cursor = None
        try:
            cursor = self.client.clone(self.canonical_cursor)
            items = self._canonical_items()
            canonical_index = len(self.client.doc_prefix(self.canonical_cursor))
            plan = plan_item_splice(
                items, candidate, maximum_start_item=canonical_index
            )
            if plan is None:
                replay_item, source_offset = find_item_boundary(
                    items, self.accepted_source, source
                )
                if replay_item > canonical_index:
                    replay_item = canonical_index
                    source_offset = sum(
                        len(item.text) for item in items[:canonical_index]
                    )
                replacement = source[source_offset:]
                replace_count = None
                strategy = "full_replace"
            else:
                replay_item = plan.start_item
                source_offset = plan.source_offset
                replacement = plan.replacement
                replace_count = plan.end_item - plan.start_item
                strategy = plan.strategy
                # The manager's incremental splitter requires leading blanks
                # after a command.  Some legacy minimized sources place the
                # next command directly after a period.  Such a splice cannot
                # preserve source text while starting at this cursor, so parse
                # from the beginning rather than disabling the session.
                if (
                    strategy != "clear"
                    and replay_item > 0
                    and replacement
                    and not replacement[0].isspace()
                ):
                    replay_item = 0
                    source_offset = 0
                    replacement = source
                    replace_count = None
                    strategy = "full_replace"
            self.client.go_to(cursor, replay_item)
            split_start = time.time()
            try:
                if strategy == "clear":
                    self.client.clear_suffix(cursor, replace_count)
                    sentences = ()
                else:
                    sentences = self.client.replace_suffix(
                        cursor, replacement, replace_count
                    )
            except JsonRpcRequestError as exc:
                split_error = parse_sentence_split_error(exc)
                split_runtime = time.time() - split_start
                message = split_error.message
                output = render_diagnostic(
                    self.context.logical_file,
                    source,
                    message,
                    fallback_offset=source_offset,
                )
                diagnostic = RdmDiagnostic(
                    "parse", message, None, ()
                )
                observation = RdmObservation(
                    "parse_error",
                    output,
                    (diagnostic,),
                    split_runtime,
                    "parse_error",
                    0,
                    replay_item + len(split_error.sentences),
                    replay_item,
                    replay_item,
                    split_runtime,
                    0.0,
                    strategy,
                    edit_kind,
                    replace_count,
                    cursor,
                    self.generation,
                    self._notifications(),
                )
                trial = RdmSessionTrial(
                    self, self.generation, cursor, source, observation, False
                )
                self._active[cursor] = trial
                return trial
            split_runtime = time.time() - split_start
            actual = self.client.contents(cursor, False, True)
            if actual != source:
                raise RdmUnsupported(
                    "Candidate document contents differ after suffix replacement"
                )
            observation = self._run_cursor(
                cursor,
                source,
                replay_item=replay_item,
                split_runtime=split_runtime,
                edit_strategy=strategy,
                edit_kind=edit_kind,
                replaced_items=replace_count,
            )
            trial = RdmSessionTrial(
                self, self.generation, cursor, source, observation, True
            )
            self._active[cursor] = trial
            return trial
        except BaseException:
            if cursor is not None:
                try:
                    self.client.dispose(cursor)
                except BaseException:
                    pass
            raise
        finally:
            self.client.set_deadline(None)

    def finish(self, trial, accepted):
        if trial.session is not self or trial.generation != self.generation:
            raise RdmUnavailable("Trial belongs to a stale document session")
        current = self._active.pop(trial.cursor, None)
        if current is None:
            raise RdmError("Document trial was already resolved")
        if not accepted:
            self.client.dispose(trial.cursor)
            return
        if not trial.promotable:
            self.client.dispose(trial.cursor)
            raise RdmUnsupported(
                "Accepted candidate did not produce promotable document state"
            )
        old_cursor = self.canonical_cursor
        self.canonical_cursor = trial.cursor
        self.accepted_source = trial.source
        self.canonical_observation = trial.observation
        if old_cursor != trial.cursor:
            self.client.dispose(old_cursor)

    def close(self):
        if self._closed:
            return
        first_error = None
        if self.client is not None:
            for cursor in tuple(self._active):
                try:
                    self.client.dispose(cursor)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                self._active.pop(cursor, None)
            if self.canonical_cursor is not None:
                try:
                    self.client.dispose(self.canonical_cursor)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            try:
                self.client.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self.client = None
        self.canonical_cursor = None
        if self._private_dir is not None:
            try:
                shutil.rmtree(self._private_dir)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._private_dir = None
        self._private_file = None
        self._closed = True
        if first_error is not None:
            raise first_error


class RdmSessionPool(object):
    """Context-keyed sessions rebuilt from the last accepted raw source."""

    def __init__(
        self,
        primary_manager,
        accepted_source,
        target_policy,
        passing_manager=None,
        restart_every=0,
        client_factory=None,
        log=None,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
    ):
        self.primary_manager = tuple(primary_manager)
        self.passing_manager = tuple(passing_manager or primary_manager)
        self.accepted_source = accepted_source
        self.target_policy = target_policy
        self.restart_every = int(restart_every)
        if self.restart_every < 0:
            raise ValueError("restart_every must not be negative")
        self._client_factory = client_factory
        self._request_timeout = float(request_timeout)
        self._log = log or (lambda *args, **kwargs: None)
        self._sessions = {}
        self._generation = 0
        self._attempts = {}
        self.restart_count = 0
        self._closed = False

    @property
    def session_count(self):
        return len(self._sessions)

    @property
    def cursor_count(self):
        return sum(
            1 + session.active_trial_count
            for session in self._sessions.values()
        )

    def _manager_for(self, context):
        return (
            self.passing_manager
            if context.role == "passing"
            else self.primary_manager
        )

    def _baseline_healthy(self, context, observation):
        evaluation = _observation_as_evaluation(observation)
        if context.role == "passing":
            return self.target_policy.passing_succeeds(evaluation)
        return self.target_policy.primary_preserves(evaluation)

    def _new_session(self, context):
        self._generation += 1
        session = RdmSession(
            self._manager_for(context),
            context,
            self.accepted_source,
            generation=self._generation,
            client_factory=self._client_factory,
            log=self._log,
            request_timeout=self._request_timeout,
        )
        if not self._baseline_healthy(context, session.canonical_observation):
            try:
                session.close()
            finally:
                raise RdmUnsupported(
                    "Document session did not reproduce the accepted baseline verdict"
                )
        self._sessions[context] = session
        self._attempts[context] = 0
        return session

    def _drop(self, context):
        session = self._sessions.pop(context, None)
        self._attempts.pop(context, None)
        if session is not None:
            session.close()

    def _session(self, context):
        if self._closed:
            raise RdmUnavailable("Document session pool is closed")
        # Keep one live context per role.  Context exploration (notably
        # argument minimization) must not accumulate manager processes.
        for other_context, other_session in tuple(self._sessions.items()):
            if other_context != context and other_context.role == context.role:
                if other_session.active_trial_count:
                    raise RdmUnavailable(
                        "Another document context for this role has an active trial"
                    )
                self._drop(other_context)
        session = self._sessions.get(context)
        if session is not None and session.accepted_source != self.accepted_source:
            if session.active_trial_count:
                raise RdmUnavailable(
                    "Stale document session still has an unresolved trial"
                )
            self._drop(context)
            session = None
        if (
            session is not None
            and self.restart_every > 0
            and self._attempts.get(context, 0) >= self.restart_every
            and not session.active_trial_count
        ):
            self._drop(context)
            self.restart_count += 1
            session = None
        return session or self._new_session(context)

    def begin(self, context, candidate):
        try:
            session = self._session(context)
            self._attempts[context] = self._attempts.get(context, 0) + 1
            return session.begin(candidate)
        except (JsonRpcError, OSError, EOFError):
            try:
                self._drop(context)
            except BaseException:
                pass
            self.restart_count += 1
            session = self._new_session(context)
            self._attempts[context] = 1
            return session.begin(candidate)

    def advance_accepted_source(self, source):
        """Record a compiler-confirmed acceptance without a document trial."""
        self.accepted_source = source
        for context, session in tuple(self._sessions.items()):
            if not session.active_trial_count:
                self._drop(context)

    def finish(self, trial, accepted):
        context = trial.session.context
        session = self._sessions.get(context)
        if session is not trial.session:
            raise RdmUnavailable("Document trial session is no longer canonical")
        try:
            session.finish(trial, accepted)
        except BaseException:
            try:
                self._drop(context)
            finally:
                if accepted:
                    self.accepted_source = trial.source
            raise
        if accepted:
            self.accepted_source = trial.source
            for other_context, other_session in tuple(self._sessions.items()):
                if (
                    other_context != context
                    and not other_session.active_trial_count
                    and other_session.accepted_source != self.accepted_source
                ):
                    self._drop(other_context)

    def close(self):
        if self._closed:
            return
        first_error = None
        for context in tuple(self._sessions):
            try:
                self._drop(context)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error


def _observation_as_evaluation(observation):
    """Create the minimal Evaluation shape expected by target policies."""
    return Evaluation(
        observation.status,
        observation.output,
        (),
        0 if observation.status == "success" else 1,
        observation.runtime,
        None,
        observation.diagnostics,
        (),
    )


class RdmEvaluator(object):
    """Stateful document engine used by shadow and hybrid adapters."""

    def __init__(
        self,
        primary_manager,
        accepted_source,
        target_policy,
        passing_manager=None,
        restart_every=0,
        cwd=None,
        environment=None,
        client_factory=None,
        log=None,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        require_trusted=False,
    ):
        self.primary_manager = tuple(primary_manager)
        self.passing_manager = tuple(passing_manager or primary_manager)
        self.target_policy = target_policy
        self._log = log or (lambda *args, **kwargs: None)
        self._client_factory = client_factory
        self._restart_every = restart_every
        self._request_timeout = float(request_timeout)
        self._require_trusted = bool(require_trusted)
        self._disabled = {}
        self._compiler_versions = {}
        self._closed = False
        self.primary_probe = None
        self.passing_probe = None
        self.probe_errors = ()
        self._context_probes = {}
        errors = []
        try:
            self.primary_probe = probe_rdm(
                self.primary_manager, cwd=cwd, environment=environment
            )
        except RdmError as exc:
            errors.append(("primary", str(exc)))
        if self.passing_manager == self.primary_manager:
            self.passing_probe = self.primary_probe
        else:
            try:
                self.passing_probe = probe_rdm(
                    self.passing_manager, cwd=cwd, environment=environment
                )
            except RdmError as exc:
                errors.append(("passing", str(exc)))
        self.probe_errors = tuple(errors)
        self.pool = RdmSessionPool(
            self.primary_manager,
            accepted_source,
            target_policy,
            passing_manager=self.passing_manager,
            restart_every=restart_every,
            client_factory=client_factory,
            log=self._log,
            request_timeout=self._request_timeout,
        )

    @property
    def identity(self):
        def probe_identity(probe):
            if probe is None:
                return None
            return (probe.fingerprint, probe.sha256, probe.methods)

        return (
            "rdm-evaluator",
            1,
            probe_identity(self.primary_probe),
            probe_identity(self.passing_probe),
        )

    @property
    def accepted_source(self):
        return self.pool.accepted_source

    def _probe_for(self, context):
        probe = self._context_probes.get(context)
        if probe is not None:
            return probe
        manager = (
            self.passing_manager
            if context.role == "passing"
            else self.primary_manager
        )
        try:
            probe = probe_rdm(
                manager,
                cwd=context.cwd,
                environment=context.environment.as_dict(),
                timeout=min(self._request_timeout, 10.0),
            )
        except RdmError as exc:
            self._disabled[context] = "capability-probe-failed: %s" % exc
            return None
        self._context_probes[context] = probe
        return probe

    def _compiler_version(self, context):
        key = (context.executable_identity, context.cwd, context.environment.digest)
        if key not in self._compiler_versions:
            try:
                completed = subprocess.run(
                    list(context.executable) + ["--version"],
                    cwd=context.cwd,
                    env=context.environment.as_dict(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=min(self._request_timeout, 10.0),
                )
                text = completed.stdout.decode("utf-8", "replace")
                match = re.search(r"(?:version\s+)?([0-9]+\.[0-9]+)", text)
                self._compiler_versions[key] = (
                    match.group(1)
                    if completed.returncode == 0 and match is not None
                    else None
                )
            except (OSError, subprocess.TimeoutExpired):
                self._compiler_versions[key] = None
        return self._compiler_versions[key]

    def unsupported_reason(self, context):
        if self._closed:
            return "document evaluator is closed"
        if context in self._disabled:
            return self._disabled[context]
        probe = self._probe_for(context)
        if probe is None:
            return "document-manager capability probe failed"
        if probe.missing_methods:
            return "missing-capability: %s" % ", ".join(probe.missing_methods)
        if self._require_trusted and probe.sha256 not in TRUSTED_MANAGER_SHA256:
            return "untrusted-manager-build: %s" % probe.sha256
        if self._require_trusted and os.name != "posix":
            return "unsupported-platform: process-group supervision unavailable"
        if self._require_trusted and self._compiler_version(context) != "9.2":
            return "toolchain-mismatch: trusted manager requires Rocq 9.2"
        if context.checker_executable is not None:
            return "coqchk/checker contexts are compiler-only"
        if context.is_toplevel or context.pass_on_stdin:
            return "coqtop/stdin execution is not represented by the document backend"
        request = context.resource_policy.request
        if any(
            value is not None
            for value in (
                request.max_mem_rss,
                request.max_mem_as,
                request.max_mem_rss_multiplier,
                request.max_mem_as_multiplier,
                request.cgroup,
            )
        ):
            return "memory-limited candidates require the fresh compiler oracle"
        return None

    def disable(self, context, reason):
        self._disabled[context] = str(reason)
        session = self.pool._sessions.get(context)
        if session is not None and not session.active_trial_count:
            try:
                self.pool._drop(context)
            except BaseException:
                pass

    def begin(self, context, candidate):
        if not isinstance(candidate, CandidateChange):
            raise TypeError("Document evaluator requires a CandidateChange")
        reason = self.unsupported_reason(context)
        if reason is not None:
            raise RdmUnsupported(reason)
        try:
            return self.pool.begin(context, candidate)
        except (RdmError, OSError, EOFError) as exc:
            reason = "%s: %s" % (type(exc).__name__, exc)
            self._disabled[context] = reason
            raise RdmUnavailable(reason)

    def finish(self, trial, accepted):
        return self.pool.finish(trial, accepted)

    def advance_accepted_source(self, source):
        return self.pool.advance_accepted_source(source)

    def reset(self):
        if self._closed:
            raise RdmUnavailable("Document evaluator is closed")
        accepted_source = self.pool.accepted_source
        self.pool.close()
        self._disabled.clear()
        self._compiler_versions.clear()
        self._context_probes.clear()
        self.pool = RdmSessionPool(
            self.primary_manager,
            accepted_source,
            self.target_policy,
            passing_manager=self.passing_manager,
            restart_every=self._restart_every,
            client_factory=self._client_factory,
            log=self._log,
            request_timeout=self._request_timeout,
        )

    def close(self):
        if self._closed:
            return
        self.pool.close()
        self._closed = True


class RdmOnlyEvaluator(CandidateEvaluator):
    """Use the document manager as the sole candidate decision authority.

    This deliberately unsafe experimental backend never confirms a candidate
    verdict with ``coqc`` and never falls back to it.  Unsupported contexts and
    document infrastructure failures therefore abort evaluation instead of
    silently changing authority.
    """

    def __init__(
        self,
        context_evaluator,
        primary_manager,
        accepted_source,
        target_policy,
        passing_manager=None,
        restart_every=DEFAULT_HYBRID_RESTART_EVERY,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        cwd=None,
        environment=None,
        client_factory=None,
        log=None,
        require_trusted=True,
    ):
        for name in ("materialize_context", "reset_calibration", "close"):
            if not hasattr(context_evaluator, name):
                raise TypeError("context evaluator lacks %s" % name)
        self.context_evaluator = context_evaluator
        self.target_policy = target_policy
        self._log = log or (lambda *args, **kwargs: None)
        self._jsonl_path = os.environ.get("COQ_TOOLS_RDM_JSONL")
        self.document_evaluator = RdmEvaluator(
            primary_manager,
            accepted_source,
            target_policy,
            passing_manager=passing_manager,
            restart_every=restart_every,
            cwd=cwd,
            environment=environment,
            client_factory=client_factory,
            log=self._log,
            request_timeout=request_timeout,
            require_trusted=require_trusted,
        )
        self._closed = False

    @property
    def identity(self):
        return (
            "rdm-only-evaluator",
            1,
            self.context_evaluator.identity,
            self.document_evaluator.identity,
            self.target_policy.identity,
        )

    @property
    def requires_materialization_for_accept(self):
        return True

    @property
    def invalidates_cache_after_accept(self):
        return True

    def materialize_context(self, spec):
        if self._closed:
            raise CandidateLifecycleError("Document-only evaluator is closed")
        return self.context_evaluator.materialize_context(spec)

    @staticmethod
    def _predicate(policy, role, evaluation):
        if role == "passing":
            return policy.passing_succeeds(evaluation)
        return policy.primary_preserves(evaluation)

    def begin(self, context, candidate, target_policy=None):
        if self._closed:
            raise CandidateLifecycleError("Document-only evaluator is closed")
        if not isinstance(candidate, CandidateChange):
            raise TypeError("Document-only evaluator requires a CandidateChange")
        policy = target_policy or self.target_policy
        if policy.identity != self.target_policy.identity:
            raise RdmUnsupported("target-policy-mismatch")
        reason = self.document_evaluator.unsupported_reason(context)
        if reason is not None:
            raise RdmUnsupported(
                "rdm-only cannot evaluate this context: %s" % reason
            )
        document_trial = self.document_evaluator.begin(context, candidate)
        observation = document_trial.observation
        evaluation = _observation_as_evaluation(observation)
        verdict = self._predicate(policy, context.role, evaluation)
        token = DocumentOnlyTrialToken(
            candidate.source,
            context,
            context.role,
            policy.identity,
            document_trial,
            observation,
            verdict,
        )
        return EvaluationTrial(
            evaluation,
            token,
            bool(document_trial.promotable and verdict),
            False,
        )

    def acceptance_authorized(self, trial, target_policy, role):
        token = trial.token
        return bool(
            isinstance(token, DocumentOnlyTrialToken)
            and token.document_trial is not None
            and token.role == role
            and token.policy_identity == target_policy.identity
            and token.document_verdict
        )

    def finish(self, trial, accepted):
        token = trial.token
        if not isinstance(token, DocumentOnlyTrialToken):
            raise CandidateLifecycleError("Invalid document-only trial token")
        if accepted and not token.document_verdict:
            try:
                self.document_evaluator.finish(token.document_trial, False)
            finally:
                raise CandidateLifecycleError(
                    "Document-only acceptance lacks document authorization"
                )
        self.document_evaluator.finish(token.document_trial, bool(accepted))

    def record_target_decision(self, trial, target_policy, role):
        token = trial.token
        if not isinstance(token, DocumentOnlyTrialToken):
            return
        document = token.document_observation
        record = {
            "schema": 1,
            "role": role,
            "document_status": document.status,
            "document_verdict": token.document_verdict,
        }
        self._log(
            "rdm-only: %s"
            % json.dumps(record, sort_keys=True, separators=(",", ":")),
            level=1,
        )
        pool = getattr(self.document_evaluator, "pool", None)
        artifact = dict(record)
        artifact.update(
            {
                "event": "candidate-comparison",
                "mode": "rdm-only",
                "source_sha256": _sha256_text(token.source),
                "source_bytes": len(token.source.encode("utf-8")),
                "context": _context_record(token.context),
                "backend_identity": self.document_evaluator.identity,
                "document_runtime": document.runtime,
                "document_split_runtime": document.split_runtime,
                "document_execution_runtime": document.execution_runtime,
                "reused_items": document.reused_items,
                "commands_replayed": document.processed_items,
                "candidate_items": document.candidate_items,
                "edit_strategy": document.edit_strategy,
                "edit_kind": document.edit_kind,
                "replaced_items": document.replaced_items,
                "restart_count": getattr(pool, "restart_count", None),
                "live_session_count": getattr(pool, "session_count", None),
                "live_cursor_count": getattr(pool, "cursor_count", None),
            }
        )
        _append_jsonl(self._jsonl_path, artifact)

    def reset_calibration(self, context=None):
        self.context_evaluator.reset_calibration(context)
        self.document_evaluator.reset()

    def close(self):
        if self._closed:
            return
        first_error = None
        try:
            self.document_evaluator.close()
        except BaseException as exc:
            first_error = exc
        try:
            self.context_evaluator.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error


class RdmShadowEvaluator(CandidateEvaluator):
    """Run both document manager and compiler while keeping compiler authority."""

    def __init__(
        self,
        compiler_evaluator,
        primary_manager,
        accepted_source,
        target_policy,
        passing_manager=None,
        restart_every=0,
        cwd=None,
        environment=None,
        client_factory=None,
        log=None,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
    ):
        if not isinstance(compiler_evaluator, CoqcEvaluator):
            # Tests may supply a compatible fake, so require behavior rather
            # than an exact type after this lightweight sanity check.
            for name in ("materialize_context", "begin", "finish", "close"):
                if not hasattr(compiler_evaluator, name):
                    raise TypeError("compiler evaluator lacks %s" % name)
        self.compiler_evaluator = compiler_evaluator
        self._log = log or (lambda *args, **kwargs: None)
        self._jsonl_path = os.environ.get("COQ_TOOLS_RDM_JSONL")
        self.document_evaluator = RdmEvaluator(
            primary_manager,
            accepted_source,
            target_policy,
            passing_manager=passing_manager,
            restart_every=restart_every,
            cwd=cwd,
            environment=environment,
            client_factory=client_factory,
            log=self._log,
            request_timeout=request_timeout,
        )
        self._closed = False

    @property
    def identity(self):
        return (
            "rdm-shadow-evaluator",
            1,
            self.compiler_evaluator.identity,
            self.document_evaluator.identity,
        )

    @property
    def requires_materialization_for_accept(self):
        return True

    def materialize_context(self, spec):
        if self._closed:
            raise CandidateLifecycleError("Shadow evaluator is closed")
        return self.compiler_evaluator.materialize_context(spec)

    @staticmethod
    def _shadow_details(observation, reason, document_identity):
        if observation is None:
            return (
                ("available", False),
                ("unsupported_reason", reason),
                ("backend_identity", document_identity),
            )
        return (
            ("available", True),
            ("unsupported_reason", None),
            ("backend_identity", document_identity),
            ("status", observation.status),
            ("output", observation.output),
            ("runtime", observation.runtime),
            ("completion", observation.completion),
            ("processed_items", observation.processed_items),
            ("candidate_items", observation.candidate_items),
            ("replay_item", observation.replay_item),
            ("reused_items", observation.reused_items),
            ("split_runtime", observation.split_runtime),
            ("execution_runtime", observation.execution_runtime),
            ("edit_strategy", observation.edit_strategy),
            ("edit_kind", observation.edit_kind),
            ("replaced_items", observation.replaced_items),
            ("cursor", observation.cursor),
            ("generation", observation.generation),
        )

    def begin(self, context, candidate, target_policy=None):
        if self._closed:
            raise CandidateLifecycleError("Shadow evaluator is closed")
        if not isinstance(candidate, CandidateChange):
            raise TypeError("Shadow evaluator requires a CandidateChange")
        source = candidate.source
        document_trial = None
        document_observation = None
        reason = self.document_evaluator.unsupported_reason(context)
        if reason is None:
            try:
                document_trial = self.document_evaluator.begin(context, candidate)
                document_observation = document_trial.observation
            except (RdmError, OSError, EOFError) as exc:
                reason = "%s: %s" % (type(exc).__name__, exc)
                self._log(
                    "rdm-shadow unavailable: %s" % reason,
                    level=1,
                )
        compiler_trial = None
        try:
            compiler_trial = self.compiler_evaluator.begin(
                context, candidate, target_policy
            )
        except BaseException:
            if document_trial is not None:
                try:
                    self.document_evaluator.finish(document_trial, False)
                except BaseException:
                    pass
            raise
        compiler = compiler_trial.evaluation
        details = tuple(compiler.details) + (
            (
                "rdm_shadow",
                self._shadow_details(
                    document_observation,
                    reason,
                    self.document_evaluator.identity,
                ),
            ),
        )
        evaluation = Evaluation(
            compiler.status,
            compiler.output,
            compiler.commands,
            compiler.returncode,
            compiler.runtime,
            compiler.peak_rss_kb,
            compiler.diagnostics,
            details,
        )
        token = ShadowTrialToken(
            source,
            context,
            compiler_trial,
            document_trial,
            document_observation,
            reason,
        )
        return EvaluationTrial(
            evaluation,
            token,
            document_trial is not None and document_trial.promotable,
            False,
        )

    def finish(self, trial, accepted):
        token = trial.token
        if not isinstance(token, ShadowTrialToken):
            raise CandidateLifecycleError("Invalid shadow trial token")
        try:
            self.compiler_evaluator.finish(token.compiler_trial, accepted)
        except BaseException:
            if token.document_trial is not None:
                try:
                    self.document_evaluator.finish(token.document_trial, False)
                except BaseException:
                    pass
            raise
        if token.document_trial is not None:
            try:
                self.document_evaluator.finish(token.document_trial, accepted)
            except BaseException as exc:
                # Shadow state is observational.  Its loss after an
                # authoritative write must not undo or misreport that write.
                self._log(
                    "rdm-shadow state discarded after finalization failure: %s"
                    % exc,
                    level=1,
                )

    def record_target_decision(self, trial, target_policy, role):
        token = trial.token
        if not isinstance(token, ShadowTrialToken):
            return
        compiler_evaluation = token.compiler_trial.evaluation
        if role == "passing":
            compiler_verdict = target_policy.passing_succeeds(compiler_evaluation)
            document_verdict = (
                None
                if token.document_observation is None
                else target_policy.passing_succeeds(
                    _observation_as_evaluation(token.document_observation)
                )
            )
        else:
            compiler_verdict = target_policy.primary_preserves(compiler_evaluation)
            document_verdict = (
                None
                if token.document_observation is None
                else target_policy.primary_preserves(
                    _observation_as_evaluation(token.document_observation)
                )
            )
        record = {
            "schema": 1,
            "role": role,
            "compiler_status": compiler_evaluation.status,
            "document_status": (
                None
                if token.document_observation is None
                else token.document_observation.status
            ),
            "compiler_verdict": compiler_verdict,
            "document_verdict": document_verdict,
            "agreement": (
                None
                if document_verdict is None
                else compiler_verdict == document_verdict
            ),
            "unsupported_reason": token.unsupported_reason,
            "document_runtime": (
                None
                if token.document_observation is None
                else token.document_observation.runtime
            ),
            "replay_item": (
                None
                if token.document_observation is None
                else token.document_observation.replay_item
            ),
            "processed_items": (
                None
                if token.document_observation is None
                else token.document_observation.processed_items
            ),
        }
        self._log(
            "rdm-shadow: %s"
            % json.dumps(record, sort_keys=True, separators=(",", ":")),
            level=1,
        )
        document = token.document_observation
        pool = getattr(self.document_evaluator, "pool", None)
        artifact = dict(record)
        artifact.update(
            {
                "event": "candidate-comparison",
                "mode": "rdm-shadow",
                "source_sha256": _sha256_text(token.source),
                "source_bytes": len(token.source.encode("utf-8")),
                "context": _context_record(token.context),
                "backend_identity": self.document_evaluator.identity,
                "compiler_runtime": compiler_evaluation.runtime,
                "compiler_peak_rss_kb": compiler_evaluation.peak_rss_kb,
                "document_split_runtime": (
                    None if document is None else document.split_runtime
                ),
                "document_execution_runtime": (
                    None if document is None else document.execution_runtime
                ),
                "reused_items": None if document is None else document.reused_items,
                "commands_replayed": (
                    None if document is None else document.processed_items
                ),
                "candidate_items": None if document is None else document.candidate_items,
                "edit_strategy": None if document is None else document.edit_strategy,
                "edit_kind": None if document is None else document.edit_kind,
                "replaced_items": None if document is None else document.replaced_items,
                "diagnostic_agreement": (
                    None
                    if document is None
                    else _normalize_output(compiler_evaluation.output)
                    == _normalize_output(document.output)
                ),
                "restart_count": getattr(pool, "restart_count", None),
                "live_session_count": getattr(pool, "session_count", None),
                "live_cursor_count": getattr(pool, "cursor_count", None),
            }
        )
        _append_jsonl(getattr(self, "_jsonl_path", None), artifact)

    def reset_calibration(self, context=None):
        self.compiler_evaluator.reset_calibration(context)
        try:
            self.document_evaluator.reset()
        except BaseException as exc:
            self._log("rdm-shadow reset failed: %s" % exc, level=1)

    def close(self):
        if self._closed:
            return
        first_error = None
        try:
            self.document_evaluator.close()
        except BaseException as exc:
            first_error = exc
        try:
            self.compiler_evaluator.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error


class RdmHybridEvaluator(CandidateEvaluator):
    """Use the document manager only as a rejection oracle.

    A live compiler trial is required before ``acceptance_authorized`` can
    succeed.  Document infrastructure failures fail open to the compiler;
    document verdicts can only avoid a compiler run for a rejected candidate.
    """

    def __init__(
        self,
        compiler_evaluator,
        primary_manager,
        accepted_source,
        target_policy,
        passing_manager=None,
        restart_every=DEFAULT_HYBRID_RESTART_EVERY,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        check_rejected_every=0,
        cwd=None,
        environment=None,
        client_factory=None,
        log=None,
        require_trusted=True,
    ):
        for name in (
            "materialize_context",
            "begin",
            "finish",
            "reset_calibration",
            "close",
        ):
            if not hasattr(compiler_evaluator, name):
                raise TypeError("compiler evaluator lacks %s" % name)
        self.compiler_evaluator = compiler_evaluator
        self.target_policy = target_policy
        self._log = log or (lambda *args, **kwargs: None)
        self._jsonl_path = os.environ.get("COQ_TOOLS_RDM_JSONL")
        self._check_rejected_every = int(check_rejected_every)
        if self._check_rejected_every < 0:
            raise ValueError("check_rejected_every must not be negative")
        self._reject_counts = {}
        self._baseline_confirmed = {}
        self.document_evaluator = RdmEvaluator(
            primary_manager,
            accepted_source,
            target_policy,
            passing_manager=passing_manager,
            restart_every=restart_every,
            cwd=cwd,
            environment=environment,
            client_factory=client_factory,
            log=self._log,
            request_timeout=request_timeout,
            require_trusted=require_trusted,
        )
        self._closed = False

    @property
    def identity(self):
        return (
            "rdm-hybrid-evaluator",
            1,
            self.compiler_evaluator.identity,
            self.document_evaluator.identity,
            self.target_policy.identity,
            self._check_rejected_every,
        )

    @property
    def requires_materialization_for_accept(self):
        return True

    @property
    def invalidates_cache_after_accept(self):
        return True

    def materialize_context(self, spec):
        if self._closed:
            raise CandidateLifecycleError("Hybrid evaluator is closed")
        return self.compiler_evaluator.materialize_context(spec)

    @staticmethod
    def _predicate(policy, role, evaluation):
        if role == "passing":
            return policy.passing_succeeds(evaluation)
        return policy.primary_preserves(evaluation)

    def _audit_rejection(self, context):
        count = self._reject_counts.get(context, 0) + 1
        self._reject_counts[context] = count
        return (
            self._check_rejected_every > 0
            and count % self._check_rejected_every == 0
        )

    def _ensure_reference_baseline(self, context, policy):
        accepted_source = self.document_evaluator.accepted_source
        if self._baseline_confirmed.get(context) == accepted_source:
            return None
        candidate = CandidateChange.from_sources(accepted_source, accepted_source)
        trial = self.compiler_evaluator.begin(context, candidate, policy)
        try:
            healthy = self._predicate(policy, context.role, trial.evaluation)
        finally:
            self.compiler_evaluator.finish(trial, False)
        if not healthy:
            reason = "reference-baseline-mismatch"
            self.document_evaluator.disable(context, reason)
            return reason
        self._baseline_confirmed[context] = accepted_source
        return None

    @staticmethod
    def _details(
        route,
        document_observation,
        document_verdict,
        compiler_evaluation,
        compiler_verdict,
        reason,
        audited,
        document_identity,
    ):
        return (
            ("route", route),
            ("document", RdmShadowEvaluator._shadow_details(
                document_observation, reason, document_identity
            )),
            ("document_verdict", document_verdict),
            ("compiler_ran", compiler_evaluation is not None),
            ("compiler_verdict", compiler_verdict),
            ("audited", audited),
            ("fallback_reason", reason),
        )

    @staticmethod
    def _copy_with_details(evaluation, details):
        return Evaluation(
            evaluation.status,
            evaluation.output,
            evaluation.commands,
            evaluation.returncode,
            evaluation.runtime,
            evaluation.peak_rss_kb,
            evaluation.diagnostics,
            tuple(evaluation.details) + (("rdm_hybrid", details),),
        )

    def begin(self, context, candidate, target_policy=None):
        if self._closed:
            raise CandidateLifecycleError("Hybrid evaluator is closed")
        if not isinstance(candidate, CandidateChange):
            raise TypeError("Hybrid evaluator requires a CandidateChange")
        source = candidate.source
        policy = target_policy or self.target_policy
        if policy.identity != self.target_policy.identity:
            raise RdmUnsupported("target-policy-mismatch")

        document_trial = None
        document_observation = None
        document_verdict = None
        compiler_trial = None
        compiler_evaluation = None
        compiler_verdict = None
        reason = self.document_evaluator.unsupported_reason(context)
        if reason is None:
            try:
                reason = self._ensure_reference_baseline(context, policy)
            except BaseException as exc:
                reason = "reference-baseline-unavailable: %s" % exc
                self.document_evaluator.disable(context, reason)
        if reason is None:
            try:
                document_trial = self.document_evaluator.begin(context, candidate)
                document_observation = document_trial.observation
                if document_observation.status == EvaluationStatus.PARSE_ERROR:
                    reason = "parser-target-is-compiler-only"
                    self.document_evaluator.finish(document_trial, False)
                    document_trial = None
                    document_verdict = None
                else:
                    document_verdict = self._predicate(
                        policy,
                        context.role,
                        _observation_as_evaluation(document_observation),
                    )
            except (RdmError, OSError, EOFError) as exc:
                reason = "%s: %s" % (type(exc).__name__, exc)
                document_trial = None
                document_observation = None
                self._log("rdm-hybrid compiler fallback: %s" % reason, level=1)

        audited = False
        if document_verdict is False:
            audited = self._audit_rejection(context)
            if not audited:
                document_evaluation = _observation_as_evaluation(
                    document_observation
                )
                details = self._details(
                    "fast_reject",
                    document_observation,
                    False,
                    None,
                    None,
                    reason,
                    False,
                    self.document_evaluator.identity,
                )
                return EvaluationTrial(
                    self._copy_with_details(document_evaluation, details),
                    HybridTrialToken(
                        "fast_reject",
                        source,
                        context,
                        context.role,
                        policy.identity,
                        None,
                        document_trial,
                        document_observation,
                        False,
                        None,
                        reason,
                        False,
                    ),
                    False,
                    False,
                )

        route = (
            "audited_reject"
            if audited
            else ("reference_confirm" if document_trial is not None else "compiler_fallback")
        )
        try:
            compiler_trial = self.compiler_evaluator.begin(context, candidate, policy)
            compiler_evaluation = compiler_trial.evaluation
            compiler_verdict = self._predicate(
                policy, context.role, compiler_evaluation
            )
        except BaseException:
            if document_trial is not None:
                try:
                    self.document_evaluator.finish(document_trial, False)
                except BaseException:
                    pass
            raise

        if (
            document_verdict is not None
            and document_verdict != compiler_verdict
        ):
            direction = (
                "false-negative" if not document_verdict else "false-positive"
            )
            reason = "document-%s" % direction
            self.document_evaluator.disable(context, reason)

        details = self._details(
            route,
            document_observation,
            document_verdict,
            compiler_evaluation,
            compiler_verdict,
            reason,
            audited,
            self.document_evaluator.identity,
        )
        return EvaluationTrial(
            self._copy_with_details(compiler_evaluation, details),
            HybridTrialToken(
                route,
                source,
                context,
                context.role,
                policy.identity,
                compiler_trial,
                document_trial,
                document_observation,
                document_verdict,
                compiler_verdict,
                reason,
                audited,
            ),
            bool(
                compiler_verdict
                and document_trial is not None
                and document_verdict
                and reason is None
            ),
            False,
        )

    def acceptance_authorized(self, trial, target_policy, role):
        token = trial.token
        return bool(
            isinstance(token, HybridTrialToken)
            and token.compiler_trial is not None
            and token.source is not None
            and token.role == role
            and token.policy_identity == target_policy.identity
            and token.compiler_verdict
        )

    def finish(self, trial, accepted):
        token = trial.token
        if not isinstance(token, HybridTrialToken):
            raise CandidateLifecycleError("Invalid hybrid trial token")
        if accepted and not (
            token.compiler_trial is not None and token.compiler_verdict
        ):
            if token.document_trial is not None:
                try:
                    self.document_evaluator.finish(token.document_trial, False)
                except BaseException:
                    pass
            raise CandidateLifecycleError(
                "Hybrid acceptance lacks compiler authorization"
            )
        if token.compiler_trial is not None:
            try:
                self.compiler_evaluator.finish(token.compiler_trial, accepted)
            except BaseException:
                if token.document_trial is not None:
                    try:
                        self.document_evaluator.finish(token.document_trial, False)
                    except BaseException:
                        pass
                raise
        if token.document_trial is not None:
            promote = bool(
                accepted
                and token.document_verdict
                and token.compiler_verdict
                and token.reason is None
            )
            try:
                self.document_evaluator.finish(token.document_trial, promote)
                if token.reason is not None:
                    self.document_evaluator.disable(
                        token.context, token.reason
                    )
                if accepted and not promote:
                    self.document_evaluator.advance_accepted_source(token.source)
            except BaseException as exc:
                if accepted:
                    # The already-written compiler-confirmed source is the
                    # recovery checkpoint even if document promotion was lost.
                    try:
                        self.document_evaluator.advance_accepted_source(token.source)
                    except BaseException:
                        pass
                    self._log(
                        "rdm-hybrid state discarded after promotion failure: %s"
                        % exc,
                        level=1,
                    )
                else:
                    raise
        elif accepted:
            self.document_evaluator.advance_accepted_source(token.source)
        if accepted:
            self._baseline_confirmed[token.context] = token.source

    def record_target_decision(self, trial, target_policy, role):
        token = trial.token
        if not isinstance(token, HybridTrialToken):
            return
        record = {
            "schema": 1,
            "role": role,
            "route": token.route,
            "document_status": (
                None
                if token.document_observation is None
                else token.document_observation.status
            ),
            "document_verdict": token.document_verdict,
            "compiler_ran": token.compiler_trial is not None,
            "compiler_verdict": token.compiler_verdict,
            "agreement": (
                None
                if token.document_verdict is None or token.compiler_verdict is None
                else token.document_verdict == token.compiler_verdict
            ),
            "audited": token.audited,
            "reason": token.reason,
        }
        self._log(
            "rdm-hybrid: %s"
            % json.dumps(record, sort_keys=True, separators=(",", ":")),
            level=1,
        )
        document = token.document_observation
        pool = getattr(self.document_evaluator, "pool", None)
        compiler = (
            None
            if token.compiler_trial is None
            else token.compiler_trial.evaluation
        )
        artifact = dict(record)
        artifact.update(
            {
                "event": "candidate-comparison",
                "mode": "rdm-hybrid",
                "source_sha256": _sha256_text(token.source),
                "source_bytes": len(token.source.encode("utf-8")),
                "context": _context_record(token.context),
                "backend_identity": self.document_evaluator.identity,
                "compiler_status": None if compiler is None else compiler.status,
                "compiler_runtime": None if compiler is None else compiler.runtime,
                "compiler_peak_rss_kb": (
                    None if compiler is None else compiler.peak_rss_kb
                ),
                "document_runtime": None if document is None else document.runtime,
                "document_split_runtime": (
                    None if document is None else document.split_runtime
                ),
                "document_execution_runtime": (
                    None if document is None else document.execution_runtime
                ),
                "reused_items": None if document is None else document.reused_items,
                "commands_replayed": (
                    None if document is None else document.processed_items
                ),
                "candidate_items": None if document is None else document.candidate_items,
                "edit_strategy": None if document is None else document.edit_strategy,
                "edit_kind": None if document is None else document.edit_kind,
                "replaced_items": None if document is None else document.replaced_items,
                "diagnostic_agreement": (
                    None
                    if document is None or compiler is None
                    else _normalize_output(compiler.output)
                    == _normalize_output(document.output)
                ),
                "restart_count": getattr(pool, "restart_count", None),
                "live_session_count": getattr(pool, "session_count", None),
                "live_cursor_count": getattr(pool, "cursor_count", None),
            }
        )
        _append_jsonl(getattr(self, "_jsonl_path", None), artifact)

    def reset_calibration(self, context=None):
        self.compiler_evaluator.reset_calibration(context)
        self._reject_counts.clear()
        self._baseline_confirmed.clear()
        try:
            self.document_evaluator.reset()
        except BaseException as exc:
            self._log("rdm-hybrid reset failed: %s" % exc, level=1)

    def close(self):
        if self._closed:
            return
        first_error = None
        try:
            self.document_evaluator.close()
        except BaseException as exc:
            first_error = exc
        try:
            self.compiler_evaluator.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error
