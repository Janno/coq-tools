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
import shutil
import subprocess
import tempfile
import time
from collections import namedtuple

from .candidate_evaluator import (
    CandidateEvaluator,
    CandidateLifecycleError,
    CoqcEvaluator,
    Evaluation,
    EvaluationStatus,
    EvaluationTrial,
)


DEFAULT_MAX_FRAME_SIZE = 64 * 1024 * 1024


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
        "cursor generation notifications"
    ),
)
RdmSessionTrial = namedtuple(
    "RdmSessionTrial", "session generation cursor source observation promotable"
)
ShadowTrialToken = namedtuple(
    "ShadowTrialToken",
    "compiler_trial document_trial document_observation unsupported_reason",
)

REQUIRED_METHODS = frozenset(
    (
        "load_file",
        "doc_prefix",
        "doc_suffix",
        "clone",
        "go_to",
        "revert_before",
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
    ):
        self.command = tuple(command)
        self.cwd = cwd
        self._environment = None if environment is None else dict(environment)
        self._max_frame_size = max_frame_size
        self._process = None
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        self._notifications = []
        self._next_id = 0
        self._in_request = False
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
            )
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

    def _receive(self):
        if self._process is None or self._process.stdout is None:
            raise JsonRpcError("JSON-RPC process is not running")
        try:
            return _validate_packet(
                read_json_rpc_frame(
                    self._process.stdout, max_frame_size=self._max_frame_size
                )
            )
        except JsonRpcFramingError as exc:
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
        while True:
            packet = self._receive()
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

    def _send(self, packet):
        self._ensure_usable()
        if self._process.stdin is None:
            raise JsonRpcError("JSON-RPC stdin is unavailable")
        try:
            self._process.stdin.write(encode_json_rpc_frame(packet))
            self._process.stdin.flush()
        except (IOError, OSError) as exc:
            self._usable = False
            code = self._poll_after_eof()
            raise JsonRpcProcessError(
                "Failed to write JSON-RPC request: %s" % exc,
                returncode=code,
                stderr_tail=self._stderr_tail(),
            )

    def request(self, method, params):
        self._ensure_usable()
        if self._in_request:
            raise JsonRpcProtocolError("Concurrent or reentrant request is unsupported")
        if not isinstance(method, str) or not isinstance(params, (list, tuple)):
            raise TypeError("JSON-RPC method must be text and params must be a sequence")
        request_id = self._next_id
        self._next_id += 1
        self._in_request = True
        try:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": list(params),
                }
            )
            while True:
                packet = self._receive()
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

    def _cleanup_after_startup_failure(self):
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=0.5)
            except BaseException:
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except BaseException:
                    pass
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
                    process.terminate()
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
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


def _default_client_factory(command, cwd, environment):
    return RdmClient(
        JsonRpcProcess(command, cwd=cwd, environment=environment)
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
    ):
        self.manager_command = tuple(manager_command)
        self.context = context
        self.accepted_source = accepted_source
        self.generation = generation
        self._client_factory = client_factory or _default_client_factory
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
        self.client = self._client_factory(
            tuple(command),
            self.context.cwd,
            self.context.environment.as_dict(),
        )
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

    @property
    def private_file(self):
        return self._private_file

    @property
    def active_trial_count(self):
        return len(self._active)

    def _notifications(self):
        transport = getattr(self.client, "transport", None)
        return tuple(getattr(transport, "notifications", ()))

    def _run_cursor(self, cursor, source, replay_item, split_runtime):
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
            cursor,
            self.generation,
            self._notifications(),
        )

    def _canonical_items(self):
        return self.client.doc_prefix(self.canonical_cursor) + self.client.doc_suffix(
            self.canonical_cursor
        )

    def begin(self, source):
        if self._closed:
            raise RdmUnavailable("Document session is closed")
        cursor = self.client.clone(self.canonical_cursor)
        try:
            items = self._canonical_items()
            replay_item, source_offset = find_item_boundary(
                items, self.accepted_source, source
            )
            # A canonical error cursor cannot advance through its failing item.
            # If the textual edit begins later, replay from the current cursor
            # and reinstall the failing item together with the changed tail.
            canonical_index = len(self.client.doc_prefix(self.canonical_cursor))
            if replay_item > canonical_index:
                replay_item = canonical_index
                source_offset = sum(
                    len(item.text) for item in items[:canonical_index]
                )
            self.client.go_to(cursor, replay_item)
            split_start = time.time()
            try:
                sentences = self.client.replace_suffix(
                    cursor, source[source_offset:], None
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
            )
            trial = RdmSessionTrial(
                self, self.generation, cursor, source, observation, True
            )
            self._active[cursor] = trial
            return trial
        except BaseException:
            try:
                self.client.dispose(cursor)
            except BaseException:
                pass
            raise

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
    ):
        self.primary_manager = tuple(primary_manager)
        self.passing_manager = tuple(passing_manager or primary_manager)
        self.accepted_source = accepted_source
        self.target_policy = target_policy
        self.restart_every = int(restart_every)
        if self.restart_every < 0:
            raise ValueError("restart_every must not be negative")
        self._client_factory = client_factory
        self._log = log or (lambda *args, **kwargs: None)
        self._sessions = {}
        self._generation = 0
        self._attempts = {}
        self.restart_count = 0
        self._closed = False

    @property
    def session_count(self):
        return len(self._sessions)

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

    def begin(self, context, source):
        try:
            session = self._session(context)
            self._attempts[context] = self._attempts.get(context, 0) + 1
            return session.begin(source)
        except (JsonRpcError, OSError, EOFError):
            try:
                self._drop(context)
            except BaseException:
                pass
            self.restart_count += 1
            session = self._new_session(context)
            self._attempts[context] = 1
            return session.begin(source)

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
    """Stateful document engine used observationally by the shadow adapter."""

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
    ):
        self.primary_manager = tuple(primary_manager)
        self.passing_manager = tuple(passing_manager or primary_manager)
        self.target_policy = target_policy
        self._log = log or (lambda *args, **kwargs: None)
        self._client_factory = client_factory
        self._restart_every = restart_every
        self._disabled = {}
        self._closed = False
        self.primary_probe = None
        self.passing_probe = None
        self.probe_errors = ()
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
        return self.passing_probe if context.role == "passing" else self.primary_probe

    def unsupported_reason(self, context):
        if self._closed:
            return "document evaluator is closed"
        if context in self._disabled:
            return self._disabled[context]
        probe = self._probe_for(context)
        if probe is None:
            return "document-manager capability probe failed"
        if probe.missing_methods:
            return "missing RPC methods: %s" % ", ".join(probe.missing_methods)
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

    def begin(self, context, source):
        reason = self.unsupported_reason(context)
        if reason is not None:
            raise RdmUnsupported(reason)
        try:
            return self.pool.begin(context, source)
        except (RdmError, OSError, EOFError) as exc:
            reason = "%s: %s" % (type(exc).__name__, exc)
            self._disabled[context] = reason
            raise RdmUnavailable(reason)

    def finish(self, trial, accepted):
        return self.pool.finish(trial, accepted)

    def reset(self):
        if self._closed:
            raise RdmUnavailable("Document evaluator is closed")
        accepted_source = self.pool.accepted_source
        self.pool.close()
        self._disabled.clear()
        self.pool = RdmSessionPool(
            self.primary_manager,
            accepted_source,
            self.target_policy,
            passing_manager=self.passing_manager,
            restart_every=self._restart_every,
            client_factory=self._client_factory,
            log=self._log,
        )

    def close(self):
        if self._closed:
            return
        self.pool.close()
        self._closed = True


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
    ):
        if not isinstance(compiler_evaluator, CoqcEvaluator):
            # Tests may supply a compatible fake, so require behavior rather
            # than an exact type after this lightweight sanity check.
            for name in ("materialize_context", "begin", "finish", "close"):
                if not hasattr(compiler_evaluator, name):
                    raise TypeError("compiler evaluator lacks %s" % name)
        self.compiler_evaluator = compiler_evaluator
        self._log = log or (lambda *args, **kwargs: None)
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
            ("cursor", observation.cursor),
            ("generation", observation.generation),
        )

    def begin(self, context, source):
        if self._closed:
            raise CandidateLifecycleError("Shadow evaluator is closed")
        document_trial = None
        document_observation = None
        reason = self.document_evaluator.unsupported_reason(context)
        if reason is None:
            try:
                document_trial = self.document_evaluator.begin(context, source)
                document_observation = document_trial.observation
            except (RdmError, OSError, EOFError) as exc:
                reason = "%s: %s" % (type(exc).__name__, exc)
                self._log(
                    "rdm-shadow unavailable: %s" % reason,
                    level=1,
                )
        compiler_trial = None
        try:
            compiler_trial = self.compiler_evaluator.begin(context, source)
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
