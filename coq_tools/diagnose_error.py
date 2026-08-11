from __future__ import with_statement, print_function
import os, tempfile, subprocess, re, time, math, threading, shutil, traceback, sys
from collections import namedtuple
from subprocess4 import Popen
from .memoize import memoize
from .file_util import clean_v_file
from .util import re_escape, get_peak_rss_bytes, apply_memory_limit, cleanup_cgroup
from .custom_arguments import LOG_ALWAYS
from .coq_version import (
    group_coq_args,
    get_coqc_help,
    get_coq_accepts_Q,
)
from . import util

__all__ = [
    "has_error",
    "get_error_line_number",
    "get_error_byte_locations",
    "make_reg_string",
    "get_coq_output",
    "get_error_string",
    "get_timeout",
    "get_timeout_snapshot",
    "get_memory_usage_snapshot",
    "reset_timeout",
    "reset_coq_output_cache",
    "is_timeout",
    "is_memory_limit",
]

ACTUAL_ERROR_REG_STRING = "(?!Warning)(?!The command has indeed failed with message:)"  # maybe we should just require Error or Timeout?
DEFAULT_PRE_PRE_ERROR_REG_STRING = 'File "[^"]+", line ([0-9-]+), characters [0-9-]+:\n'
DEFAULT_PRE_ERROR_REG_STRING = (
    DEFAULT_PRE_PRE_ERROR_REG_STRING + ACTUAL_ERROR_REG_STRING
)
DEFAULT_PRE_ERROR_REG_STRING_WITH_BYTES = (
    'File "[^"]+", line ([0-9-]+), characters ([0-9]+)-([0-9]+):\n'
    + ACTUAL_ERROR_REG_STRING
)
DEFAULT_ERROR_REG_STRING = DEFAULT_PRE_ERROR_REG_STRING + "((?:.|\n)+)"
DEFAULT_ERROR_REG_STRING_WITH_BYTES = (
    DEFAULT_PRE_ERROR_REG_STRING_WITH_BYTES + "((?:.|\n)+)"
)
DEFAULT_ERROR_REG_STRING_GENERIC = DEFAULT_PRE_PRE_ERROR_REG_STRING + "(%s)"


def clean_output(output):
    return adjust_error_message_for_selected_errors(util.normalize_newlines(output))


def process_coqchk_output(output, v_file_name, line_count):
    """
    Process coqchk output and insert error preamble before "Fatal Error:" lines.

    When coqchk outputs a line starting with "Fatal Error:", we emit a fake Coq
    error location pointing to the end of the .v file before that line.
    """
    lines = output.split("\n")
    result_lines = []
    for line in lines:
        if line.startswith("Fatal Error:"):
            # Insert fake Coq error location before the Fatal Error line
            result_lines.append(
                f'File "{v_file_name}", line {line_count}, characters 0-0:'
            )
            result_lines.append("Error:")
        result_lines.append(line)
    return "\n".join(result_lines)


def _log_secondary_cleanup_failure(log, description):
    if log is None:
        return
    try:
        log(
            "Non-fatal error during %s:\n%s" % (description, traceback.format_exc()),
            level=LOG_ALWAYS,
        )
    except BaseException:
        pass


def _run_cleanup(cleaner, log, description, preserve_active_exception):
    if cleaner is None:
        return
    try:
        cleaner()
    except BaseException:
        if not preserve_active_exception:
            raise
        _log_secondary_cleanup_failure(log, description)


def _stop_native_debug_probe_process(process, log):
    """Best-effort stop and reap without replacing an active probe failure."""
    try:
        process.terminate()
    except BaseException:
        _log_secondary_cleanup_failure(log, "native-debug probe termination")
    try:
        process.wait(timeout=1)
        return
    except subprocess.TimeoutExpired:
        pass
    except BaseException:
        _log_secondary_cleanup_failure(log, "native-debug probe reap")
        return
    try:
        process.kill()
    except BaseException:
        _log_secondary_cleanup_failure(log, "native-debug probe kill")
    try:
        process.wait()
    except BaseException:
        _log_secondary_cleanup_failure(log, "native-debug probe reap after kill")


def _probe_coq_accepts_fine_grained_debug(
    coqc,
    debug_kind,
    cwd=None,
    process_environment=None,
    log=None,
):
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".v", dir=(cwd if cwd is not None else "."), delete=False
    )
    temp_file_name = os.path.abspath(temp_file.name)
    try:
        temp_file.close()
        coqc_command = list(coqc) if isinstance(coqc, tuple) else [coqc]
        process = subprocess.Popen(
            coqc_command + ["-q", "-d", debug_kind, temp_file_name],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            cwd=cwd,
            env=(
                None
                if process_environment is None
                else dict(process_environment)
            ),
        )
        try:
            (stdout, stderr) = process.communicate()
        except BaseException:
            _stop_native_debug_probe_process(process, log)
            raise
        return (
            "Unknown option -d" not in util.s(stdout)
            and "-d: no such file or directory" not in util.s(stdout)
            and "There is no debug flag" not in util.s(stdout)
        )
    finally:
        _run_cleanup(
            lambda: clean_v_file(temp_file_name),
            log,
            "native-debug capability-probe cleanup",
            sys.exc_info()[0] is not None,
        )


_FINE_GRAINED_DEBUG_CAPABILITY_CACHE = {}


def get_coq_accepts_fine_grained_debug(
    coqc,
    debug_kind,
    cwd=None,
    process_environment=None,
    log=None,
    executable_identity=None,
):
    # The executable fingerprint is supplied by candidate contexts.  Legacy
    # callers omit it and retain their historical command/context cache key.
    process_environment_items = (
        None
        if process_environment is None
        else tuple(sorted(dict(process_environment).items()))
    )
    cache_key = (
        tuple(coqc) if isinstance(coqc, (list, tuple)) else coqc,
        debug_kind,
        cwd,
        process_environment_items,
        executable_identity,
    )
    if cache_key not in _FINE_GRAINED_DEBUG_CAPABILITY_CACHE:
        _FINE_GRAINED_DEBUG_CAPABILITY_CACHE[cache_key] = (
            _probe_coq_accepts_fine_grained_debug(
                coqc,
                debug_kind,
                cwd=cwd,
                process_environment=process_environment,
                log=log,
            )
        )
    return _FINE_GRAINED_DEBUG_CAPABILITY_CACHE[cache_key]


def get_coq_debug_native_compiler_args(
    coqc,
    cwd=None,
    process_environment=None,
    log=None,
    executable_identity=None,
):
    if get_coq_accepts_fine_grained_debug(
        coqc,
        "native-compiler",
        cwd=cwd,
        process_environment=process_environment,
        log=log,
        executable_identity=executable_identity,
    ):
        return ["-d", "native-compiler"]
    return ["-debug"]


@memoize
def get_error_match(
    output,
    reg_string=DEFAULT_ERROR_REG_STRING,
    pre_reg_string=DEFAULT_PRE_ERROR_REG_STRING,
):
    """Returns the final match of reg_string"""
    locations = [0] + [m.start() for m in re.finditer(pre_reg_string, output)]
    reg = re.compile(reg_string)
    results = (reg.search(output[start_loc:]) for start_loc in reversed(locations))
    for result in results:
        if result:
            return result
    return None


@memoize
def has_error(
    output,
    reg_string=DEFAULT_ERROR_REG_STRING,
    pre_reg_string=DEFAULT_PRE_ERROR_REG_STRING,
):
    """Returns True if the coq output encoded in output has an error
    matching the given regular expression, False otherwise.
    """
    errors = get_error_match(
        output, reg_string=reg_string, pre_reg_string=pre_reg_string
    )
    if errors:
        return True
    else:
        return False


TIMEOUT_POSTFIX = "\nTimeout! (external)"
MEMORY_LIMIT_POSTFIXES = (
    "\nFatal error: Not enough heap memory to reserve minor heaps",
    "\nFatal error: not enough memory",
    "\nFatal error: out of memory.",
    "\nFatal error: out of memory",
)


@memoize
def is_timeout(output):
    """Returns True if the output was killed with a timeout, False otherwise"""
    return output.endswith(TIMEOUT_POSTFIX)


@memoize
def is_memory_limit(output):
    """Returns True if the output was killed because of a memory limit, False otherwise"""
    output = output.strip()
    return any(
        output.endswith(postfix) or output == postfix.lstrip("\n")
        for postfix in MEMORY_LIMIT_POSTFIXES
    )


def adjust_error_message_for_selected_errors(
    output,
    *,
    file_name: str = "default.v",
    line_number: int = -1,
    characters: str = "0-0",
):
    """Sometimes errors don't come with the "File ..." lines, so we add them for selected errors"""
    prefix = f'\nAUTOMATICALLY INSERTED ERROR LINE\nFile "{file_name}", line {line_number}, characters {characters}:\nError:\n'
    # Find the last occurrence of MEMORY_LIMIT_POSTFIX and insert prefix before it
    for postfix in MEMORY_LIMIT_POSTFIXES:
        last_index = output.rfind(postfix)
        if last_index != -1:
            return output[:last_index] + prefix + output[last_index:].lstrip("\n")
        if output.startswith(postfix.strip("\n")):
            return prefix + output.lstrip("\n")
    return output


@memoize
def get_error_line_number(
    output,
    reg_string=DEFAULT_ERROR_REG_STRING,
    pre_reg_string=DEFAULT_PRE_ERROR_REG_STRING,
):
    """Returns the line number that the error matching reg_string
    occured on.

    Precondition: has_error(output, reg_string)
    """
    errors = get_error_match(
        output, reg_string=reg_string, pre_reg_string=pre_reg_string
    )
    return int(errors.groups()[0])


@memoize
def get_error_byte_locations(
    output,
    reg_string=DEFAULT_ERROR_REG_STRING_WITH_BYTES,
    pre_reg_string=DEFAULT_PRE_ERROR_REG_STRING_WITH_BYTES,
):
    """Returns the byte locations that the error matching reg_string
    occured on.

    Precondition: has_error(output, reg_string)
    """
    errors = get_error_match(
        output, reg_string=reg_string, pre_reg_string=pre_reg_string
    )
    return (int(errors.groups()[1]), int(errors.groups()[2]))


@memoize
def get_error_string(
    output,
    reg_string=DEFAULT_ERROR_REG_STRING,
    pre_reg_string=DEFAULT_PRE_ERROR_REG_STRING,
):
    """Returns the error string of the error matching reg_string.

    Precondition: has_error(output, reg_string)
    """
    errors = get_error_match(
        output, reg_string=reg_string, pre_reg_string=pre_reg_string
    )
    return errors.groups()[1]


@memoize
def make_reg_string(output, strict_whitespace=False):
    """Returns a regular expression for matching the particular error
    in output.

    Precondition: has_error(output)
    """
    unstrictify_whitespace = lambda s: s
    if not strict_whitespace:
        unstrictify_whitespace = (
            lambda s: re.sub(
                r"(?:\\s\\+)+",
                r"\\s+",
                re.sub(
                    r"(?:\\ )+",
                    r"\\s+",
                    re.sub(r"(\\n|\n)(?:\\ )+", r"\\s+", s.replace("\\\n", "\n")),
                ),
            )
            .replace("\n", r"\s")
            .replace(r"\s+\s", r"\s+")
            .replace(r"\s++", r"\s+")
        )

    error_string = get_error_string(output).strip()
    if not util.PY3:
        error_string = error_string.decode("utf-8")
    if "Anomaly" in error_string and re.search(
        r"Constant [^\s]+\s+does not appear in the environment", error_string
    ):
        re_string = re.sub(
            r"(Constant\\ )[^\s]+(\\ )", r"\1[^\\s]+\2", re_escape(error_string)
        )
    elif "Anomaly" in error_string and re.search(
        r"Universe [^\s]+ undefined", error_string
    ):
        re_string = re.sub(
            r"(Universe\\ )[^\s]+(\\ )", r"\1[^\\s]+\2", re_escape(error_string)
        )
    elif (
        "Universe inconsistency" in error_string
        or "universe inconsistency" in error_string
    ):
        re_string = re.sub(
            r"([Uu]niverse\\ inconsistency.*) because(.|\n)*",
            r"\1 because.*",
            re_escape(error_string),
        )
        re_string = re.sub(r"(\s)[^\s]+?\.([0-9]+)", r"\1[^\\s]+?\\.\2", re_string)
        re_string = re.sub(
            r"([Uu]niverse\\ inconsistency(?::|\\.)*)([\s\S]*)",
            lambda m: m.group(1) + re.sub(
                r"\w+(?:\\\.\w+)+", r"[A-Za-z_\\d\\.]+", m.group(2)
            ),
            re_string,
        )
    elif "Unsatisfied constraints" in error_string:
        re_string = re.sub(
            r"Error:\\ Unsatisfied\\ constraints:.*(?:\n.+)*.*\\\(maybe\\ a\\ bugged\\ tactic\\\)",
            r"Error:\\ Unsatisfied\\ constraints:.*(?:\\n.+)*.*\\(maybe\\ a\\ bugged\\ tactic\\)",
            re_escape(error_string),
            re.DOTALL,
        )
    elif re.search(r"Universe [^ ]* is unbound", error_string):
        re_string = re.sub(
            r"Universe\\ [^ ]*\\ is\\ unbound",
            r"Universe\\ [^ ]*\\ is\\ unbound",
            re_escape(error_string),
        )
    elif re.search(r"Compilation of file /tmp/[^ ]*", error_string):
        re_string = re.sub(
            r"Compilation\\ of\\ file\\ /tmp/[^ \.]*\\\.",
            r"Compilation\\ of\\ file\\ /tmp/[^ \.]*\\.",
            re_escape(error_string),
        )
    else:
        re_string = re_escape(error_string)
    re_string = re.sub(r"tmp(?:[A-Za-z_\d]|\\_)+", r"[A-Za-z_\\d\\.]+", re_string)
    if r"Universe\ instance\ should\ have\ length\ " not in re_string:
        re_string = re.sub(r"[\d]+", r"[\\d]+", re_string)
    ret = DEFAULT_ERROR_REG_STRING_GENERIC % unstrictify_whitespace(re_string)
    if not util.PY3:
        ret = ret.encode("utf-8")
    return ret


TIMEOUT = {}
MEMORY_USAGE = {}


def get_timeout(coqc=None):
    return TIMEOUT.get(coqc)


def get_timeout_snapshot(coqc=None):
    """Return the current timeout calibration without exposing mutable state."""
    return TIMEOUT.get(coqc)


def set_timeout(key, value, **kwargs):
    kwargs["log"]("\nThe timeout for %s has been set to: %d" % (key, value))
    TIMEOUT[key] = value


def reset_timeout():
    global TIMEOUT
    TIMEOUT = {}


def get_memory_usage(key):
    return MEMORY_USAGE.get(key)


def get_memory_usage_snapshot(key):
    """Return one current memory baseline without exposing mutable state."""
    return MEMORY_USAGE.get(key)


def set_memory_usage(key, value):
    if value is None:
        return
    MEMORY_USAGE[key] = value


def reset_memory_usage():
    global MEMORY_USAGE
    MEMORY_USAGE = {}


def timeout_Popen_communicate(log, *args, **kwargs):
    ret = {"value": ("", ""), "returncode": None, "rusage": None}
    timeout = kwargs.pop("timeout", None)
    input_val = kwargs.pop("input", None)
    if input_val is not None:
        input_val = input_val.encode("utf-8")

    # Extract memory limit parameters
    max_mem_rss = kwargs.pop("max_mem_rss", None)
    max_mem_as = kwargs.pop("max_mem_as", None)
    max_mem_rss_multiplier = kwargs.pop("max_mem_rss_multiplier", None)
    max_mem_as_multiplier = kwargs.pop("max_mem_as_multiplier", None)
    memory_usage_key = kwargs.pop("memory_usage_key", None)
    cgroup = kwargs.pop("cgroup", None)
    mem_limit_method = kwargs.pop("mem_limit_method", "prlimit")

    def resolve_dynamic_limit(limit_value, multiplier_value, limit_kind):
        if multiplier_value is None:
            return limit_value
        if multiplier_value <= 0:
            return limit_value
        if memory_usage_key is None:
            return None
        usage = get_memory_usage((memory_usage_key, limit_kind))
        if usage is None:
            return None
        resolved = int(math.ceil(usage * multiplier_value))
        return resolved if resolved > 0 else None

    max_mem_rss = resolve_dynamic_limit(max_mem_rss, max_mem_rss_multiplier, "rss")
    max_mem_as = resolve_dynamic_limit(max_mem_as, max_mem_as_multiplier, "as")

    def record_memory_usage(usage_bytes):
        if memory_usage_key is None or usage_bytes is None:
            return
        set_memory_usage((memory_usage_key, "rss"), usage_bytes)
        set_memory_usage((memory_usage_key, "as"), usage_bytes)

    # Get command from args
    cmd = list(args[0]) if args else []
    cg_path = None

    # Apply memory limits using the selected method
    if mem_limit_method != "none" and (max_mem_as or max_mem_rss):
        cmd, cg_path = apply_memory_limit(
            cmd,
            method=mem_limit_method,
            as_bytes=max_mem_as,
            rss_bytes=max_mem_rss,
            cgroup=cgroup,
        )
        args = (cmd,) + args[1:]

    # No preexec_fn needed anymore!
    p = Popen(*args, allow_non_posix=True, **kwargs)

    def target():
        ret["value"] = tuple(map(util.s, p.communicate(input=input_val)))
        ret["returncode"] = p.returncode
        ret["rusage"] = getattr(p, "rusage", None)

    def get_peak_rss():
        return get_peak_rss_bytes(ret["rusage"]) if ret["rusage"] else 0

    thread = threading.Thread(target=target)
    thread.start()

    thread.join(timeout)
    if not thread.is_alive():
        cleanup_cgroup(cg_path)
        peak_rss_bytes = get_peak_rss()
        record_memory_usage(peak_rss_bytes)
        return (ret["value"], ret["returncode"], peak_rss_bytes)

    p.terminate()
    thread.join()
    cleanup_cgroup(cg_path)
    peak_rss_bytes = get_peak_rss()
    record_memory_usage(peak_rss_bytes)
    return (
        tuple(map((lambda s: (s if s else "") + TIMEOUT_POSTFIX), ret["value"])),
        ret["returncode"],
        peak_rss_bytes,
    )


def memory_robust_timeout_Popen_communicate(log, *args, **kwargs):
    while True:
        try:
            return timeout_Popen_communicate(log, *args, **kwargs)
        except OSError as e:
            log(
                "Warning: subprocess.Popen%s%s failed with %s\nTrying again in 10s"
                % (repr(tuple(args)), repr(kwargs), repr(e)),
                force_stdout=True,
                level=LOG_ALWAYS,
            )
            time.sleep(10)


COQ_OUTPUT = {}


def sanitize_cmd(cmd):
    return re.sub(
        r'("/tmp/tmp|"/var/folders/.*?tmp)[^/"]*?((?:/[^"]*?)?(?:\.v)?")',
        r"\1XXXXXXXX\2",
        cmd,
    )


def get_filepath_of_coq_args(coqc_prog, coqc_prog_args, **kwargs):
    coqc_help = get_coqc_help(coqc_prog, **kwargs)
    for arg_group in group_coq_args(coqc_prog_args, coqc_help):
        if arg_group[0] == "-top":
            ret = arg_group[1].split(".")
            return ret[:-1], ret[-1] + ".v"
    return None, None


prepare_cmds_for_coq_output_printed_cmd_already = set()


class SubprocessStageResult(
    namedtuple(
        "SubprocessStageResultBase",
        (
            "role command effective_deadline max_mem_rss max_mem_as "
            "returncode runtime peak_rss_kb memory_updates calibration_update"
        ),
    )
):
    """Immutable metadata for one compiler or checker subprocess."""

    __slots__ = ()

    def __new__(
        cls,
        role,
        command,
        effective_deadline,
        max_mem_rss,
        max_mem_as,
        returncode,
        runtime,
        peak_rss_kb,
        memory_updates=(),
        calibration_update=None,
    ):
        return super(SubprocessStageResult, cls).__new__(
            cls,
            role,
            tuple(command),
            effective_deadline,
            max_mem_rss,
            max_mem_as,
            returncode,
            runtime,
            peak_rss_kb,
            tuple(memory_updates),
            calibration_update,
        )


class CoqOutputResult(
    namedtuple(
        "CoqOutputResultBase",
        "output commands returncode runtime peak_rss_kb stages debug_results",
    )
):
    """Private structured execution result with an exact legacy conversion."""

    __slots__ = ()

    def __new__(
        cls,
        output,
        commands,
        returncode,
        runtime,
        peak_rss_kb,
        stages=(),
        debug_results=(),
    ):
        return super(CoqOutputResult, cls).__new__(
            cls,
            output,
            tuple(commands),
            returncode,
            runtime,
            peak_rss_kb,
            tuple(stages),
            tuple(debug_results),
        )

    def as_legacy_tuple(self):
        return (
            self.output,
            self.commands,
            self.returncode,
            self.runtime,
            self.peak_rss_kb,
        )

    @classmethod
    def from_legacy_tuple(cls, value):
        return cls(value[0], value[1], value[2], value[3], value[4])


def default_retry_with_debug_when(output):
    return "is not a compiled interface for this version of OCaml" in output


_UNSET = object()


def prepare_cmds_for_coq_output(
    coqc_prog,
    coqc_prog_args,
    contents,
    cwd=None,
    timeout_val=0,
    ocamlpath=None,
    use_cache=True,
    process_environment=None,
    log_cwd=_UNSET,
    **kwargs,
):
    """Prepare one execution and transfer temporary-source ownership.

    When ``use_cache`` is false a fresh source is always created.  Any failure
    after source allocation but before this function returns cleans the owned
    source before propagating the original exception.
    """

    def make_rmtree_onerror(file_name):
        def rmtree_onerror(function, path, exc_info):
            try:
                kwargs["log"](
                    "Non-fatal error encountered when cleaning up %s:\n" % file_name
                )
                etype, value, tb = exc_info
                if hasattr(traceback, "TracebackException"):
                    kwargs["log"](
                        "".join(
                            traceback.TracebackException(
                                type(value), value, tb, capture_locals=True
                            ).format()
                        )
                    )
                else:
                    kwargs["log"](
                        traceback.format_exception(type(value), value, tb)
                    )
            except BaseException:
                # Cleanup diagnostics must never turn a best-effort rmtree
                # callback into a new primary failure.
                pass

        return rmtree_onerror

    key = (
        coqc_prog,
        tuple(coqc_prog_args),
        kwargs["pass_on_stdin"],
        contents,
        timeout_val,
        cwd,
        ocamlpath,
    )
    if process_environment is not None:
        key = key + (
            (
                "process_environment",
                tuple(
                    sorted(
                        (str(key), str(value))
                        for key, value in dict(process_environment).items()
                    )
                ),
            ),
        )
    assert isinstance(coqc_prog, tuple), coqc_prog
    cmds = list(coqc_prog) + list(coqc_prog_args)
    cleaner = None
    if use_cache and key in COQ_OUTPUT:
        file_name = COQ_OUTPUT[key][0]
        cleaner = lambda: None
    else:
        try:
            intermediate_dirs, topfilename = get_filepath_of_coq_args(
                coqc_prog, coqc_prog_args, **kwargs
            )
            if topfilename is None:
                source_file = tempfile.NamedTemporaryFile(
                    suffix=".v", delete=False, mode="wb"
                )
                file_name = source_file.name
                cleaner = lambda file_name=file_name: clean_v_file(file_name)
                try:
                    source_file.write(contents.encode("utf-8"))
                except BaseException:
                    _run_cleanup(
                        source_file.close,
                        kwargs.get("log"),
                        "temporary source handle cleanup",
                        True,
                    )
                    raise
                else:
                    source_file.close()
            else:
                temp_dir_name = tempfile.mkdtemp()
                cleaner = lambda: shutil.rmtree(
                    temp_dir_name,
                    onerror=make_rmtree_onerror(
                        os.path.join(temp_dir_name, topfilename)
                    ),
                )
                file_path = os.path.join(temp_dir_name, *intermediate_dirs)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                file_name = os.path.join(file_path, topfilename)
                with open(file_name, mode="wb") as source_file:
                    source_file.write(contents.encode("utf-8"))
                if get_coq_accepts_Q(coqc_prog, **kwargs):
                    cmds.extend(["-Q", temp_dir_name, ""])
                else:
                    # Bind the entire tree; use -R for Coq 8.4 compatibility.
                    cmds.extend(["-R", temp_dir_name, ""])
        except BaseException:
            _run_cleanup(
                cleaner,
                kwargs.get("log"),
                "temporary source preparation cleanup",
                True,
            )
            raise

    try:
        file_name_root = os.path.splitext(file_name)[0]
        pseudocmds = ""
        input_val = None
        if kwargs["is_coqtop"]:
            if kwargs["pass_on_stdin"]:
                input_val = contents
                cmds.extend(["-q"])
                pseudocmds = '" < "%s' % file_name
            else:
                cmds.extend(["-load-vernac-source", file_name_root, "-q"])
        else:
            cmds.extend([file_name, "-q"])
        cmd_to_print = '"%s%s"' % ('" "'.join(cmds), pseudocmds)
        displayed_cwd = cwd if log_cwd is _UNSET else log_cwd
        extra_cmd = "" if displayed_cwd is None else " (in: %s)" % displayed_cwd
        kwargs["log"](
            "\nRunning command%s: %s" % (extra_cmd, cmd_to_print),
            level=(
                kwargs["verbose_base"]
                - (
                    1
                    if sanitize_cmd(cmd_to_print)
                    not in prepare_cmds_for_coq_output_printed_cmd_already
                    else 0
                )
            ),
        )
        prepare_cmds_for_coq_output_printed_cmd_already.add(
            sanitize_cmd(cmd_to_print)
        )
        kwargs["log"](
            "\nContents:\n%s\n" % contents,
            level=kwargs["verbose_base"] + 1,
        )
        env = dict(
            os.environ if process_environment is None else process_environment
        )
        if process_environment is None and ocamlpath is not None:
            env["OCAMLPATH"] = ocamlpath
        return key, file_name, cmds, input_val, cleaner, env
    except BaseException:
        _run_cleanup(
            cleaner,
            kwargs.get("log"),
            "temporary command-preparation cleanup",
            True,
        )
        raise


def reset_coq_output_cache(
    coqc_prog,
    coqc_prog_args,
    contents,
    timeout_val,
    cwd=None,
    is_coqtop=False,
    pass_on_stdin=False,
    verbose_base=1,
    ocamlpath=None,
    coqchk_prog=None,
    coqchk_prog_args=(),
    **kwargs,
):
    key_extra = (coqchk_prog, tuple(coqchk_prog_args)) if coqchk_prog else ()
    key, file_name, cmds, input_val, cleaner, env = prepare_cmds_for_coq_output(
        coqc_prog,
        coqc_prog_args,
        contents,
        cwd=cwd,
        timeout_val=timeout_val,
        is_coqtop=is_coqtop,
        pass_on_stdin=pass_on_stdin,
        verbose_base=verbose_base,
        ocamlpath=ocamlpath,
        **kwargs,
    )
    key = key + key_extra
    cleaner()

    if key in COQ_OUTPUT.keys():
        del COQ_OUTPUT[key]


def _resolve_staged_limit(fixed_limit, multiplier, usage):
    if multiplier is None or multiplier <= 0:
        return fixed_limit
    if usage is None:
        return None
    resolved = int(math.ceil(usage * multiplier))
    return resolved if resolved > 0 else None


def _stage_memory_kwargs(coqc_prog, kwargs, memory_plan, checker=False):
    """Return fixed per-stage process limits and their descriptive values."""
    if memory_plan is None:
        keys = (
            "max_mem_rss",
            "max_mem_as",
            "cgroup",
            "mem_limit_method",
            "max_mem_rss_multiplier",
            "max_mem_as_multiplier",
            "memory_usage_key",
        )
        result = {key: kwargs[key] for key in keys if key in kwargs}
        result.setdefault("memory_usage_key", coqc_prog)
        return result, result.get("max_mem_rss"), result.get("max_mem_as")

    request = memory_plan.request
    if checker:
        if request.memory_usage_key is None:
            usage_rss = None
            usage_as = None
        else:
            usage_rss = get_memory_usage((request.memory_usage_key, "rss"))
            usage_as = get_memory_usage((request.memory_usage_key, "as"))
        max_mem_rss = _resolve_staged_limit(
            request.max_mem_rss, request.max_mem_rss_multiplier, usage_rss
        )
        max_mem_as = _resolve_staged_limit(
            request.max_mem_as, request.max_mem_as_multiplier, usage_as
        )
    else:
        max_mem_rss = memory_plan.compiler_max_mem_rss
        max_mem_as = memory_plan.compiler_max_mem_as
    result = {
        "max_mem_rss": max_mem_rss,
        "max_mem_as": max_mem_as,
        "memory_usage_key": request.memory_usage_key,
        "mem_limit_method": request.memory_limit_method,
        "cgroup": request.cgroup,
    }
    # Resolved per-stage limits are fixed.  Never pass active multipliers here.
    return result, max_mem_rss, max_mem_as


def _get_coq_output_result(
    coqc_prog,
    coqc_prog_args,
    contents,
    timeout_val,
    cwd=None,
    is_coqtop=False,
    pass_on_stdin=False,
    verbose_base=1,
    retry_with_debug_when=default_retry_with_debug_when,
    ocamlpath=None,
    coqchk_prog=None,
    coqchk_prog_args=(),
    use_cache=True,
    process_environment=None,
    log_cwd=_UNSET,
    effective_timeout=_UNSET,
    should_calibrate_timeout=None,
    memory_plan=None,
    automatic_debug_retry=True,
    **kwargs,
):
    """Private structured executor used by the compiler candidate adapter."""
    if (
        effective_timeout is _UNSET
        and timeout_val is not None
        and timeout_val < 0
        and get_timeout(coqc_prog) is not None
    ):
        return _get_coq_output_result(
            coqc_prog,
            coqc_prog_args,
            contents,
            get_timeout(coqc_prog),
            cwd=cwd,
            is_coqtop=is_coqtop,
            pass_on_stdin=pass_on_stdin,
            verbose_base=verbose_base,
            retry_with_debug_when=retry_with_debug_when,
            ocamlpath=ocamlpath,
            coqchk_prog=coqchk_prog,
            coqchk_prog_args=coqchk_prog_args,
            use_cache=use_cache,
            process_environment=process_environment,
            log_cwd=log_cwd,
            effective_timeout=_UNSET,
            should_calibrate_timeout=should_calibrate_timeout,
            memory_plan=memory_plan,
            automatic_debug_retry=automatic_debug_retry,
            **kwargs,
        )

    if effective_timeout is _UNSET:
        process_deadline = (
            timeout_val if timeout_val is not None and timeout_val > 0 else None
        )
    else:
        process_deadline = effective_timeout
    if should_calibrate_timeout is None:
        calibrate = get_timeout(coqc_prog) is None and timeout_val is not None
    else:
        calibrate = bool(should_calibrate_timeout)

    key_extra = (coqchk_prog, tuple(coqchk_prog_args)) if coqchk_prog else ()
    key, file_name, cmds, input_val, cleaner, env = prepare_cmds_for_coq_output(
        coqc_prog,
        coqc_prog_args,
        contents,
        cwd=cwd,
        timeout_val=timeout_val,
        is_coqtop=is_coqtop,
        pass_on_stdin=pass_on_stdin,
        verbose_base=verbose_base,
        ocamlpath=ocamlpath,
        use_cache=use_cache,
        process_environment=process_environment,
        log_cwd=log_cwd,
        **kwargs,
    )
    key = key + key_extra
    if use_cache and key in COQ_OUTPUT:
        cleaner()
        return CoqOutputResult.from_legacy_tuple(COQ_OUTPUT[key][1])

    stages = []
    result = None
    try:
        compiler_kwargs, compiler_rss, compiler_as = _stage_memory_kwargs(
            coqc_prog, kwargs, memory_plan, checker=False
        )
        start = time.time()
        ((stdout, stderr), returncode, peak_rss_bytes) = (
            memory_robust_timeout_Popen_communicate(
                kwargs["log"],
                cmds,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                timeout=process_deadline,
                input=input_val,
                cwd=cwd,
                env=env,
                **compiler_kwargs,
            )
        )
        finish = time.time()
        runtime = finish - start
        peak_rss_kb = peak_rss_bytes / 1024
        memory_key = compiler_kwargs.get("memory_usage_key")
        updates = (
            ()
            if memory_key is None or peak_rss_bytes is None
            else (
                ((memory_key, "rss"), peak_rss_bytes),
                ((memory_key, "as"), peak_rss_bytes),
            )
        )
        stages.append(
            SubprocessStageResult(
                "compiler",
                cmds,
                process_deadline,
                compiler_rss,
                compiler_as,
                returncode,
                runtime,
                peak_rss_kb,
                updates,
            )
        )
        kwargs["log"](
            "\nretcode: %s\nstdout:\n%s\n\nstderr:\n%s\nruntime:\n%f\npeak_rss:\n%f kb\n\n"
            % (returncode, util.s(stdout), util.s(stderr), runtime, peak_rss_kb),
            level=verbose_base + 1,
        )

        combined_stdout = util.s(stdout)
        combined_returncode = returncode
        combined_runtime = runtime
        combined_peak_rss_kb = peak_rss_kb

        if coqchk_prog is not None and returncode == 0:
            vo_file = os.path.splitext(file_name)[0] + ".vo"
            assert isinstance(coqchk_prog, tuple), coqchk_prog
            coqchk_cmds = list(coqchk_prog) + list(coqchk_prog_args) + [vo_file]
            checker_kwargs, checker_rss, checker_as = _stage_memory_kwargs(
                coqc_prog, kwargs, memory_plan, checker=True
            )
            chk_start = time.time()
            ((chk_stdout, chk_stderr), chk_returncode, chk_peak_rss_bytes) = (
                memory_robust_timeout_Popen_communicate(
                    kwargs["log"],
                    coqchk_cmds,
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    timeout=process_deadline,
                    cwd=cwd,
                    env=env,
                    **checker_kwargs,
                )
            )
            chk_finish = time.time()
            chk_runtime = chk_finish - chk_start
            chk_peak_rss_kb = chk_peak_rss_bytes / 1024
            checker_key = checker_kwargs.get("memory_usage_key")
            checker_updates = (
                ()
                if checker_key is None or chk_peak_rss_bytes is None
                else (
                    ((checker_key, "rss"), chk_peak_rss_bytes),
                    ((checker_key, "as"), chk_peak_rss_bytes),
                )
            )
            stages.append(
                SubprocessStageResult(
                    "checker",
                    coqchk_cmds,
                    process_deadline,
                    checker_rss,
                    checker_as,
                    chk_returncode,
                    chk_runtime,
                    chk_peak_rss_kb,
                    checker_updates,
                )
            )
            kwargs["log"](
                "\ncoqchk retcode: %s\nstdout:\n%s\n\nstderr:\n%s\nruntime:\n%f\npeak_rss:\n%f kb\n\n"
                % (
                    chk_returncode,
                    util.s(chk_stdout),
                    util.s(chk_stderr),
                    chk_runtime,
                    chk_peak_rss_kb,
                ),
                level=verbose_base + 1,
            )
            line_count = contents.count("\n") + 1
            processed_chk_output = process_coqchk_output(
                util.s(chk_stdout), file_name, line_count
            )
            combined_stdout = combined_stdout + "\n" + processed_chk_output
            combined_returncode = (
                chk_returncode if chk_returncode != 0 else combined_returncode
            )
            combined_runtime = runtime + chk_runtime
            combined_peak_rss_kb = max(peak_rss_kb, chk_peak_rss_kb)

        if calibrate:
            calibration_runtime = (
                combined_runtime if coqchk_prog is not None else runtime
            )
            calibrated_timeout = 3 * max(
                (1, int(math.ceil(calibration_runtime)))
            )
            set_timeout(coqc_prog, calibrated_timeout, **kwargs)
            calibration_update = (coqc_prog, calibrated_timeout)
            stage_index = len(stages) - 1
            stages[stage_index] = stages[stage_index]._replace(
                calibration_update=calibration_update
            )

        result = CoqOutputResult(
            clean_output(combined_stdout),
            tuple(cmds),
            combined_returncode,
            combined_runtime,
            combined_peak_rss_kb,
            tuple(stages),
        )
    finally:
        _run_cleanup(
            cleaner,
            kwargs.get("log"),
            "temporary executor cleanup",
            sys.exc_info()[0] is not None,
        )

    if use_cache:
        COQ_OUTPUT[key] = (file_name, result.as_legacy_tuple())
        kwargs["log"](
            "Storing result: COQ_OUTPUT[%s]:\n%s"
            % (repr(key), repr(COQ_OUTPUT[key])),
            level=verbose_base + 2,
        )

    if automatic_debug_retry and retry_with_debug_when(result.output):
        debug_args = get_coq_debug_native_compiler_args(
            coqc_prog,
            cwd=cwd,
            process_environment=process_environment,
            log=kwargs.get("log"),
        )
        kwargs["log"](
            "Retrying with %s..." % " ".join(debug_args),
            level=verbose_base - 1,
        )
        retry_result = _get_coq_output_result(
            coqc_prog,
            list(debug_args) + list(coqc_prog_args),
            contents,
            timeout_val,
            cwd=cwd,
            is_coqtop=is_coqtop,
            pass_on_stdin=pass_on_stdin,
            verbose_base=verbose_base,
            retry_with_debug_when=(lambda output: False),
            ocamlpath=ocamlpath,
            coqchk_prog=coqchk_prog,
            coqchk_prog_args=coqchk_prog_args,
            use_cache=use_cache,
            process_environment=process_environment,
            log_cwd=log_cwd,
            effective_timeout=effective_timeout,
            should_calibrate_timeout=should_calibrate_timeout,
            memory_plan=memory_plan,
            automatic_debug_retry=False,
            **kwargs,
        )
        return CoqOutputResult(
            retry_result.output,
            retry_result.commands,
            retry_result.returncode,
            retry_result.runtime,
            retry_result.peak_rss_kb,
            retry_result.stages,
            (result,) + retry_result.debug_results,
        )
    return result


def get_coq_output(
    coqc_prog,
    coqc_prog_args,
    contents,
    timeout_val,
    cwd=None,
    is_coqtop=False,
    pass_on_stdin=False,
    verbose_base=1,
    retry_with_debug_when=default_retry_with_debug_when,
    ocamlpath=None,
    coqchk_prog=None,
    coqchk_prog_args=(),
    use_cache=True,
    process_environment=None,
    **kwargs,
):
    """Return the exact historical five-field compiler result tuple."""
    return _get_coq_output_result(
        coqc_prog,
        coqc_prog_args,
        contents,
        timeout_val,
        cwd=cwd,
        is_coqtop=is_coqtop,
        pass_on_stdin=pass_on_stdin,
        verbose_base=verbose_base,
        retry_with_debug_when=retry_with_debug_when,
        ocamlpath=ocamlpath,
        coqchk_prog=coqchk_prog,
        coqchk_prog_args=coqchk_prog_args,
        use_cache=use_cache,
        process_environment=process_environment,
        **kwargs,
    ).as_legacy_tuple()
