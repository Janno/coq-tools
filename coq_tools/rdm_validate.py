"""rocq-doc-manager differential corpus and benchmark runner.

The runner is deliberately standard-library-only and Python 3.6 compatible.
It executes the version-controlled example manifest in an isolated disposable
clone, retains JSONL records and logs outside that clone, benchmarks late-suffix
candidate evaluation, and evaluates rollout gates.
"""
from __future__ import print_function

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import uuid


DEFAULT_MANIFEST = os.path.join("tests", "rdm_corpus.json")
REQUIRED_CASE_FIELDS = (
    "id",
    "kind",
    "script",
    "source_revision",
    "expected_category",
    "expected_fallback",
    "platforms",
    "toolchains",
    "environment_keys",
    "archived_bug",
    "archive_provenance",
    "expected_outcome",
)
COMPARISON_EVENT = "candidate-comparison"
OUTPUT_MARKER = ".rdm-corpus-output.json"


def _json_dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _open_regular_file(path, flags, mode=0o600):
    descriptor = os.open(
        path, flags | getattr(os, "O_NOFOLLOW", 0), mode
    )
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError("Refusing to write non-regular output %s" % path)
    return descriptor


def _write_all(descriptor, payload):
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise IOError("Unable to write validation output")
        offset += written


def _append_jsonl(path, value):
    payload = (_json_dump(value) + "\n").encode("utf-8")
    descriptor = _open_regular_file(
        path, os.O_WRONLY | os.O_CREAT | os.O_APPEND
    )
    try:
        _write_all(descriptor, payload)
    finally:
        os.close(descriptor)


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError("Invalid JSONL line %d: %s" % (line_number, exc))
            if not isinstance(value, dict):
                raise ValueError("JSONL line %d is not an object" % line_number)
            records.append(value)
    return records


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command, cwd=None, environment=None, timeout=30):
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    return completed.returncode, completed.stdout.decode("utf-8", "replace").strip()


def _checked_command(command, cwd=None, environment=None, timeout=300):
    code, output = _command_output(
        command, cwd=cwd, environment=environment, timeout=timeout
    )
    if code != 0:
        raise RuntimeError(
            "Command failed with return code %r: %s\n%s"
            % (code, " ".join(command), output or "")
        )
    return output


def _git(repository, arguments):
    code, output = _command_output(["git"] + list(arguments), cwd=repository)
    if code != 0:
        return None
    return output


def _example_scripts(repository):
    return set(
        os.path.relpath(path, repository)
        for path in glob.glob(os.path.join(repository, "examples", "run-example-*.sh"))
    )


def load_manifest(path, repository=None):
    with open(path, "r", encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("Corpus manifest must be a schema-1 object")
    configuration = manifest.get("declared_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Manifest lacks declared_configuration")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest cases must be a nonempty list")
    identifiers = set()
    scripts = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("Case %d is not an object" % index)
        missing = [field for field in REQUIRED_CASE_FIELDS if field not in case]
        if missing:
            raise ValueError("Case %d lacks %s" % (index, ", ".join(missing)))
        identifier = case["id"]
        if not isinstance(identifier, str) or re.match(
            r"\Aexample-[0-9]{3}(?:-[0-9]+)?\Z", identifier
        ) is None:
            raise ValueError("Invalid case id %r" % identifier)
        if identifier in identifiers:
            raise ValueError("Duplicate case id %s" % identifier)
        identifiers.add(identifier)
        script = case["script"]
        if script in scripts:
            raise ValueError("Duplicate case script %s" % script)
        scripts.add(script)
        if not isinstance(case["platforms"], list) or not case["platforms"]:
            raise ValueError("Case %s has no platforms" % identifier)
        if not isinstance(case["toolchains"], list) or not case["toolchains"]:
            raise ValueError("Case %s has no toolchains" % identifier)
    required_archives = manifest.get("archive_policy", {}).get("required_count", 20)
    archives = [case for case in cases if case.get("archived_bug")]
    if len(archives) < required_archives:
        raise ValueError(
            "Manifest has %d archived bugs, expected at least %d"
            % (len(archives), required_archives)
        )
    if any(not case.get("archive_provenance") for case in archives):
        raise ValueError("Every archived bug must state archive_provenance")
    if repository is not None:
        actual = _example_scripts(repository)
        if scripts != actual:
            missing = sorted(actual.difference(scripts))
            stale = sorted(scripts.difference(actual))
            raise ValueError(
                "Manifest/example mismatch; missing=%r stale=%r" % (missing, stale)
            )
        for case in cases:
            if not os.path.isfile(os.path.join(repository, case["script"])):
                raise ValueError("Missing script %s" % case["script"])
    return manifest


def _resolved_executable(command, environment=None):
    if os.path.isabs(command) or os.path.dirname(command):
        return os.path.abspath(command)
    return shutil.which(command, path=(environment or os.environ).get("PATH"))


def machine_metadata(repository, manager, coqc, environment=None):
    environment = dict(os.environ if environment is None else environment)
    manager_path = _resolved_executable(manager, environment)
    coqc_path = _resolved_executable(coqc, environment)
    manager_code, manager_docs = _command_output(
        [manager, "--api-docs"], cwd=repository, environment=environment
    )
    coqc_code, coqc_version = _command_output(
        [coqc, "--version"], cwd=repository, environment=environment
    )
    try:
        load = os.getloadavg()
    except (AttributeError, OSError):
        load = None
    metadata = {
        "repository_commit": _git(repository, ["rev-parse", "HEAD"]),
        "repository_status_sha256": hashlib.sha256(
            (_git(repository, ["status", "--porcelain=v1"]) or "").encode("utf-8")
        ).hexdigest(),
        "platform": platform.platform(),
        "system": platform.system().lower(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average": load,
        "manager_path": manager_path,
        "manager_sha256": (
            _sha256_file(manager_path) if manager_path and os.path.isfile(manager_path) else None
        ),
        "manager_api_probe_returncode": manager_code,
        "manager_api_docs_sha256": (
            hashlib.sha256(manager_docs.encode("utf-8")).hexdigest()
            if manager_code == 0 and manager_docs is not None
            else None
        ),
        "coqc_path": coqc_path,
        "coqc_sha256": (
            _sha256_file(coqc_path) if coqc_path and os.path.isfile(coqc_path) else None
        ),
        "coqc_version_returncode": coqc_code,
        "coqc_version": coqc_version,
    }
    return metadata


def _write_wrapper(path, python, repository, mode, manager, passing_manager=None):
    arguments = [
        repr(python),
        repr(os.path.join(repository, "find-bug.py")),
        '"$@"',
        repr("--backend=" + mode),
    ]
    if mode != "coqc":
        arguments.append(repr("--rdm=" + manager))
        if passing_manager:
            arguments.append(repr("--passing-rdm=" + passing_manager))
    contents = "#!/bin/sh\nexec %s\n" % " ".join(arguments)
    with open(path, "w", encoding="utf-8") as output:
        output.write(contents)
    os.chmod(path, 0o755)


def _case_selected(case, requested, pattern):
    if requested and case["id"] not in requested:
        return False
    if pattern and re.search(pattern, case["id"]) is None:
        return False
    return True


def _case_tokens(case):
    identifier = case["id"].split("example-", 1)[-1]
    return identifier.split("-", 1)[0], identifier.replace("-", "_")


def _clean_case(repository, case):
    """Reset generated files inside the validator-owned disposable clone."""
    directory_token, artifact_token = _case_tokens(case)
    paths = [
        os.path.join("examples", "example_" + directory_token),
        os.path.join("examples", "example_%s_output.v" % artifact_token),
        os.path.join("examples", "example_%s_log.log" % artifact_token),
        os.path.join("examples", "example_%s_make.log" % artifact_token),
        os.path.join("examples", "example_%s_result.log" % artifact_token),
    ]
    _checked_command(["git", "clean", "-fdx", "--"] + paths, cwd=repository)


def _untracked_v_files(repository, case):
    outputs = (
        _checked_command(
            ["git", "ls-files", "--others", "--exclude-standard", "examples"],
            cwd=repository,
        ),
        _checked_command(
            [
                "git",
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "examples",
            ],
            cwd=repository,
        ),
    )
    unused_directory_token, artifact_token = _case_tokens(case)
    markers = (
        "example_%s" % artifact_token,
        "bug_%s" % artifact_token,
    )
    return sorted(
        path
        for path in set(
            item for output in outputs for item in output.splitlines()
        )
        if path.endswith(".v") and any(marker in path for marker in markers)
    )


def _retain_outputs(repository, output_directory, case):
    retained = []
    case_directory = os.path.join(output_directory, "final", case["id"])
    for relative in _untracked_v_files(repository, case):
        source = os.path.join(repository, relative)
        destination = os.path.join(case_directory, relative.replace(os.sep, "__"))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        with open(source, "rb") as value:
            contents = value.read()
        retained.append(
            {
                "source_path": relative,
                "artifact_path": os.path.relpath(destination, output_directory),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "bytes": len(contents),
                "lines": contents.count(b"\n") + (0 if not contents or contents.endswith(b"\n") else 1),
            }
        )
    return retained


def run_case(
    repository,
    output_directory,
    records_path,
    run_id,
    case,
    mode,
    manager,
    passing_manager,
    python,
    coqbin,
    timeout,
):
    _clean_case(repository, case)
    wrapper_directory = os.path.join(output_directory, "wrappers")
    os.makedirs(wrapper_directory, exist_ok=True)
    wrapper = os.path.join(wrapper_directory, case["id"] + "-find-bug")
    _write_wrapper(wrapper, python, repository, mode, manager, passing_manager)
    log_directory = os.path.join(output_directory, "logs")
    os.makedirs(log_directory, exist_ok=True)
    log_path = os.path.join(log_directory, case["id"] + ".log")
    command = ["bash", os.path.join(repository, case["script"])]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON": python,
            "FIND_BUG": wrapper,
            "COQ_TOOLS_RDM_JSONL": records_path,
            "COQ_TOOLS_RDM_CASE_ID": case["id"],
            "COQ_TOOLS_RDM_RUN_ID": run_id,
        }
    )
    if coqbin:
        normalized_coqbin = coqbin.rstrip(os.sep) + os.sep
        environment["COQBIN"] = normalized_coqbin
        environment["PATH"] = normalized_coqbin + os.pathsep + environment.get("PATH", "")
    start_record = {
        "event": "case-start",
        "schema": 1,
        "run_id": run_id,
        "case_id": case["id"],
        "source_revision": case["source_revision"],
        "expected_category": case["expected_category"],
        "expected_fallback": case["expected_fallback"],
        "archived_bug": case["archived_bug"],
        "expected_outcome": case["expected_outcome"],
        "mode": mode,
        "command": command,
        "environment": {
            key: (
                hashlib.sha256(environment.get(key, "").encode("utf-8")).hexdigest()
                if key in environment
                else None
            )
            for key in case["environment_keys"]
        },
        "started_at": time.time(),
    }
    _append_jsonl(records_path, start_record)
    started = time.monotonic()
    timed_out = False
    with open(log_path, "wb") as log:
        process = subprocess.Popen(
            command,
            cwd=repository,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        try:
            process.communicate(input=("y\n" * 20000).encode("ascii"), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                try:
                    os.killpg(process.pid, 15)
                except OSError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, 9)
                    except OSError:
                        pass
                else:
                    process.kill()
                process.wait()
    runtime = time.monotonic() - started
    outputs = _retain_outputs(repository, output_directory, case)
    end_record = {
        "event": "case-end",
        "schema": 1,
        "run_id": run_id,
        "case_id": case["id"],
        "mode": mode,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "expected_outcome": case["expected_outcome"],
        "case_satisfied": (
            (process.returncode == 0 and not timed_out)
            or case["expected_outcome"] == "disabled"
        ),
        "runtime": runtime,
        "log_path": os.path.relpath(log_path, output_directory),
        "log_sha256": _sha256_file(log_path),
        "outputs": outputs,
        "finished_at": time.time(),
    }
    _append_jsonl(records_path, end_record)
    return end_record


def _direction(record):
    document = record.get("document_verdict")
    compiler = record.get("compiler_verdict")
    if document is None or compiler is None or document == compiler:
        return None
    return "false-negative" if document is False else "false-positive"


def summarize_records(records, manifest=None):
    comparisons = [item for item in records if item.get("event") == COMPARISON_EVENT]
    starts = [item for item in records if item.get("event") == "case-start"]
    ends = [item for item in records if item.get("event") == "case-end"]
    directions = {"false-positive": 0, "false-negative": 0}
    unsupported = {}
    unclassified = []
    for record in comparisons:
        direction = _direction(record)
        if direction:
            directions[direction] += 1
            reason = record.get("reason") or record.get("unsupported_reason")
            if not reason:
                unclassified.append(record)
        reason = record.get("reason") or record.get("unsupported_reason")
        if reason:
            unsupported[reason] = unsupported.get(reason, 0) + 1
    latest_end_by_case = {}
    for item in ends:
        latest_end_by_case[item.get("case_id")] = item
    final_ends = list(latest_end_by_case.values())
    successful = sum(
        1
        for item in final_ends
        if item.get("case_satisfied", (
            item.get("returncode") == 0 and not item.get("timed_out")
        ))
    )
    archived = 0
    minimum = 10000
    declared_case_count = None
    if manifest is not None:
        archived = sum(1 for case in manifest["cases"] if case.get("archived_bug"))
        minimum = manifest["declared_configuration"].get("minimum_comparisons", minimum)
        declared_case_count = len(manifest["cases"])
    compiler_runtimes = [
        item["compiler_runtime"]
        for item in comparisons
        if isinstance(item.get("compiler_runtime"), (int, float))
    ]
    document_runtimes = [
        item["document_runtime"]
        for item in comparisons
        if isinstance(item.get("document_runtime"), (int, float))
    ]
    split_runtimes = [
        item["document_split_runtime"]
        for item in comparisons
        if isinstance(item.get("document_split_runtime"), (int, float))
    ]
    execution_runtimes = [
        item["document_execution_runtime"]
        for item in comparisons
        if isinstance(item.get("document_execution_runtime"), (int, float))
    ]
    reused_items = [
        item["reused_items"]
        for item in comparisons
        if isinstance(item.get("reused_items"), int)
    ]
    replayed = [
        item["commands_replayed"]
        for item in comparisons
        if isinstance(item.get("commands_replayed"), int)
    ]
    diagnostic_values = [
        item["diagnostic_agreement"]
        for item in comparisons
        if isinstance(item.get("diagnostic_agreement"), bool)
    ]
    case_runtimes = [
        item["runtime"]
        for item in final_ends
        if isinstance(item.get("runtime"), (int, float))
    ]
    edit_strategy_counts = {}
    edit_strategy_split_runtime = {}
    edit_kind_counts = {}
    for item in comparisons:
        edit_kind = item.get("edit_kind")
        if edit_kind is not None:
            edit_kind_counts[edit_kind] = edit_kind_counts.get(edit_kind, 0) + 1
        strategy = item.get("edit_strategy")
        if strategy is None:
            continue
        edit_strategy_counts[strategy] = edit_strategy_counts.get(strategy, 0) + 1
        runtime = item.get("document_split_runtime")
        if isinstance(runtime, (int, float)):
            edit_strategy_split_runtime[strategy] = (
                edit_strategy_split_runtime.get(strategy, 0.0) + runtime
            )
    return {
        "schema": 1,
        "case_starts": len(starts),
        "case_ends": len(final_ends),
        "case_attempts": len(ends),
        "successful_cases": successful,
        "failed_cases": len(final_ends) - successful,
        "declared_case_count": declared_case_count,
        "archived_bug_count": archived,
        "comparison_count": len(comparisons),
        "agreement_count": sum(item.get("agreement") is True for item in comparisons),
        "unavailable_count": sum(item.get("agreement") is None for item in comparisons),
        "false_positive_count": directions["false-positive"],
        "false_negative_count": directions["false-negative"],
        "unclassified_disagreement_count": len(unclassified),
        "unsupported_reasons": unsupported,
        "compiler_runtime_total": sum(compiler_runtimes),
        "document_runtime_total": sum(document_runtimes),
        "document_split_runtime_total": sum(split_runtimes),
        "document_execution_runtime_total": sum(execution_runtimes),
        "edit_strategy_counts": edit_strategy_counts,
        "edit_strategy_split_runtime_total": edit_strategy_split_runtime,
        "edit_kind_counts": edit_kind_counts,
        "case_runtime_total": sum(case_runtimes),
        "case_runtime_median": _median(case_runtimes),
        "reused_item_observations": len(reused_items),
        "reused_items_positive_count": sum(value > 0 for value in reused_items),
        "reused_items_median": _median(reused_items),
        "reused_items_maximum": max(reused_items) if reused_items else None,
        "commands_replayed_median": _median(replayed),
        "commands_replayed_maximum": max(replayed) if replayed else None,
        "diagnostic_agreement_count": sum(diagnostic_values),
        "diagnostic_comparison_count": len(diagnostic_values),
        "maximum_compiler_peak_rss_kb": max(
            [item.get("compiler_peak_rss_kb") or 0 for item in comparisons]
            or [0]
        ),
        "maximum_restart_count": max(
            [item.get("restart_count") or 0 for item in comparisons] or [0]
        ),
        "maximum_live_session_count": max(
            [item.get("live_session_count") or 0 for item in comparisons] or [0]
        ),
        "maximum_live_cursor_count": max(
            [item.get("live_cursor_count") or 0 for item in comparisons] or [0]
        ),
        "gates": {
            "all_declared_cases_completed": (
                declared_case_count is not None
                and len(final_ends) == declared_case_count
            ),
            "all_executed_cases_passed": (
                len(final_ends) > 0 and successful == len(final_ends)
            ),
            "archived_bug_count": archived >= 20,
            "minimum_comparisons": len(comparisons) >= minimum,
            "disagreements_classified": not unclassified,
        },
    }


def _repository_has_tracked_changes(repository):
    environment = dict(os.environ)
    # Keep even Git's optional index refresh out of the caller checkout.
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return bool(
        _checked_command(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=repository,
            environment=environment,
        )
    )


def _create_isolated_repository(repository, destination):
    """Clone committed HEAD without sharing writable Git object storage."""
    top_level = _checked_command(
        ["git", "rev-parse", "--show-toplevel"], cwd=repository
    )
    if not top_level or os.path.realpath(top_level) != os.path.realpath(repository):
        raise RuntimeError("--repository must name the Git working-tree root")
    commit = _checked_command(["git", "rev-parse", "HEAD"], cwd=repository)
    if not commit:
        raise RuntimeError("Unable to identify repository HEAD")
    if _repository_has_tracked_changes(repository):
        raise RuntimeError(
            "run-corpus requires a clean index and tracked working tree; "
            "untracked and ignored caller files are allowed and remain untouched"
        )
    _checked_command(
        ["git", "clone", "--no-hardlinks", "--no-checkout", "--", repository, destination]
    )
    _checked_command(["git", "checkout", "--detach", commit], cwd=destination)
    _checked_command(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=destination,
    )
    return destination


def _initialize_output_directory(output_directory, resume):
    marker_path = os.path.join(output_directory, OUTPUT_MARKER)
    if not resume:
        if os.path.lexists(output_directory):
            raise RuntimeError(
                "run-corpus --output must not already exist without --resume"
            )
        os.makedirs(output_directory)
        descriptor = _open_regular_file(
            marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        try:
            _write_all(
                descriptor,
                (_json_dump({"schema": 1, "kind": "rdm-corpus-output"}) + "\n").encode(
                    "utf-8"
                ),
            )
        finally:
            os.close(descriptor)
        return

    if os.path.islink(output_directory) or not os.path.isdir(output_directory):
        raise RuntimeError("--resume requires an owned corpus output directory")
    if os.path.islink(marker_path) or not os.path.isfile(marker_path):
        raise RuntimeError("--resume output lacks the corpus ownership marker")
    with open(marker_path, "r", encoding="utf-8") as source:
        try:
            marker = json.load(source)
        except ValueError as exc:
            raise RuntimeError("Invalid corpus ownership marker: %s" % exc)
    if marker != {"schema": 1, "kind": "rdm-corpus-output"}:
        raise RuntimeError("Unrecognized corpus ownership marker")
    def walk_error(error):
        raise RuntimeError(
            "Unable to inspect resume output %s: %s"
            % (getattr(error, "filename", output_directory), error)
        )

    for root, directories, files in os.walk(
        output_directory, followlinks=False, onerror=walk_error
    ):
        for name in directories:
            path = os.path.join(root, name)
            if os.path.islink(path) or not os.path.isdir(path):
                raise RuntimeError(
                    "Refusing to resume with unsafe output directory %s" % path
                )
        for name in files:
            path = os.path.join(root, name)
            if os.path.islink(path) or not os.path.isfile(path):
                raise RuntimeError(
                    "Refusing to resume with symlinked or non-regular output %s"
                    % path
                )
            if os.stat(path).st_nlink != 1:
                raise RuntimeError(
                    "Refusing to resume with hard-linked output %s" % path
                )


def run_corpus(arguments):
    source_repository = os.path.abspath(arguments.repository)
    manifest_path = os.path.abspath(arguments.manifest)
    manifest = load_manifest(manifest_path, repository=source_repository)
    output_directory = os.path.abspath(arguments.output)
    try:
        output_within_repository = os.path.commonpath(
            (os.path.realpath(source_repository), os.path.realpath(output_directory))
        ) == os.path.realpath(source_repository)
    except ValueError:
        output_within_repository = False
    if output_within_repository:
        raise RuntimeError(
            "run-corpus --output must be outside the source repository"
        )
    _initialize_output_directory(output_directory, arguments.resume)
    records_path = os.path.join(output_directory, "records.jsonl")
    run_id = arguments.run_id or str(uuid.uuid4())
    configuration = manifest["declared_configuration"]
    work_directory = tempfile.mkdtemp(prefix="rdm-corpus-")
    repository = os.path.join(work_directory, "repository")
    try:
        _create_isolated_repository(source_repository, repository)
        # The user may choose another manifest, but its scripts and schema must
        # describe this exact immutable checkout before any case is executed.
        load_manifest(manifest_path, repository=repository)
        metadata = machine_metadata(repository, arguments.manager, arguments.coqc)
        _append_jsonl(
            records_path,
            {
                "event": "run-start",
                "schema": 1,
                "run_id": run_id,
                "mode": arguments.mode,
                "manifest_sha256": _sha256_file(manifest_path),
                "declared_configuration": configuration,
                "machine": metadata,
                "started_at": time.time(),
            },
        )
        requested = set(arguments.case or ())
        selected = [
            case
            for case in manifest["cases"]
            if _case_selected(case, requested, arguments.case_pattern)
        ]
        if arguments.max_cases is not None:
            selected = selected[: arguments.max_cases]
        completed = set()
        if arguments.resume:
            completed = set(
                item.get("case_id")
                for item in _read_jsonl(records_path)
                if item.get("event") == "case-end" and item.get("returncode") == 0
            )
        for index, case in enumerate(selected, 1):
            if case["id"] in completed:
                continue
            print("[%d/%d] %s" % (index, len(selected), case["id"]), flush=True)
            result = run_case(
                repository,
                output_directory,
                records_path,
                run_id,
                case,
                arguments.mode,
                arguments.manager,
                arguments.passing_manager,
                os.path.abspath(arguments.python),
                arguments.coqbin,
                arguments.case_timeout,
            )
            if result["returncode"] != 0 and arguments.fail_fast:
                break
        records = _read_jsonl(records_path)
        summary = summarize_records(records, manifest)
        summary.update({"run_id": run_id, "machine": metadata})
        summary_path = os.path.join(output_directory, "summary.json")
        summary_payload = (
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        summary_descriptor = _open_regular_file(
            summary_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        )
        try:
            _write_all(summary_descriptor, summary_payload)
        finally:
            os.close(summary_descriptor)
        _append_jsonl(
            records_path,
            {
                "event": "run-end",
                "schema": 1,
                "run_id": run_id,
                "summary_sha256": _sha256_file(summary_path),
                "finished_at": time.time(),
            },
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if all(summary["gates"].values()) else 2
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            shutil.rmtree(work_directory)
        except OSError as exc:
            message = "Unable to remove disposable corpus clone %s: %s" % (
                work_directory,
                exc,
            )
            if not active_exception:
                raise RuntimeError(message)
            try:
                print("Warning: " + message, file=sys.stderr)
            except BaseException:
                pass


def _benchmark_source(definitions, spin=0):
    if spin > 0:
        header = (
            "Fixpoint bench_spin (n : nat) : nat := "
            "match n with O => O | S n0 => bench_spin n0 end.\n"
        )
        prefix = header + "".join(
            "Definition bench_%04d := Eval vm_compute in bench_spin %d.\n"
            % (index, spin)
            for index in range(definitions)
        )
    else:
        prefix = "".join(
            "Definition bench_%04d : nat := %d.\n" % (index, index % 17)
            for index in range(definitions)
        )
    return prefix, prefix + "Check missing_baseline.\n"


def _median(values):
    return statistics.median(values) if values else None


def benchmark_prefix(arguments):
    from .candidate_evaluator import (
        CandidateChange,
        CoqcEvaluator,
        EnvironmentSnapshot,
        EvaluationContextSpec,
        LegacyTargetPolicy,
        ResourceRequest,
    )
    from .rdm_backend import RdmSessionPool

    repository = os.path.abspath(arguments.repository)
    output = os.path.abspath(arguments.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    environment = dict(os.environ)
    if arguments.coqbin:
        environment["PATH"] = arguments.coqbin.rstrip(os.sep) + os.pathsep + environment.get("PATH", "")
    work = tempfile.mkdtemp(prefix="rdm-benchmark-")
    log = lambda *unused_args, **unused_kwargs: None
    compiler = CoqcEvaluator(log=log, verbose_base=3)
    prefix, baseline = _benchmark_source(arguments.definitions, arguments.spin)
    policy = LegacyTargetPolicy(False, "missing_baseline")
    spec = EvaluationContextSpec(
        (arguments.coqc,),
        (),
        cwd=work,
        environment=EnvironmentSnapshot(environment),
        logical_file=os.path.join(work, "benchmark.v"),
        resource_request=ResourceRequest(None, memory_usage_key=None),
    )
    context = compiler.materialize_context(spec)
    pool = RdmSessionPool(
        (arguments.manager,),
        baseline,
        policy,
        restart_every=0,
        request_timeout=arguments.request_timeout,
    )
    compiler_samples = []
    document_samples = []
    split_samples = []
    execution_samples = []
    reused_items = []
    try:
        total_repetitions = arguments.warmups + arguments.repetitions
        for repetition in range(total_repetitions):
            compiler_started = time.monotonic()
            for candidate_index in range(arguments.candidates):
                source = prefix + "Check I. (* candidate %d *)\n" % candidate_index
                candidate = CandidateChange.from_sources(baseline, source)
                trial = compiler.begin(context, candidate, policy)
                compiler.finish(trial, False)
            compiler_runtime = time.monotonic() - compiler_started

            document_started = time.monotonic()
            current_split = []
            current_execution = []
            current_reused = []
            for candidate_index in range(arguments.candidates):
                source = prefix + "Check I. (* candidate %d *)\n" % candidate_index
                candidate = CandidateChange.from_sources(baseline, source)
                trial = pool.begin(context, candidate)
                current_split.append(trial.observation.split_runtime)
                current_execution.append(trial.observation.execution_runtime)
                current_reused.append(trial.observation.reused_items)
                pool.finish(trial, False)
            document_runtime = time.monotonic() - document_started
            if repetition >= arguments.warmups:
                compiler_samples.append(compiler_runtime)
                document_samples.append(document_runtime)
                split_samples.append(sum(current_split))
                execution_samples.append(sum(current_execution))
                reused_items.extend(current_reused)
        compiler_median = _median(compiler_samples)
        document_median = _median(document_samples)
        speedup = (
            compiler_median / document_median
            if document_median and compiler_median is not None
            else None
        )
        result = {
            "schema": 1,
            "event": "prefix-benchmark",
            "repository_commit": _git(repository, ["rev-parse", "HEAD"]),
            "machine": machine_metadata(repository, arguments.manager, arguments.coqc, environment),
            "definitions": arguments.definitions,
            "spin": arguments.spin,
            "candidates_per_repetition": arguments.candidates,
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "compiler_samples": compiler_samples,
            "document_samples": document_samples,
            "split_samples": split_samples,
            "execution_samples": execution_samples,
            "compiler_median": compiler_median,
            "document_median": document_median,
            "candidate_oracle_speedup": speedup,
            "minimum_reused_items": min(reused_items) if reused_items else None,
            "maximum_reused_items": max(reused_items) if reused_items else None,
            "gate_candidate_oracle_2x": speedup is not None and speedup >= 2.0,
        }
        with open(output, "w", encoding="utf-8") as destination:
            json.dump(result, destination, indent=2, sort_keys=True)
            destination.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate_candidate_oracle_2x"] else 2
    finally:
        pool.close()
        compiler.close()
        shutil.rmtree(work, ignore_errors=True)


def _rss_kb(pid):
    try:
        with open("/proc/%d/status" % pid, "r", encoding="ascii") as source:
            for line in source:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (IOError, OSError, ValueError):
        return None
    return None


def long_session(arguments):
    from .candidate_evaluator import (
        CandidateChange,
        CoqcEvaluator,
        EnvironmentSnapshot,
        EvaluationContextSpec,
        LegacyTargetPolicy,
        ResourceRequest,
    )
    from .rdm_backend import RdmSessionPool

    output = os.path.abspath(arguments.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    work = tempfile.mkdtemp(prefix="rdm-long-session-")
    environment = dict(os.environ)
    if arguments.coqbin:
        environment["PATH"] = arguments.coqbin.rstrip(os.sep) + os.pathsep + environment.get("PATH", "")
    compiler = CoqcEvaluator(lambda *args, **kwargs: None)
    source = "Definition x := True.\nCheck missing_long_session.\n"
    policy = LegacyTargetPolicy(False, "missing_long_session")
    context = compiler.materialize_context(
        EvaluationContextSpec(
            (arguments.coqc,),
            (),
            cwd=work,
            environment=EnvironmentSnapshot(environment),
            logical_file=os.path.join(work, "long_session.v"),
            resource_request=ResourceRequest(None, memory_usage_key=None),
        )
    )
    pool = RdmSessionPool(
        (arguments.manager,),
        source,
        policy,
        restart_every=0,
        request_timeout=arguments.request_timeout,
    )
    peak_rss = None
    initial_rss = None
    started = time.monotonic()
    try:
        candidate = CandidateChange.from_sources(source, source)
        for index in range(arguments.trials):
            trial = pool.begin(context, candidate)
            pool.finish(trial, False)
            if index == 0 or (index + 1) % arguments.sample_every == 0:
                session = pool._sessions.get(context)
                process = getattr(getattr(session.client, "transport", None), "_process", None)
                rss = _rss_kb(process.pid) if process is not None else None
                if initial_rss is None:
                    initial_rss = rss
                if rss is not None:
                    peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        session = pool._sessions.get(context)
        process = getattr(getattr(session.client, "transport", None), "_process", None)
        final_rss = _rss_kb(process.pid) if process is not None else None
        result = {
            "schema": 1,
            "event": "long-session",
            "trials": arguments.trials,
            "runtime": time.monotonic() - started,
            "restart_count": pool.restart_count,
            "live_session_count": pool.session_count,
            "active_trial_count": session.active_trial_count,
            "initial_rss_kb": initial_rss,
            "peak_rss_kb": peak_rss,
            "final_rss_kb": final_rss,
            "rss_growth_kb": (
                None
                if initial_rss is None or final_rss is None
                else final_rss - initial_rss
            ),
            "gate_10000_trials": arguments.trials >= 10000,
            "gate_no_live_trial_growth": session.active_trial_count == 0,
            "gate_one_session": pool.session_count == 1,
        }
        with open(output, "w", encoding="utf-8") as destination:
            json.dump(result, destination, indent=2, sort_keys=True)
            destination.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if all(
            result[key]
            for key in (
                "gate_10000_trials",
                "gate_no_live_trial_growth",
                "gate_one_session",
            )
        ) else 2
    finally:
        pool.close()
        compiler.close()
        shutil.rmtree(work, ignore_errors=True)


def summarize_command(arguments):
    manifest = load_manifest(arguments.manifest, repository=arguments.repository)
    summary = summarize_records(_read_jsonl(arguments.records), manifest)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output:
            json.dump(summary, output, indent=2, sort_keys=True)
            output.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(summary["gates"].values()) else 2


def validate_command(arguments):
    manifest = load_manifest(arguments.manifest, repository=arguments.repository)
    value = {
        "schema": manifest["schema"],
        "case_count": len(manifest["cases"]),
        "archived_bug_count": sum(case["archived_bug"] for case in manifest["cases"]),
        "all_example_scripts_named": True,
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.getcwd())
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", default=DEFAULT_MANIFEST)
    validate.set_defaults(action=validate_command)

    corpus = subparsers.add_parser(
        "run-corpus",
        description=(
            "Run committed HEAD in a disposable clone. Caller untracked and "
            "ignored files are not copied or cleaned; logs and generated "
            "outputs are retained only under --output, which must be outside "
            "the source repository and newly created unless --resume is used."
        ),
    )
    corpus.add_argument("--manifest", default=DEFAULT_MANIFEST)
    corpus.add_argument("--output", required=True)
    corpus.add_argument("--mode", choices=("rdm-shadow", "rdm-hybrid"), default="rdm-shadow")
    corpus.add_argument("--manager", default="rocq-doc-manager")
    corpus.add_argument("--passing-manager")
    corpus.add_argument("--coqc", default="coqc")
    corpus.add_argument("--python", default=sys.executable)
    corpus.add_argument("--coqbin", default="")
    corpus.add_argument("--case", action="append")
    corpus.add_argument("--case-pattern")
    corpus.add_argument("--max-cases", type=int)
    corpus.add_argument("--case-timeout", type=float, default=3600)
    corpus.add_argument("--run-id")
    corpus.add_argument("--resume", action="store_true")
    corpus.add_argument("--fail-fast", action="store_true")
    corpus.set_defaults(action=run_corpus)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--manifest", default=DEFAULT_MANIFEST)
    summary.add_argument("--records", required=True)
    summary.add_argument("--output")
    summary.set_defaults(action=summarize_command)

    benchmark = subparsers.add_parser("benchmark-prefix")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--manager", default="rocq-doc-manager")
    benchmark.add_argument("--coqc", default="coqc")
    benchmark.add_argument("--coqbin", default="")
    benchmark.add_argument("--definitions", type=int, default=400)
    benchmark.add_argument("--candidates", type=int, default=10)
    benchmark.add_argument("--spin", type=int, default=0)
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument("--repetitions", type=int, default=5)
    benchmark.add_argument("--request-timeout", type=float, default=30)
    benchmark.set_defaults(action=benchmark_prefix)

    long_run = subparsers.add_parser("long-session")
    long_run.add_argument("--output", required=True)
    long_run.add_argument("--manager", default="rocq-doc-manager")
    long_run.add_argument("--coqc", default="coqc")
    long_run.add_argument("--coqbin", default="")
    long_run.add_argument("--trials", type=int, default=10000)
    long_run.add_argument("--sample-every", type=int, default=100)
    long_run.add_argument("--request-timeout", type=float, default=30)
    long_run.set_defaults(action=long_session)
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not hasattr(arguments, "action"):
        parser.error("a subcommand is required")
    arguments.repository = os.path.abspath(arguments.repository)
    if hasattr(arguments, "manifest") and not os.path.isabs(arguments.manifest):
        arguments.manifest = os.path.join(arguments.repository, arguments.manifest)
    return arguments.action(arguments)


if __name__ == "__main__":
    sys.exit(main())
