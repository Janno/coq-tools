"""Regression test for glob generation when the .v file lives in the temp dir.

When the input .v file is directly inside ``tempfile.gettempdir()``, the
temporary glob path (``gettempdir()/<base>.glob``) and the final glob path
(``<v_file_root>.glob``) are the same file.  A bug in make_one_glob_file_helper
then deleted that single glob in its ``finally`` cleanup, so reading it back
raised ``FileNotFoundError``.  This reproduces the failure seen while wiring
coqchk into the bug minimizer (the bug file was created in /tmp); see
https://github.com/rocq-prover/rocq/issues/22125.
"""

import errno
import os
import shutil
import stat
import tempfile

from coq_tools import import_util

FAKE_COQC = """#!/usr/bin/env python3
import sys

args = sys.argv[1:]
if "-h" in args or "--help" in args:
    sys.stdout.write(
        "Usage:\\n"
        "  -q\\n"
        "  -boot\\n"
        "  -nois\\n"
        "  -o f\\toutput file\\n"
        "  -dump-glob f\\tdump glob\\n"
        "  -Q dir coqdir\\tmap\\n"
        "  -R dir coqdir\\tmap\\n"
        "  -coqlib dir\\tset\\n"
    )
    sys.exit(0)
if "-dump-glob" in args:
    g = args[args.index("-dump-glob") + 1]
    with open(g, "w") as f:
        f.write("DUMMY GLOB\\n")
sys.exit(0)
"""


def _make_fake_coqc(dirname):
    path = os.path.join(dirname, "fake_coqc")
    with open(path, "w") as f:
        f.write(FAKE_COQC)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_glob_preserved_for_v_file_directly_in_tmpdir():
    bindir = tempfile.mkdtemp()
    fake_coqc = _make_fake_coqc(bindir)

    # The .v file must be created *directly* in gettempdir() to trigger the
    # tmp_glob_file == glob_file collision.
    fd, v_file = tempfile.mkstemp(
        suffix=".v", prefix="coqchkminimizer_test_", dir=tempfile.gettempdir()
    )
    os.close(fd)
    with open(v_file, "w") as f:
        f.write("(* test file *)\n")
    glob_file = os.path.splitext(v_file)[0] + ".glob"
    vo_file = os.path.splitext(v_file)[0] + ".vo"

    try:
        import_util.make_one_glob_file(
            v_file,
            coqc=(fake_coqc,),
            libnames=(),
            non_recursive_libnames=(),
            ocaml_dirnames=(),
            walk_tree=False,
            use_coq_makefile_for_deps=False,
        )
        # The bug deleted the glob in cleanup; it must survive and be readable.
        assert os.path.exists(glob_file), "glob file was deleted during cleanup"
        with open(glob_file) as f:
            assert "DUMMY GLOB" in f.read()
    finally:
        for p in (v_file, glob_file, vo_file, fake_coqc):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(bindir)


def test_existing_installed_glob_survives_read_only_destination(monkeypatch):
    bindir = tempfile.mkdtemp()
    source_dir = tempfile.mkdtemp()
    fake_coqc = _make_fake_coqc(bindir)
    v_file = os.path.join(source_dir, "installed_source.v")
    glob_file = os.path.splitext(v_file)[0] + ".glob"
    tmp_glob_file = os.path.join(tempfile.gettempdir(), "installed_source.glob")
    with open(v_file, "w") as f:
        f.write("(* installed source *)\n")
    with open(glob_file, "w") as f:
        f.write("INSTALLED GLOB\n")

    real_move = shutil.move

    def read_only_move(src, dst, *args, **kwargs):
        if os.path.realpath(dst) == os.path.realpath(glob_file):
            raise OSError(errno.EROFS, "Read-only file system", dst)
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(import_util.shutil, "move", read_only_move)
    try:
        import_util.make_one_glob_file(
            v_file,
            coqc=(fake_coqc,),
            libnames=(),
            non_recursive_libnames=(),
            ocaml_dirnames=(),
            walk_tree=False,
            use_coq_makefile_for_deps=False,
        )
        with open(glob_file) as f:
            assert f.read() == "INSTALLED GLOB\n"
        assert not os.path.exists(tmp_glob_file)
    finally:
        for path in (v_file, glob_file, tmp_glob_file, fake_coqc):
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(source_dir)
        os.rmdir(bindir)
