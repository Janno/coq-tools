Test Suite
==========

To add an example to the test suite, update the `DEFAULT_TESTS` line
in the `Makefile` with your example number, make a new directory
`example_NN/` with the example `.v` files, and make a
`run-example-NN.sh` file based on the commented template
`run-example-00.sh`.  You can look at the other `run-example-NN.sh`
files for inspiration on simpler tests to write.

The rocq-doc-manager corpus runner executes these scripts from committed
`HEAD` in a disposable clone.  It requires the index and tracked working tree
to be clean, but it does not copy, clean, or otherwise modify untracked and
ignored files in the caller's checkout.  Logs and generated outputs are copied
to the directory named by `run-corpus --output` before the clone is removed.
That output path must be outside the source repository and must not already
exist for a new run.  `--resume` accepts only a runner-owned output directory
and rejects symlinks beneath it before writing.
