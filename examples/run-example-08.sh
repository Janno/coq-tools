#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
PS4='$ '
set -x
python2 run-example-08.py || exit $?
