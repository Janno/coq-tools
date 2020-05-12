#!/usr/bin/env python2

import split_file
import subprocess as proc
import select
import signal
from collections import deque
from sys import stdout, stderr, argv

if __name__ == '__main__':
    statements = split_file.split_coq_file_contents(file(argv[1]).read())

    # args: file.v -- coqtop args
    assert(argv[2] == '--')
    coqtop_args = argv[3:]

    command = ['perf', 'record', '--call-graph', 'dwarf,65528', '-F', '999', '--', 'coqtop', '-q', '-time'] + list(coqtop_args)
    print ' '.join(command)

    perf = proc.Popen(['perf', 'record', '--switch-output=signal', '-o', 'perf.data', '--call-graph', 'dwarf,65528', '-F', '999', '--', 'coqtop', '-quiet', '-time'] + list(coqtop_args), stdout=proc.PIPE, stderr=proc.PIPE, stdin=proc.PIPE)

    reads = [ perf.stdout.fileno(), perf.stderr.fileno()]
    writes = [perf.stdin.fileno()]
    # writes = []
    to_pipe = { pipe.fileno():pipe for pipe in [perf.stdout, perf.stderr, perf.stdin] }

    prompt = "Coq < "
    lprompt = list(prompt)

    # only call this function if you know stderr will eventually be readable
    def read_stdout_until_exhausted(call_on_stdout):
        while True:
            sreads, _, _ = select.select(reads, [], [])
            if perf.stdout.fileno() in sreads:
                call_on_stdout()
            else:
                break


    def read_until_prompt(call_on_stdout):
        is_prompt = deque(maxlen=len(lprompt))

        def min_distance_to_prompt():
            buf = list(is_prompt)
            for i in xrange(len(lprompt)):
                if buf[i:] == lprompt[:6-i]:
                    return i
            else:
                return 6

        while True:
            read_stdout_until_exhausted(call_on_stdout)
            sreads, _, _ = select.select([perf.stderr.fileno()], [], [])
            if sreads:
                cs = perf.stderr.read(min_distance_to_prompt())
                # stderr.write(c)
                is_prompt.extend(list(cs))
            if list(is_prompt) == list(prompt):
                break
            print list(is_prompt)


    for s in statements:
        read_until_prompt(lambda: perf.stdout.read(1))

        # cycle perf.data files

        perf.send_signal(signal.SIGUSR2)

        print " <", s
        perf.stdin.write(s)
        perf.stdin.write("\n")
        perf.stdin.flush()


    read_until_prompt(lambda: perf.stdout.read(1))
    perf.stdin.close()
