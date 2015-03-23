#!/usr/bin/python
from __future__ import with_statement
import argparse, os, sys, shutil, re
from split_file import split_coq_file_contents_with_comments
from strip_comments import strip_comments

SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Move various statements out of proof blocks')
parser.add_argument('input_files', metavar='INFILE', nargs='+', type=str,
                    help='.v files to update')
parser.add_argument('--in-place', '-i', metavar='SUFFIX', dest='suffix', nargs='?', type=str, default='',
                    help='update files in place (makes backup if SUFFIX supplied)')
parser.add_argument('--verbose', '-v', dest='verbose',
                    action='count',
                    help='display some extra information')
parser.add_argument('--quiet', '-q', dest='quiet',
                    action='count',
                    help='the inverse of --verbose')
parser.add_argument('--log-file', '-l', dest='log_files', nargs='*', type=argparse.FileType('w'),
                    default=[sys.stderr],
                    help='The files to log output to.  Use - for stdout.')

DEFAULT_VERBOSITY=1

def make_logger(log_files):
    def log(text):
        for i in log_files:
            i.write(str(text) + '\n')
            i.flush()
            if i.fileno() > 2: # stderr
                os.fsync(i.fileno())
    return log

DEFAULT_LOG = make_logger([sys.stderr])

def backup(file_name, ext='.bak'):
    if not ext:
        raise ValueError
    if os.path.exists(file_name):
        backup(file_name + ext)
        os.rename(file_name, file_name + ext)

def write_to_file(file_name, contents, do_backup=False, ext='.bak'):
    if do_backup:
        backup(file_name, ext=ext)
    try:
        with open(file_name, 'w', encoding='UTF-8') as f:
            f.write(contents)
    except TypeError:
        with open(file_name, 'w') as f:
            f.write(contents)

ALL_DEFINITIONS_REG = re.compile(r'^\s*(?:(?:Global|Local|Polymorphic|Monomorphic|Time|Timeout)\s+)?(?:' +
                                r'Theorem|Lemma|Fact|Remark|Corollary|Proposition|Property' +
                                r'|Definition|Example|SubClass' +
                                r'|Let|Fixpoint|CoFixpoint' +
                                r'|Structure|Coercion|Instance' +
                                r'|Add Parametric Morphism' +
                                r')\s', re.MULTILINE)

ONELINE_DEFINITIONS_REG = re.compile(r'^\s*(?:(?:Global|Local|Polymorphic|Monomorphic|Time|Timeout)\s+)?(?:' +
                                     r'Coercion|Existing\s+Instance' +
                                     r')\s', re.MULTILINE)

ALL_ENDINGS = re.compile(r'^(?:[}{]|\s)*(?:(?:Time|Timeout)\s+)?(?:Qed|Defined|Save|Admitted|Abort)\s*\.', re.MULTILINE)

COPY_UP_REG = re.compile(r'^\s*(?:(?:Global|Local|Polymorphic|Monomorphic|Time|Timeout)\s+)?' +
                         r'(?:Opaque\s|Transparent\s|Arguments\s|Implicit\s+Arguments\s)', re.MULTILINE)
MOVE_UP_REG = re.compile(r'^\s*(?:(?:Global|Local|Polymorphic|Monomorphic|Time|Timeout)\s+)?' +
                         r'(?:Require|Import|Notation|Ltac|Tactic\s+Notation|Infix|Delimit\s+Scope|Reserved\s+Notation|Reserved\+Infix|Existing\s+Instance|Coercion|Hint|Set\s+Printing|Unset\+Printing)\s', re.MULTILINE)
SPECIAL_TACTICS = r'W_eq|PropXRel|ILAlgoTypes|NoDup|W_neq|PropXTac|SEP_FACTS|SF\.'
SAFE_REG = re.compile(r'^\s*(?:(?:Global|Local|Polymorphic|Monomorphic|Time|Timeout)\s+)?(?:' +
                      r'[a-z]|[0-9]+\s*:|\(\s*[a-z]|' + SPECIAL_TACTICS +
                      r'|{|}|-|\+|\*' +
                      r'|(?:Unfocus|Proof|Focus|Grab\s+Existentials?|Open\s+Scope|Close\s+Scope)(?:\.|\s)|\(\*' +
                      r')', re.MULTILINE)

reg = re.compile(r'Require\s.*?\.\s+', re.MULTILINE | re.DOTALL)

def get_leading_space(string):
    return re.findall(r'^(?:\s*?\n)?([ \t]*)(?!\s)', string)[0]

def remove_leading_space(string, space_count):
    return re.sub(r'(^|\n)' + (' ' * space_count), r'\1', string, re.MULTILINE)

def set_leading_space(string, space_count):
    return remove_leading_space(string, max(0, len(get_leading_space(string)) - space_count))

def strip_parens(string):
    last = string
    cur = re.sub('"[^"]+|{[^{}]+}|\([^\(\)]+\)', '', last)
    while cur != last:
        last, cur = cur, re.sub('"[^"]+|{[^{}]+}|\([^\(\)]+\)', '', cur)
    cur = re.sub(r'\slet\s[^\(\){}"]+?:=[^\(\){}"]+?\sin\s', '', cur, re.MULTILINE)
    return cur


def space_canonicalize(stmt):
    return re.sub(r'\s+', ' ', stmt.strip(' \t\n\r.'))

def canonicalize(stmt):
    return space_canonicalize(stmt).replace('Transparent ', 'Opaque ')

def cancels(statement1, statement2):
    """Returns True if statement2 cancels the effect of statement1; False otherwise"""
    return canonicalize(statement1) == canonicalize(statement2)

def commutes(statement1, statement2):
    if any(i in statement1 or i in statement2 for i in ('(*', '*)', '"')):
        return False
    statement1, statement2 = space_canonicalize(statement1), space_canonicalize(statement2)
    m1 = re.match(r'^(?:Local |Global )?(?:Transparent|Opaque) (.*)$', statement1)
    m2 = re.match(r'^(?:Local |Global )?(?:Transparent|Opaque) (.*)$', statement2)
    if not m1 or not m2:
        return False
    constants1, constants2 = set(m1.groups()[0].split(' ')), set(m2.groups()[0].split(' '))
    return constants1.isdisjoint(constants2)

def preminimize_lifted_statements(statements):
    """Remove useless deferred statements, such as transparent followed by opaque"""

    prev = set()
    for statement in statements:
        prev = set(last for last in prev if not cancels(last, statement))
        if not all(commutes(last, statement) for last in prev):
            for last in sorted(prev):
                yield last
            prev = set()
        prev.add(statement)
    for last in sorted(prev):
        yield last


def minimize_lifted_statements(statements):
    statements = list(preminimize_lifted_statements(statements))
    sans_copy = [i for i in statements if COPY_UP_REG.match(i.strip()) is None]
    sans_copy_hint = [i for i in sans_copy if re.match(r'^(?:Local\s+|Global\s+)?Hint\s', i.strip()) is None]
    if len(sans_copy_hint) == 1:
        return sans_copy
    else:
        return statements

def move_from_proof(filename, **kwargs):
    if kwargs['verbose']: kwargs['log']('Processing %s...' % filename)
    try:
        with open(filename, 'r') as f:
            contents = f.read()
    except IOError, e:
        if kwargs['verbose']: kwargs['log']('Failed to process %s' % filename)
        if kwargs['verbose'] >= 2: kwargs['log'](repr(e))
        return
    ret = []
    cur_statements = []
    cur_statement = []
    deferred_copied_statements = []
    deferred_statements = []
    orig_space_count = 0
    cur_diff_space_count = 0
    if ''.join(split_coq_file_contents_with_comments(contents)) != contents:
        kwargs['log']('WARNING: Could not split %s' % filename)
        return
    for i in split_coq_file_contents_with_comments(contents):
        is_definition_full = (ALL_DEFINITIONS_REG.match(i) is not None
                              and (':=' in strip_parens(strip_comments(i))
                                   or ONELINE_DEFINITIONS_REG.match(i)))
        is_definition_start = (ALL_DEFINITIONS_REG.match(i) is not None
                               and ':=' not in strip_parens(strip_comments(i))
                               and not ONELINE_DEFINITIONS_REG.match(i))
        #print((is_definition_start, ONELINE_DEFINITIONS_REG.match(i), ALL_DEFINITIONS_REG.match(i), i))
        is_definition_end = ALL_ENDINGS.match(i) is not None
        #if 'not_reachable_iff' in i:
        #    print((ALL_DEFINITIONS_REG.match(i), strip_parens(strip_comments(i)), ONELINE_DEFINITIONS_REG.match(i)))
        if not is_definition_start and not cur_statements and not cur_statement:
            if kwargs['verbose'] >= 3: kwargs['log'](repr(i))
            ret.append(i)
        elif is_definition_start:
            if kwargs['verbose'] >= 2: kwargs['log']('Starting definition (%d): %s' % (len(deferred_statements), repr(i)))
            if not cur_statement and not deferred_statements:
                orig_space_count = len(get_leading_space(i))
            if cur_statement:
                deferred_statements.append((cur_diff_space_count, cur_statement))
            cur_diff_space_count = max(0, len(get_leading_space(i)) - orig_space_count)
            cur_statement = [remove_leading_space(i, cur_diff_space_count)]
        elif (SAFE_REG.match(i) or not i.strip()) and cur_statement:
            if kwargs['verbose'] >= 3: kwargs['log'](repr(i))
            cur_statement.append(remove_leading_space(i, cur_diff_space_count))
        elif is_definition_end and cur_statement:
            if kwargs['verbose'] >= 2: kwargs['log']('Ending definition: ' + repr(i))
            cur_statement.append(remove_leading_space(i, cur_diff_space_count))
            cur_statements.append(''.join(cur_statement))
            #print(''.join(preminimize_lifted_statements(cur_statements)))
            if deferred_statements:
                cur_diff_space_count, cur_statement = deferred_statements.pop()
                deferred_copied_statements = list(preminimize_lifted_statements(deferred_copied_statements))
                cur_statement.extend(deferred_copied_statements) # todo: fix indentation
            else:
                #print('extending')
                #print(''.join(minimize_lifted_statements(cur_statements)))
                ret.extend(list(minimize_lifted_statements(cur_statements)))
                cur_statement = []
                cur_statements = []
                deferred_copied_statements = []
        elif MOVE_UP_REG.match(i) or is_definition_full:
            if kwargs['verbose'] >= 2: kwargs['log']('Lifting: ' + repr(i))
            cur_statements.append(set_leading_space(i, orig_space_count))
        elif COPY_UP_REG.match(i) and cur_statement:
            if kwargs['verbose'] >= 2: kwargs['log']('Lift-copying: ' + repr(i))
            cur_statement.append(remove_leading_space(i, cur_diff_space_count))
            cur_statements.append(set_leading_space(i, orig_space_count))
            deferred_copied_statements.append(remove_leading_space(i, cur_diff_space_count))
        else:
            raw_input('WARNING: Unrecognized: %s' % repr(i))
        #print(cur_diff_space_count)
        #print(remove_leading_space(i, cur_diff_space_count))
    if cur_statements or deferred_statements or cur_statement:
        raw_input('WARNING: extra statements: %s' % repr((cur_statements, cur_statement, deferred_statements)))
        cur_statements.append(''.join(cur_statement))
        for i in deferred_statements:
            cur_statements.append(''.join(i))
        ret.extend(list(minimize_lifted_statements(cur_statements)))
    ret = ''.join(ret)
    if ret == contents:
        return
    if kwargs['inplace']:
        do_backup = kwargs['suffix'] is not None and len(kwargs['suffix']) > 0
        write_to_file(filename, ret, do_backup=do_backup, ext=kwargs['suffix'])
    else:
        print(ret)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose is None: args.verbose = DEFAULT_VERBOSITY
    if args.quiet is None: args.quiet = 0
    verbose = args.verbose - args.quiet
    log = make_logger(args.log_files)
    env = {
        'verbose': verbose,
        'log': log,
        'inplace': args.suffix != '', # it's None if they passed no argument, and '' if they didn't pass -i
        'suffix': args.suffix,
        }

    for f in args.input_files:
        move_from_proof(f, **env)