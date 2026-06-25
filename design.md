# Design System — Sistem Analisis Sentimen NBC

Dokumen ini mencatat tema visual yang dipakai di seluruh aplikasi (halaman login & admin panel) agar konsisten saat menambah halaman/komponen baru.

## Stack Frontend

- **CSS framework:** Bootstrap 5 (via `vendors/css/vendor.bundle.base.css`)
- **Admin template dasar:** Star Admin2 (`vertical-layout-light`), compiled CSS di `app/static/css/vertical-layout-light/style.css`. SCSS source ada di `app/static/scss/` tapi **tidak ada build process** (tidak ada package.json/gulp/webpack) — jadi edit SCSS tidak berefek, semua override harus lewat CSS biasa.
- **Override tema:** `app/static/css/theme-admin.css` — di-load setelah `style.css` vendor, isinya semua kustomisasi warna/font/radius tema biru.
- **Font:** [Poppins](https://fonts.google.com/specimen/Poppins) (Google Fonts), fallback `Segoe UI, sans-serif`.
- **Icon:** Font Awesome 6 + MDI (Material Design Icons) — dipakai campur, ikuti yang sudah ada di tiap halaman.
- **Lain-lain:** Toastify.js (notifikasi toast), DataTables (tabel data), Chart.js & progressbar.js tersedia tapi belum dipakai aktif di halaman manapun.

## Palet Warna

Token warna utama (didefinisikan di `:root` pada `theme-admin.css` dan halaman login):

| Variabel | Hex | Kegunaan |
|---|---|---|
| `--blue-deep` | `#0d47a1` | Warna paling gelap — header utama, hover/active state, teks brand |
| `--blue-mid` | `#1565c0` | Warna brand utama — tombol primary, link, focus ring, sidebar active |
| `--blue-light` | `#4e8ef7` | Aksen terang — gradient stop, elemen sekunder |
| `--blue-slate` | `#3a5a8c` | Variasi biru keabu-abuan untuk gradient sekunder |
| `--blue-pale` | `#e8f1fd` | Background lembut (hover sidebar, brand badge, divider) |

**Warna semantik (tidak ikut tema biru, sengaja dipertahankan default Bootstrap):**
- Sukses: hijau (`text-success`, `alert-success`, `badge.bg-success` — termasuk badge sentiment "Positif")
- Gagal/error: merah (`text-danger`, `alert-danger`, `badge.bg-danger` — termasuk badge sentiment "Negatif")
- Peringatan: oranye (`text-warning`, `alert-warning`)
- Info/netral: abu-abu/biru info default (`badge.bg-secondary` — badge sentiment "Netral")

> Aturan: jangan timpa `.text-success/.text-danger/.text-warning/.text-info`, `.alert-*`, atau `.badge.bg-success/.bg-danger/.bg-secondary` dengan warna brand — warna-warna ini membawa makna status (lihat prinsip *color-not-decorative-only*).

## Border Radius & Shadow

- `--radius: 14px` — dipakai di card, input, button.
- Card: `border-0 shadow-sm` (ringan) atau `shadow-lg border-0` (card utama per halaman), radius mengikuti `var(--radius)`.
- Login card/blob: shadow lebih halus, tanpa border, dengan `border-radius` mengikuti bentuk blob.

## Tipografi

- Body text: Poppins 400, ukuran dasar `16px`, `line-height` mengikuti default Bootstrap (~1.5).
- Heading (`h1`–`h3`): Poppins 600–700.
- Label/caps text (contoh: label input di login): Poppins 600, `font-size: 12px`, `letter-spacing: 0.04em`, uppercase.

## Komponen Kunci

### Card header dekoratif (section divider)
Semua halaman konten memakai pola yang sama:
```html
<div class="card shadow-lg border-0">
  <div class="card-header bg-gradient-primary text-white py-3">
    <h3 class="mb-0 fw-bold"><i class="fas fa-..."></i> Judul Halaman</h3>
  </div>
  <div class="card-body p-4">...</div>
</div>
```
Class `bg-gradient-primary/secondary/success/info/warning/danger/dark` di-override di `theme-admin.css` menjadi variasi gradient biru monokrom (bukan warna-warni) — gunakan class ini untuk membedakan section secara visual, **bukan** untuk menyampaikan status.

### Sidebar & Navbar
- Sidebar: brand header (logo + nama app) di atas, lalu daftar menu (`<ul class="nav">` → `<li class="nav-item"><a class="nav-link">`). State aktif/hover otomatis biru via `theme-admin.css`.
- Navbar: hanya toggle hamburger + brand text kecil (logo + "Analisis Sentimen NBC") — tetap minimal, tidak ada search bar/profile dropdown.
- Item logout di sidebar tetap pakai class `text-danger` (warna merah dipertahankan, ini aksi destruktif — lihat *destructive-nav-separation*).

### Form
- Input: `form-control`/`form-select`, focus ring biru (`box-shadow: 0 0 0 3px rgba(21,101,192,0.12)`).
- Checkbox/radio terisi (`form-check-input:checked`) ikut warna `--blue-mid`.
- Halaman login pakai pola custom (label uppercase kecil + input dengan icon kiri + toggle password) — lihat `app/templates/auth/login.html` sebagai referensi pattern form minimalis.

### Tombol
- Primary: `.btn-primary` → biru mid, hover ke biru deep.
- Outline: `.btn-outline-primary` → border/teks biru mid, hover terisi biru mid.
- Tombol destruktif (hapus/logout) tetap pakai `btn-danger`/`text-danger` bawaan Bootstrap.

## File Referensi

| File | Isi |
|---|---|
| `app/templates/auth/login.html` | Halaman referensi awal tema biru (split hero/blob panel) |
| `app/static/css/theme-admin.css` | Semua override tema untuk admin panel |
| `app/templates/header.html` | Load font Poppins + `theme-admin.css` |
| `app/templates/navbar.html`, `sidebar.html` | Markup navbar/sidebar dengan brand |
| `app/templates/layout.html` | Base layout yang di-extend semua halaman admin |

## Prinsip Saat Menambah Halaman/Komponen Baru

1. Pakai class Bootstrap yang sudah ada (`card`, `btn-primary`, `bg-gradient-*`, `form-control`, dst) — jangan menulis warna hex baru langsung di template, biarkan `theme-admin.css` yang mengatur tema.
2. Jangan recolor warna semantik (success/danger/warning/info) untuk keperluan dekoratif — pakai `bg-gradient-primary/secondary/dark` dkk untuk itu.
3. Halaman baru yang extend `layout.html` otomatis mengikuti tema; tidak perlu menambahkan `<style>` warna manual kecuali untuk kasus yang benar-benar spesifik ke halaman itu (taruh di akhir block content seperti pola di `dashboard/index.html`).
4. Halaman standalone (tidak extend layout, seperti login) harus mendefinisikan ulang token `--blue-deep/--blue-mid/--blue-light/--radius` di `:root` agar tetap konsisten meski tidak memuat `theme-admin.css`.
