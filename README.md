<p align="center"><img src="docs/brand/readme_banner.png" alt="Linda Probe - local Windows file search, offline OCR and photo search" width="100%"></p>

# Linda Probe — local Windows file search, offline OCR and photo search (English, Polish, Russian)

Free for personal non-commercial use, **paid for Pro and commercial use** — one licence covers the app, the command-line tools and the local OCR/search workflows ([LICENSING.md](LICENSING.md), [PROBE_LICENSING.md](legal/PROBE_LICENSING.md), contact lindapro.support@proton.me). Runs on your machine: no cloud uploads, no telemetry.
Open-weight OCR recognition model lives on Hugging Face: [Lindarixon/linda-ocr](https://huggingface.co/Lindarixon/linda-ocr) (Apache-2.0, ~4.4 MB ONNX CRNN+CTC).

## Latest Windows update: 2.0.0

[Download the Windows release](https://github.com/Lendarixon/Linda-Probe/releases/tag/v2.0.0) · [Release notes](https://github.com/Lendarixon/Linda-Probe/releases/tag/v2.0.0) · [Microsoft Store](https://apps.microsoft.com/detail/9PGM1V68193X) · [Project page](https://lendarixon.github.io/Linda/probe.html?utm_source=github)

App **2.0.0** (Fluent 2.0 interface overhaul, `Ctrl+K` Command Palette, live suggestions and diagnostics):
- **Fluent 2.0 interface overhaul:** updated dark and light themes across English, Russian and Polish.
- **Command Palette (`Ctrl+K`):** jump directly to tabs, folder scopes, recent searches, themes and system diagnostics.
- **Faster workflows:** live debounced filename suggestions as you type, incremental results rendering, export to CSV / Markdown / JSON, and 1-click Copy Citation (`filename + page + snippet`).
- **Diagnostics & licensing:** built-in `/doctor` runtime diagnostics, offline 7-day Pro trial, and machine-bound DPAPI licence cache.

## Also from this project

[Linda](https://github.com/Lendarixon/Linda) is a local detector of AI-generated text for English, Polish and Russian, with built-in **Humanizer** and **Trace** (writing-style comparison). Both apps run entirely on your PC.

## Windows app (no Python needed)

Get it from the [Microsoft Store](https://apps.microsoft.com/detail/9PGM1V68193X) (recommended, signed), or download **LindaProbe-Setup-2.0.0.exe** from the [releases page](https://github.com/Lendarixon/Linda-Probe/releases/download/v2.0.0/LindaProbe-Setup-2.0.0.exe), install it and start it: Free filename and exact-word search works immediately offline without extra downloads, and Pro features download optional models on demand. Free for personal non-commercial use; a Pro / commercial licence key ($10 person, $30 team of 10, $500 per year organization) unlocks semantic search, offline OCR, photo search, automatic tags and commercial use ([how licensing works](LICENSING.md)).

SHA-256 (`LindaProbe-Setup-2.0.0.exe`):
```text
2d4f7fd304e78dc424fb30c4e1a07bf411f99f644f47cd77b24fa61be5cc97bf
```

Verify the SHA-256 hash before running the standalone installer. Because the standalone `.exe` installer is not yet code-signed, Windows SmartScreen may show a warning — after verifying the source and hash, select **More info -> Run anyway**.

## Two editions: Free and Pro

The same app, two capability levels. Neither edition has tariff limits on file count or search speed.
* **Free**: sub-50ms filename lookups, exact-word full-text search (SQLite FTS5), Windows 11-style file explorer, previews, export, citations, manual tags and duplicate finder.
* **Pro**: hybrid AI semantic search (FAISS + FTS5, including cross-language queries), offline OCR for scans and PDFs, photo search by text description or example image, automatic local tags and commercial use.

| | Free | Pro |
|---|---|---|
| Instant filename search (USN / `QuickIndex`) | Yes | Yes |
| Exact-word full-text search (SQLite FTS5: text, code, `.docx`, `.xlsx`, `.pptx`, `.pdf` text layer, `.zip`, `.eml`, `.html`, `.rtf`) | Yes | Yes |
| File explorer, previews, CSV / MD / JSON export, citations, manual tags & duplicate finder | Yes | Yes |
| Hybrid AI semantic search (FAISS + `intfloat/multilingual-e5-small`, cross-language queries) | — | Yes |
| Offline OCR for scans & searchable PDFs (`RapidOCR`, `Tesseract`, `Lindarixon/linda-ocr`) | — | Yes |
| Photo search by description or example image (`clip-vit-b32-en` + `clip-multilingual-text`) | — | Yes |
| Automatic local tags & commercial / workplace use | — | Yes |
| Optional model downloads | none (`0 MB`) | downloaded per feature on demand: Semantic ~470 MB (`multilingual-e5-small`), OCR ~5–20 MB (`linda-ocr` / `RapidOCR`), Photo search ~600 MB (`CLIP ViT-B/32` + multilingual text encoder); ~1.1 GB total if all three packs are enabled |
| Graphics card | not needed | not needed (runs on CPU everywhere); optional DirectML / CUDA GPU accelerates batch OCR and photo indexing |

Recommended computer:

| | Free (filename & exact-word) | Pro (semantic, OCR & photo search) |
|---|---|---|
| System | Windows 10 / 11, 64-bit | Windows 10 / 11, 64-bit |
| Processor | any 64-bit processor (2+ cores) | 4 cores or more recommended for batch OCR and large semantic/photo indexes |
| Memory (RAM) | 4 GB | 8 GB (16 GB recommended when indexing large folders with semantic + CLIP photo models loaded) |
| Graphics card | not needed | not needed (CPU works on any laptop); optional DirectX 12 / CUDA GPU (4 GB+ VRAM) speeds up OCR and photo embedding |
| Free disk space | 500 MB + index (~5–65 MB per 1,000–20,000 chunks) | 2–4 GB (app + optional ~1.1 GB AI model packs + local SQLite/FAISS indexes and OCR cache; SSD recommended) |

Our own measurements on developer hardware, not an independent audit; older laptops and HDDs are slower.

## Speed

Search and indexing measurements on developer hardware (CPU, 6 embed workers for vector scale checks):

| workload | index time | query latency | index size | notes |
|---|---:|---:|---:|---|
| Filename search (`QuickIndex`, warm) | — | `< 50 ms` | in-memory | typically `1–15 ms` |
| 120 files / 960 chunks | `1.7 s` *(incl. model load)* | `14–29 ms` *(exact)* / `15–24 ms` *(semantic)* | — | vs `findstr` `17–23 ms` (exact only, no semantic) |
| 1,000 chunks (flat FAISS, CPU) | `2.0 s` | `16.2 ms` | `4.6 MB` | rank 1 on exact marker queries |
| 20,000 chunks (IVFPQ FAISS, CPU) | `14.0 s` | `82.4 ms` | `64.9 MB` | rank 1 on exact marker queries |

`Lindarixon/linda-ocr` line recognizer (~4.4 MB ONNX CRNN+CTC): character error rate **0.0358** (exact-line accuracy **0.80**) on our own held-out test set of 5,000 generated and real EN/PL/RU lines (231,890 characters). This is an in-house set, not a public benchmark; real scans can be worse.

## Test it on YOUR files (nothing leaves your machine)

Free filename and exact-word search require no extra model downloads:

```bash
pip install -e .
python -m linda_probe find "invoice" --root C:\Docs --format table
python -m linda_probe index C:\Docs
python -m linda_probe search C:\Docs "refund policy" --limit 5 --format md
python -m linda_probe dupes C:\Docs --min-size 1024 --format json
python -m linda_probe settings show
python -m linda_probe models
python -m linda_probe doctor
python -m linda_probe serve --port 8798
```

All list/search commands support `--format table|json|csv|md`. Pro commands (`ocr`, semantic vector indexing, photo search) require an active Pro licence or 7-day trial.

## What it is

Current: app **2.0.0**. Filename lookups use NTFS/directory scanning (`QuickIndex`); exact-word content search uses SQLite FTS5 with built-in extractors for text, code, `.docx`, `.xlsx`, `.pptx`, `.zip`, `.eml`, `.html`, `.rtf`, and `pypdf` for `.pdf` text layers. Pro adds hybrid FAISS + `intfloat/multilingual-e5-small` semantic indexing (with cross-language English/Polish/Russian queries), offline OCR (`RapidOCR` / `Tesseract` / `Lindarixon/linda-ocr`) for image files and scanned PDFs, and CLIP-based photo search (`clip-vit-b32-en` + `clip-multilingual-text`).

Indexing, search, OCR and photo recognition run locally on your machine with zero document uploads. A 7-day Pro trial, promo keys (`PROBE-PROMO-...`), and signed Ed25519 gift keys verify offline; purchased keys activate online via Polar using only the key and a Probe-specific hashed machine ID, stored in a Windows DPAPI-protected local cache. Personal, non-commercial use of Free features is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## Commercial licence, deployment and support

[lindapro.support@proton.me](mailto:lindapro.support@proton.me) — describe the use case, team size or deployment needs.

[LICENSING.md](LICENSING.md) · [Privacy Policy](legal/PRIVACY_POLICY.md) · [Terms & Refund](legal/TERMS_AND_REFUND.md) · [Third-Party Notices](legal/THIRD_PARTY_NOTICES.md) · [EULA](legal/EULA.txt) · [Licence Spec](legal/PROBE_LICENSING.md)
