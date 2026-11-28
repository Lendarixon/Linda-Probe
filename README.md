<p align="center"><img src="images/readme_banner.svg" alt="Linda Probe — local Windows file search, offline OCR and photo search" width="100%"></p>

# Linda Probe — local Windows file search, offline OCR and photo search (English, Polish, Russian)

Free for personal non-commercial use, **paid for Pro and commercial use** — one licence covers the app, the command-line tools, and the local OCR/search workflows ([LICENSING.md](LICENSING.md), [PROBE_LICENSING.md](legal/PROBE_LICENSING.md), contact lindapro.support@proton.me). Runs on your machine: no cloud uploads, no telemetry.
Our OCR recognition model lives on Hugging Face: [Lindarixon/linda-ocr](https://huggingface.co/Lindarixon/linda-ocr) (Apache-2.0).

## Latest Windows update: 2.0.0

[Download the Windows release](https://github.com/Lendarixon/Linda-Probe/releases/tag/v2.0.0) · [Microsoft Store](https://apps.microsoft.com/detail/9PGM1V68193X) · [Project page](https://lendarixon.github.io/Linda/probe.html?utm_source=github)

App **2.0.0** (Fluent 2.0 interface overhaul, `Ctrl+K` Command Palette, live suggestions and diagnostics):
- **Modern Fluent 2.0 UI:** Dark and Light themes across English, Russian, and Polish.
- **Command Palette (`Ctrl+K`):** jump directly to tabs, folders, recent searches, themes, and system diagnostics.
- **Faster workflows:** live debounced filename suggestions as you type, incremental results rendering, export to CSV / Markdown / JSON, and 1-click Copy Citation (`filename + page + snippet`).
- **Diagnostics & storefront:** built-in `/doctor` system diagnostics and a redesigned Pro & 7-day offline trial storefront with a status bar plan badge.

## Also from this project

[Linda](https://github.com/Lendarixon/Linda) is a local detector of AI-generated text for English, Polish, and Russian, with built-in **Humanizer** and **Trace** (writing-style comparison). Both apps run entirely on your PC.

## Windows app

Get it from the [Microsoft Store](https://apps.microsoft.com/detail/9PGM1V68193X) (recommended, signed), or download **LindaProbe-Setup-2.0.0.exe** from the [releases page](https://github.com/Lendarixon/Linda-Probe/releases/download/v2.0.0/LindaProbe-Setup-2.0.0.exe) (Windows 10/11 x64).

SHA-256 (`LindaProbe-Setup-2.0.0.exe`):
```text
2d4f7fd304e78dc424fb30c4e1a07bf411f99f644f47cd77b24fa61be5cc97bf
```

Verify the SHA-256 hash before running the standalone installer. Because the standalone `.exe` installer is not yet code-signed, Windows SmartScreen may show a warning — after verifying the source and hash, select **More info -> Run anyway**. Optional AI semantic search and OCR models are downloaded after activation and require an internet connection, compatible hardware, and sufficient disk space.

## Command line / Python

Install locally in editable mode (`pip install -e .`) or run directly with Python 3.10+:

```bash
python -m linda_probe find "invoice" --root C:\Docs --format table
python -m linda_probe index C:\Docs
python -m linda_probe search C:\Docs "refund policy" --limit 5 --format md
python -m linda_probe dupes C:\Docs --min-size 1024 --format json
python -m linda_probe settings show
python -m linda_probe models
python -m linda_probe doctor
python -m linda_probe serve --port 8798
```

All list/search commands support `--format table|json|csv|md`.

## Two editions: Free and Pro

The same app, two capability levels. Neither edition has tariff limits on file count or processing speed.
* **Free:** sub-50ms filename lookups, exact-word full-text search (SQLite FTS5), Windows 11-style file explorer, previews, export, citations, manual tags, and duplicate finder.
* **Pro:** hybrid AI semantic search (FAISS + FTS5, including cross-language queries), offline OCR for scans and PDFs (RapidOCR / Tesseract / `linda-ocr`), photo search by text description or example image, automatic local tags, and commercial use.

| Feature | Free | Pro |
|---|---|---|
| Instant filename & exact-word content search | Yes | Yes |
| File explorer, previews, export & citations | Yes | Yes |
| Manual tags & duplicate finder | Yes | Yes |
| Hybrid AI semantic search | — | Yes |
| Offline OCR for scans & searchable PDFs | — | Yes |
| Photo search by description or example image | — | Yes |
| Automatic local tags | — | Yes |
| Commercial / workplace use | — | Yes |

## Commercial licence and pricing

| Pro Licence | Price | Users & Devices | Updates |
|---|---|---|---|
| [**Personal**](https://buy.polar.sh/polar_cl_ocKiMPEuNUcBaYd1VuznOqJG0Nba621QOuHgn3wqovL) | **USD 10** once | 1 named user, up to 3 devices | Purchased version & fixes perpetually |
| [**Team**](https://buy.polar.sh/polar_cl_f2kNSLkqbeCs4C1XVl2kNDPzXugLdBBEzE3FF3KzGvj) | **USD 30** once | Up to 10 named users, up to 20 devices | Purchased version & fixes perpetually |
| [**Organization**](https://buy.polar.sh/polar_cl_9lYGni5ADbhnxtklKBjAj0WuwWPozXw3nTKwc3LMc4q) | **USD 500**/year | 1 organization at 1 site, up to 200 devices | All updates during paid year |

- **14-day full refund:** via [lindapro.support@proton.me](mailto:lindapro.support@proton.me) or Polar. Mandatory consumer rights remain unaffected.
- **Merchant of Record:** Polar processes checkout, tax, and invoicing. Operator: Vladyslav Manzyuk, Poland.
- *Personal/Team:* perpetual for the purchased major version and its fixes; new major versions or models may require a paid upgrade. *Organization:* renews yearly until cancelled in the Polar customer portal; after expiry, Pro rights end and Free features remain available.

## Activation, trials & privacy

- **Zero document uploads:** indexing, search, OCR, and photo recognition run locally on your machine.
- **7-day Pro trial & promo keys (offline):** start a 7-day trial in the app or enter a temporary promo key (`PROBE-PROMO-...`). Both work offline, send nothing to Polar, and grant Personal Pro rights through the displayed UTC expiry date (inclusive). One trial is available per Windows user and machine; Windows DPAPI protects the machine-bound local access cache, and a local marker in the user data folder plus `HKCU\Software\LindaProbe` record trial usage solely to prevent repeat trials.
- **Purchased keys (online activation):** enter your Probe key in Settings -> Licence (Linda detector keys do not unlock Probe). Polar receives the key and a Probe-specific hashed machine ID (never document contents). Click **Remove key** while online before moving devices to release a slot.
- **Gift keys:** signed Ed25519 perpetual Personal vouchers verified completely offline.
- **Licence model:** personal, non-commercial use of Free features is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). Commercial use and Pro features require a paid/gift key, active trial, or active promo.

## Legal documents and support

[Privacy Policy](legal/PRIVACY_POLICY.md) · [Terms & Refund](legal/TERMS_AND_REFUND.md) · [Third-Party Notices](legal/THIRD_PARTY_NOTICES.md) · [EULA](legal/EULA.txt) · [Licence Spec](legal/PROBE_LICENSING.md)

**Support & Refunds:** [lindapro.support@proton.me](mailto:lindapro.support@proton.me)
