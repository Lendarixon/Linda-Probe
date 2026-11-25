![Linda Probe logo](images/logo.png)

# Linda Probe — Instant Local File Search, Offline OCR & AI Photo Search for Windows

**Find any document, scanned PDF, or photo on your PC in milliseconds — 100% offline, zero telemetry, one-time purchase.**

[**Get from Microsoft Store (Signed)**](https://apps.microsoft.com/detail/9PGM1V68193X) · [**Start 7-Day Pro Trial in App (No Card)**](#pricing) · [**Project Page**](https://lendarixon.github.io/Linda/probe.html?utm_source=github)

---

## Why Linda Probe 2.0?

Most desktop search tools either miss text inside scans and PDFs, or force you to upload private files to a cloud subscription. **Linda Probe** combines a native C# filename engine with hybrid AI semantic search, offline OCR, and local photo search — all running privately on your Windows PC.

- **Instant Name & Exact Content Search (Free):** Sub-50ms filename lookups across millions of files plus exact-word full-text search, Windows 11-style file explorer, rich previews, and manual tags.
- **Hybrid AI Semantic Search (Pro):** Search by meaning, not just exact keywords (including cross-language queries) powered by local FAISS + SQLite FTS5.
- **Offline OCR for Scans & Images (Pro):** Automatically read scanned PDFs and images locally with RapidOCR / Tesseract — zero cloud uploads.
- **AI Photo Search & Auto-Tags (Pro):** Find photos by text description or by dropping an example image, plus automatic local tagging.
- **Built for Speed & Power Users (New in 2.0):**
  - **Modern Fluent 2.0 UI** in Dark and Light themes (interface available in English, Russian, and Polish).
  - **Command Palette (`Ctrl+K`)** to jump to folders, tabs, recent searches, themes, and diagnostics.
  - **Live Instant Filename Suggestions** as you type, **Export Results** (CSV / Markdown / JSON), and **1-Click Copy Citation** (`filename + page + snippet`).
  - **Built-in System Diagnostics** and a redesigned Pro storefront with a **7-day offline Pro trial**.

> **100% Private by Design:** Your documents never leave your device. No telemetry, no usage tracking, no crash uploads, no forced cloud account.

---

## Free vs Pro

Neither edition has tariff limits on file count or processing speed.

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

---

## Pricing & 7-Day Free Trial

Try every Pro feature risk-free: start a **7-day Pro trial** directly inside the app with one click — **offline, no credit card, no automatic subscription**. When the trial or a temporary promo expires, the app cleanly returns to Free while keeping all your documents, indexes, and downloaded models.

| Pro Licence | Price | Users & Devices | Updates |
|---|---|---|---|
| [**Personal**](https://buy.polar.sh/polar_cl_ocKiMPEuNUcBaYd1VuznOqJG0Nba621QOuHgn3wqovL) | **USD 10** once | 1 named user, up to 3 devices | Purchased version & fixes perpetually |
| [**Team**](https://buy.polar.sh/polar_cl_f2kNSLkqbeCs4C1XVl2kNDPzXugLdBBEzE3FF3KzGvj) *(Best Value)* | **USD 30** once | Up to 10 named users, up to 20 devices | Purchased version & fixes perpetually |
| [**Organization**](https://buy.polar.sh/polar_cl_9lYGni5ADbhnxtklKBjAj0WuwWPozXw3nTKwc3LMc4q) | **USD 500**/year | 1 organization at 1 site, up to 200 devices | All updates during paid year |

- **14-Day Full Refund:** Request a full refund within 14 days of purchase via [lindapro.support@proton.me](mailto:lindapro.support@proton.me) or Polar. Mandatory consumer rights remain unaffected.
- **Merchant of Record:** Polar processes checkout, tax, and invoicing (final total shown before payment). Operator: Vladyslav Manzyuk, Poland.
- *Personal/Team:* Perpetual for the purchased major version and its fixes; new major versions or models may require a paid upgrade. *Organization:* Renews yearly until cancelled in the Polar customer portal; after expiry, Pro rights end and Free features remain available.

---

## Download & Installation

- **Recommended (Signed):** [**Install from Microsoft Store**](https://apps.microsoft.com/detail/9PGM1V68193X)
- **Standalone Windows x64 Installer (Latest published build 1.4.10):** [**Download LindaProbe-Setup-1.4.10.exe**](https://github.com/Lendarixon/Linda-Probe/releases/download/v1.4.10/LindaProbe-Setup-1.4.10.exe)

SHA-256 (`LindaProbe-Setup-1.4.10.exe`):
```text
54489279e04992120686bcc8c2221feb206cb1abd768c1dbd7331d426e88262f
```

Verify the SHA-256 hash before running the standalone installer. Because the standalone `.exe` installer is not yet code-signed, Windows SmartScreen may show a warning — after verifying the source and hash, select **More info -> Run anyway**.

**System Requirements:** Windows 10/11 x64. Optional AI semantic search and OCR models are downloaded after activation and require an internet connection, compatible hardware, and sufficient disk space.

---

## Activation, Trials & Promo Keys

- **7-Day Pro Trial & Promo Keys (Offline):** Start a 7-day trial in the app or enter a temporary promo key (`PROBE-PROMO-...`). Both work offline, require no network connection, send nothing to Polar, and grant Personal Pro rights (including commercial use for 1 person on up to 3 devices) through the displayed UTC expiry date (inclusive). One trial is available per Windows user and machine; removing a key does not restart or restore it.
  - *Transparency note:* Windows DPAPI protects the machine-bound local access cache. A local marker in the user data folder and a registry marker under `HKCU\Software\LindaProbe` record trial usage solely to prevent repeat trials on the same user/machine. Uninstalling retains the data folder unless you choose to delete indexes and models (which removes the file marker, while the `HKCU` marker remains). Microsoft Store and installer builds may keep separate markers.
- **Purchased Keys (Online Activation):** Enter your Probe key in Settings -> Licence (Linda detector keys do not unlock Probe). Polar receives the key and a Probe-specific hashed machine ID (never document contents). Perpetual keys continue working offline when validation cannot reach the network; Organization keys expire at the cached end of the paid term. Confirmed revocations lock Pro on the next successful check. Before uninstalling or moving a device, click **Remove key** while online to release your device slot (3/20/200).
- **Gift Keys:** Signed Ed25519 perpetual Personal vouchers verified completely offline (no server-side device counter).

---

## Privacy & Licensing

- **Zero Document Uploads:** Indexing, search, OCR, and photo recognition run locally on your machine.
- **Minimal Network Calls:** Only purchased-key activation/validation (Polar) and optional post-activation model downloads (Hugging Face, which sees your IP address like any download) use the network.
- **Licence Model:** Personal, non-commercial use of Linda Probe's free features (no key) is licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode). Commercial use (paid work or clients, use inside a company, school, or organization, or in a product/service you sell) and all Pro features require a paid or gift licence key, an active trial, or an active temporary promo.

### Legal Documents & Support

[Privacy Policy](legal/PRIVACY_POLICY.md) · [Terms & Refund](legal/TERMS_AND_REFUND.md) · [Third-Party Notices](legal/THIRD_PARTY_NOTICES.md) · [EULA](legal/EULA.txt) · [Licence Spec](legal/PROBE_LICENSING.md)

Third-party components and model weights retain their own licences; Pro licenses the Linda Probe application workflow, not exclusive ownership of third-party models.

**Support & Refunds:** [lindapro.support@proton.me](mailto:lindapro.support@proton.me)
