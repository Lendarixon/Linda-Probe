# Third-party notices — LocalDocs

Our own code in `project_search/`, server, scripts: proprietary, (c) owner.
The source ZIP excludes the Python runtime and model weights. The 0.18 portable
ZIP includes an unmodified Python runtime, dependency packages, and two embedding
models: `intfloat/multilingual-e5-small` and
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0).
Original model cards and pinned revisions accompany the models. See
`licenses/MODEL_PROVENANCE.json`, `licenses/RUNTIME_INVENTORY.json`, Python license,
Apache-2.0 text and original dependency `.dist-info` license files.

| Component | License | Notes |
|---|---|---|
| faiss-cpu | MIT (Facebook) | include MIT notice |
| sentence-transformers, tokenizers, huggingface_hub | Apache-2.0 | |
| torch (CPU) | BSD-3-style | |
| numpy, scipy, scikit-learn, pypdf | BSD-3 | |
| python-docx, tqdm | MIT | include MIT notice |
| pypdfium2 (+ bundled PDFium) | Apache-2.0 / BSD-3 | |
| cryptography | Apache-2.0 or BSD (dual) | |
| Pillow | HPND (permissive) | |
| ldap3 | LGPL-3.0 | dynamic import only; replaceable via pip; include LGPL text reference |
| Tesseract OCR 5 | Apache-2.0 | executable/DLLs installed separately, not bundled |
| tessdata language data (rus/eng/pol/osd) | Apache-2.0 | included in portable; https://github.com/tesseract-ocr/tessdata_fast |
| fpdf2 (tests only) | LGPL-3.0 | not used by product runtime |
| lancedb, chromadb (bench only) | Apache-2.0 | test/benchmark deps, not product runtime |
| Google Gemini API (optional cloud OCR tier) | Google API Terms | opt-in only (`LOCALDOCS_OCR_CLOUD=1`); page images leave the machine; customer API key required |

LGPL components are used as shared Python packages (dynamic linking) and are
replaceable by the end user; no static bundling. Full license texts ship with
their PyPI distributions.

## Linda Probe 1.4.x bundled components

This section describes the 1.4.x bundle and supersedes the earlier 0.18 notes
where the component scope or licence differs (including the e5 model, tqdm and
fpdf2). The source ZIP still excludes the runtime and model weights. The list
below records the supplied bundle specification; the runtime, model files and
their licence inventories are not present in this isolated source workspace,
so exact installed versions, notices and build options are UNKNOWN here.
Original upstream licence texts and model notices govern each component.

| Component | Licence | Notices and source |
|---|---|---|
| RapidOCR and PP-OCRv6 models | Apache-2.0 | RapidOCR: https://github.com/RapidAI/RapidOCR ; PP-OCRv6 models: PaddleOCR, https://github.com/PaddlePaddle/PaddleOCR . Preserve model notices. |
| onnxruntime-directml | MIT | ONNX Runtime: https://github.com/microsoft/onnxruntime |
| opencv-python-headless | Apache-2.0; bundled FFmpeg: LGPL | OpenCV: https://github.com/opencv/opencv-python ; FFmpeg source: https://ffmpeg.org/download.html . Preserve the wheel's third-party notices; exact FFmpeg version and LGPL variant are UNKNOWN in this workspace. |
| pyclipper | MIT | https://pypi.org/project/pyclipper/ |
| shapely | BSD-3-Clause | https://pypi.org/project/shapely/ ; preserve bundled dependency notices as applicable. |
| pypdfium2 and bundled PDFium | Apache-2.0 / BSD-3-Clause | https://github.com/pypdfium2-team/pypdfium2 ; preserve PDFium third-party notices. |
| numpy, scipy, scikit-learn | BSD-3-Clause | https://pypi.org/project/numpy/ ; https://pypi.org/project/scipy/ ; https://pypi.org/project/scikit-learn/ |
| torch | BSD-3-Clause | https://github.com/pytorch/pytorch ; preserve bundled dependency notices. |
| transformers, sentence-transformers, tokenizers, safetensors, huggingface_hub | Apache-2.0 | Sources: https://github.com/huggingface/transformers ; https://github.com/UKPLab/sentence-transformers ; https://github.com/huggingface/tokenizers ; https://github.com/huggingface/safetensors ; https://github.com/huggingface/huggingface_hub |
| faiss-cpu | MIT | https://github.com/facebookresearch/faiss |
| cryptography | Apache-2.0 OR BSD-3-Clause | https://github.com/pyca/cryptography ; preserve bundled dependency notices. |
| certifi | MPL-2.0 | Upstream source: https://github.com/certifi/python-certifi ; source distributions: https://pypi.org/project/certifi/#files . |
| tqdm | MPL-2.0 / MIT | Preserve the upstream dual-licence notices and applicable file-level terms. Upstream source: https://github.com/tqdm/tqdm ; source distributions: https://pypi.org/project/tqdm/#files . |
| fpdf2 | LGPL-3.0 | Unmodified, replaceable Python library in `runtime\Lib\site-packages`. Corresponding upstream source distributions: https://pypi.org/project/fpdf2/#files . |
| ldap3 | LGPL-3.0 | Unmodified, replaceable Python library in `runtime\Lib\site-packages`. Corresponding upstream source distributions: https://pypi.org/project/ldap3/#files . |
| Jinja2, MarkupSafe, colorama, mpmath, sympy, omegaconf | BSD-3-Clause | Upstream source distributions: https://pypi.org/project/Jinja2/ ; https://pypi.org/project/MarkupSafe/ ; https://pypi.org/project/colorama/ ; https://pypi.org/project/mpmath/ ; https://pypi.org/project/sympy/ ; https://pypi.org/project/omegaconf/ |
| intfloat/multilingual-e5-small | MIT | Model card and source: https://huggingface.co/intfloat/multilingual-e5-small . This corrects the earlier grouped Apache-2.0 description for this model. |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | Model card and source: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| tessdata (rus/eng/pol/osd) | Apache-2.0 | https://github.com/tesseract-ocr/tessdata_fast |
| Linda OCR recognizer | Proprietary; (c) Vladyslav Manzyuk | Author's own trained model, licensed to users only as part of the Linda Probe application. This notice does not relicense third-party OCR models or libraries. |

The LGPL libraries above are shared Python packages, not statically linked into
the application, and may be replaced by the user in `runtime\Lib\site-packages`.
Their licences permit modification and redistribution under the applicable LGPL
terms, including reverse engineering for debugging modifications to those
libraries. The application's proprietary licence does not restrict those rights.
Obtain source matching the bundled package version from the linked PyPI source
distributions; the exact bundled versions are UNKNOWN in this workspace.
For MPL-covered files, source is available from the listed upstream repositories
and distributions; preserve the MPL notices and file-level rights. These notices
do not replace the full applicable licence texts or the notices of dependencies
included within binary wheels.
