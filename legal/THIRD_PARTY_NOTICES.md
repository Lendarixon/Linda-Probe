# Third-party notices — Linda Probe 1.4.5

Linda Probe's own code (`project_search/`, the server, the interface and scripts) and the **Linda OCR** recognizer are proprietary, (c) Vladyslav Manzyuk. Linda OCR is an original model trained by the author, who holds all rights to the model and its training data.

Linda Probe ships an unmodified Python 3.12 runtime and the packages below; each keeps its own licence files in `runtime\Lib\site-packages\<package>.dist-info\`. Optional AI models are downloaded after activation from Hugging Face (huggingface.co) and keep the licences of their authors. Paid features license the Linda Probe application workflow only, not third-party components.

## Bundled components


| Component | Licence | Notices and source |
|---|---|---|
| RapidOCR and PP-OCRv6 models | Apache-2.0 | RapidOCR: https://github.com/RapidAI/RapidOCR ; PP-OCRv6 models: PaddleOCR, https://github.com/PaddlePaddle/PaddleOCR . Preserve model notices. |
| onnxruntime-directml | MIT | ONNX Runtime: https://github.com/microsoft/onnxruntime |
| opencv-python-headless | Apache-2.0; bundled FFmpeg: LGPL | OpenCV: https://github.com/opencv/opencv-python ; FFmpeg source: https://ffmpeg.org/download.html . Preserve the wheel's third-party notices; FFmpeg is a separate replaceable DLL. |
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
| olefile 0.47 (vendored in `project_search/_ext/olefile`) | BSD-2-Clause, with inherited PIL licence terms | Copyright (c) Philippe Lagadec; full terms retained in the file headers |
| GEOS (in shapely.libs), FFmpeg (in cv2) | LGPL-2.1 | separate replaceable DLLs; texts in `shapely-*.dist-info\licenses\LICENSE_GEOS` and `cv2\LICENSE-3RD-PARTY.txt` |
| OpenBLAS, LAPACK, GCC runtime in numpy/scipy/scikit-learn/faiss wheels | BSD-3-Clause; GCC runtime GPL-3.0-or-later WITH GCC-exception-3.1 | the exception permits redistribution of the runtime; texts in each package's licence file |
| Intel IPP (in cv2), Intel OpenMP (in torch), Microsoft DirectML and Visual C++ runtime DLLs | vendor redistribution terms | redistributed unmodified as part of the packages above |
| Linda OCR recognizer | Proprietary; (c) Vladyslav Manzyuk | Author's own trained model, licensed to users only as part of the Linda Probe application. This notice does not relicense third-party OCR models or libraries. |

The LGPL libraries above are shared Python packages, not statically linked into
the application, and may be replaced by the user in `runtime\Lib\site-packages`.
Their licences permit modification and redistribution under the applicable LGPL
terms, including reverse engineering for debugging modifications to those
libraries. The application's proprietary licence does not restrict those rights.
Source for the bundled versions is available from the linked PyPI source distributions.
For MPL-covered files, source is available from the listed upstream repositories
and distributions; preserve the MPL notices and file-level rights. These notices
do not replace the full applicable licence texts or the notices of dependencies
included within binary wheels.
