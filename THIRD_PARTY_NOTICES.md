# Third-party notices

MES is distributed under the licence in [LICENSE](LICENSE). The material below is not covered by it, or carries its
own licence.

## Included in this repository

| Material | Where | Licence / status |
|---|---|---|
| mimic-code: MIMIC-IV schema (`create.sql`) and DuckDB concept SQL, from [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code) at the commit in `SOURCE.md` | `mes/tools/rwd/vendor/mimic_code/` | MIT License, Copyright (c) 2019 MIT Laboratory for Computational Physiology — see `mes/tools/rwd/vendor/mimic_code/LICENSE` |
| Weill Cornell Medicine logo | `mes/ui/assets/WCM_MB_LOGO_HZSS2L_CLR_RGB.png` (header and favicon) | a trademark of Weill Cornell Medicine; not licensed under this repository's licence |
| Inline SVG icons | `mes/ui/icons.py` | drawn in the Lucide / Feather style; any path taken from those sets is under their licences (Lucide: ISC, Feather: MIT) |

## Used at run time, not included

| Material | Licence / terms |
|---|---|
| Python packages in `requirements.txt` | their own licences, e.g. Streamlit, transformers, sentence-transformers, boto3, openai, pyarrow, requests (Apache-2.0); DuckDB, lifelines, openai-agents, LiteLLM, markdown-it-py, truststore, urllib3, pydantic, pytest (MIT); NumPy, pandas, SciPy, scikit-learn, statsmodels, PyTorch, psutil, python-dotenv, websockets, Altair (BSD); Pillow (MIT-CMU). Check the metadata of the installed versions. |
| **PyMuPDF** (PDF text extraction) | **AGPL-3.0** or a commercial licence from Artifex. If you offer MES to others over a network, the AGPL terms apply to PyMuPDF. |
| Sentence encoder `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) | Apache-2.0 |
| Inter typeface (loaded from Google Fonts by `.streamlit/config.toml`) | SIL Open Font License 1.1 |
| Language models: the OpenAI API, or a local model served by LM Studio (e.g. Gemma) | the terms of the provider and of the model you use |
| NCBI E-utilities (PubMed), Europe PMC, Unpaywall, ClinicalTrials.gov API | the usage policies of each service (NCBI asks for an API key or at most 3 requests/s and a contact e-mail; Unpaywall requires an e-mail). Retrieved abstracts and full texts keep their own copyright and licences; MES caches them locally in `.cache/`. |
| iBKH knowledge graph | see [docs/DATA.md](docs/DATA.md): derived from many biomedical sources with their own terms |
| MIMIC-IV | PhysioNet Credentialed Health Data License and data use agreement; never redistributed — see [docs/DATA.md](docs/DATA.md) |
