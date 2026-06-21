# Fed Uncertainty Index — Research Repository

This repository contains all code developed for the empirical part of the thesis
*Generative and Agentic AI in Economics, and an Uncertainty Index from Federal
Reserve Speeches*. The project constructs monthly uncertainty indices from Federal
Reserve Board speeches (2006–2026) using four NLP methodologies of increasing
sophistication, and evaluates them against macroeconomic benchmarks and outcomes.

---

## Repository Structure
fed-uncertainty/

│

├── src/

│   ├── 01_scraping_fed_speeches.py

│   ├── 02_prepare_corpus.py

│   ├── 03_build_dictionary_index.py

│   ├── 04_build_tfidf_index.py

│   ├── 05_build_embedding_index.py

│   ├── 06_build_finbert_encoder_index.py

│   ├── 07_finbert_specification_selection.py

│   ├── 08_benchmarking_indices.py

│   ├── 09_benchmarking_plots.py

│   ├── 10_historical_validation.py

│   ├── 11_macro_validation.py

│   ├── 12_prediction_validation.py

│   └── 13_descriptive_figures.py

│

├── data/                           # Both external data (LM dictionary, EPU, VIX,...) and internal data (mainly .pkl files)

│

├── figures/                        # All generated plots (PNG, 300 dpi)

│

├── README.md

└── requirements.txt

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| Federal Reserve website | Speech texts (2006–2026) | Scraped automatically by script 01 |
| Loughran-McDonald Master Dictionary | Financial sentiment word lists | https://sraf.nd.edu/loughranmcdonald-master-dictionary/ |
| Baker, Bloom & Davis EPU | Economic Policy Uncertainty index | https://www.policyuncertainty.com |
| Baker, Bloom & Davis MPU | Monetary Policy Uncertainty sub-index | https://www.policyuncertainty.com |
| CBOE VIX | Volatility index (daily close) | https://finance.yahoo.com or CBOE website |
| FRED | GDP, unemployment, CPI, industrial production, consumer sentiment, S&P 500, recession indicator | Downloaded automatically via API in scripts 11 and 12 |

The Loughran-McDonald master dictionary, EPU, MPU, and VIX files must be downloaded manually and placed in the `data/` folder before running the pipeline (currently they are, with the most up-to-date versions on 06-2026). All FRED macroeconomic series are downloaded programmatically at runtime.

---

## Execution Order

Scripts must be run in the order shown below. Each script depends on the outputs of the preceding ones.

01_scraping_fed_speeches.py

↓

02_prepare_corpus.py

↓

03_build_dictionary_index.py

04_build_tfidf_index.py

05_build_embedding_index.py

06_build_finbert_encoder_index.py

↓

07_finbert_weight_search.py

↓

08_benchmarking_indices.py

↓

09_benchmark_plots.py       (can run in parallel with 10, 11, 12, 13)

10_historical_validation.py

11_macro_validation.py

12_statistical_validation.py

13_descriptive_figures.py


Scripts 03–06 are independent of each other and can be run in any order once script 02 has completed. 
However, all outputs are stored to .pkl file, so any script can be run in any order. 
The execution order need only be followed when deleting or wanting to overwrite the data/ files.

---

## Script Descriptions

### 01 — Scraping Fed Speeches
**Purpose:** Collects all speeches published by the Board of Governors of the Federal Reserve System from the official Federal Reserve website (2006–2026).  
**Inputs:** Federal Reserve website (federalreserve.gov)  
**Outputs:** `data/all_fed_speeches` (pickle), `fed_speeches.xlsx`  
**Notes:** The script handles two URL formats used by the Fed site before and after 2011. HTML structure inconsistencies across years are handled through a cascade of fallback CSS selectors. Video transcripts and working paper links are excluded automatically.

---

### 02 — Corpus Preparation
**Purpose:** Transforms the raw speech corpus into a clean, structured NLP corpus with metadata and preprocessed text representations.  
**Inputs:** `data/all_fed_speeches`  
**Outputs:** `data/corpus.pkl`  
**Notes:** Creates four text representations per speech (raw, cleaned, tokenized, lemmatized with stopwords removed). Stopword removal is applied to the lemmatized tokens used by dictionary and TF-IDF methods; embedding and FinBERT methods use the cleaned text without stopword removal to preserve linguistic context. The `is_chair` binary flag identifies speeches by the Board Chair, explicitly excluding the Vice Chair. spaCy's `en_core_web_sm` model is required for context-aware lemmatization.

---

### 03 — Dictionary Uncertainty Index
**Purpose:** Constructs monthly uncertainty indices using the Loughran-McDonald financial sentiment dictionary.  
**Inputs:** `data/corpus.pkl`, `data/lm_uncertainty.txt`  
**Outputs:** `data/uncertainty_index.pkl`, `data/uncertainty_index_chair.pkl`, `data/speeches_with_uncertainty.pkl`  
**Notes:** The LM master dictionary must be downloaded separately (currently in data/); the word list extraction code is provided as a comment in the script. Uncertainty is measured as the proportion of lemmatized tokens matching the LM uncertainty category. Two variants are produced: full-Board and Chair-only. Both are normalized to z-scores; raw proportions are also preserved.

---

### 04 — TF-IDF Uncertainty Index
**Purpose:** Constructs a corpus-weighted uncertainty index using TF-IDF scoring over the Loughran-McDonald uncertainty vocabulary.  
**Inputs:** `data/corpus.pkl`, `data/lm_uncertainty.txt`  
**Outputs:** `data/tfidf_uncertainty_index.pkl`, `data/tfidf_speeches.pkl`  
**Notes:** The TF-IDF matrix is fitted over the entire corpus, meaning weights are in-sample by construction. The uncertainty score for each speech is the sum of TF-IDF weights for uncertainty vocabulary terms present in that speech. A single pooled (full-Board) variant is produced; the Chair restriction did not add discriminating power for this method.

---

### 05 — Semantic Embedding Index (MiniLM)
**Purpose:** Constructs uncertainty indices using semantic similarity to a three-centroid architecture in the all-MiniLM-L6-v2 embedding space.  
**Inputs:** `data/corpus.pkl`  
**Outputs:** `data/embedding_index.pkl`, `data/embedding_chair_index.pkl`, `data/embedding_pure_index.pkl`, `data/embedding_chair_pure_index.pkl`  
**Notes:** Each speech is divided into 100-word sentence-preserving chunks. Each chunk is independently encoded and compared against three centroids: uncertainty, bad-news, and academic discussion of uncertainty. The final speech score subtracts weighted projections onto the confounding centroids. Two weight specifications are produced (balanced: α=0.5, β=0.5; pure: α=0.25, β=1.0), each with full-Board and Chair-only variants. This script has significant computational cost; runtime depends on hardware but expect up to 30 minutes on CPU.

---

### 06 — FinBERT Encoder Index
**Purpose:** Constructs uncertainty indices using the same three-centroid architecture as script 05 but with the domain-adapted FinBERT encoder (yiyanghkust/finbert-tone).  
**Inputs:** `data/corpus.pkl`  
**Outputs:** `data/finbert_encoder_index.pkl`, `data/finbert_encoder_chair_index.pkl`, `data/finbert_encoder_pure_index.pkl`, `data/finbert_encoder_chair_pure_index.pkl`, `data/finbert_similarity_components.pkl`  
**Notes:** FinBERT is used exclusively as an encoder; its sentiment classification head is not used. Mean pooling of the last hidden layer produces sentence-level embeddings. Raw similarity components (sim_u, sim_b, sim_d) are saved separately to enable the weight search in script 07 without re-running the expensive encoding step. This is the most computationally intensive script in the pipeline; the script is configured to use all available CPU threads. GPU acceleration can reduce runtime substantially.

---

### 07 — FinBERT Weight Search and Specification Selection
**Purpose:** Performs an exhaustive grid search over 64 weight combinations to identify the most economically plausible FinBERT index specifications.  
**Inputs:** `data/finbert_similarity_components.pkl`  
**Outputs:** `data/finbert_index_A.pkl`, `data/finbert_chair_index_A.pkl`, `data/finbert_index_B.pkl`, `data/finbert_chair_index_B.pkl`, `data/finbert_index_C.pkl`, `data/finbert_chair_index_C.pkl`, `data/finbert_index_D.pkl`, `data/finbert_chair_index_D.pkl`  
**Notes:** Because similarity components are stored from script 06, the grid search does not require re-encoding. Specifications are evaluated by comparing their top-10 months against a calendar of historically recognized high-uncertainty episodes. Four specifications (A, B, C, D) were selected based on this face-validity criterion. The selection rationale and historical episode calendar are documented in comments within the script.

---

### 08 — Benchmarking Against External Indices
**Purpose:** Merges all Fed uncertainty indices with external benchmarks (EPU, MPU, VIX) into a single analysis dataset and computes pairwise correlations.  
**Inputs:** All 19 index pickle files, `data/epu.xlsx`, `data/mpu.xlsx`, `data/vix.xlsx`  
**Outputs:** `data/benchmark_dataset.pkl`, printed correlation tables and rankings  
**Notes:** EPU and MPU data must be downloaded from policyuncertainty.com and placed in the data folder. VIX daily data is aggregated to monthly frequency by taking the within-month mean. All external indices are standardized to z-scores. The within-family correlation heatmap for Fed indices is also produced here.

---

### 09 — Benchmark Visualizations
**Purpose:** Generates all time-series comparison plots between Fed indices and external benchmarks.  
**Inputs:** `data/benchmark_dataset.pkl`  
**Outputs:** Multiple PNG figures in `figures/`  
**Notes:** Produces both dual-axis and standardized z-score comparison plots for EPU, MPU, and VIX against selected Fed indices. Chair versus full-Board comparison plots are also generated. All figures are saved at 300 dpi.

---

### 10 — Historical Validation
**Purpose:** Identifies the top-10 highest-uncertainty months for each index and analyzes cross-index agreement.  
**Inputs:** All 19 index pickle files  
**Outputs:** `data/historical_validation_top_months.pkl`, `data/historical_validation_frequency.pkl`, multiple PNG figures  
**Notes:** Produces a frequency table of months appearing across multiple top-10 lists, a heatmap of top months by index (colour-coded by year), and a binary consensus heatmap showing which months any index flagged. The heatmap colour encodes year rather than rank to reveal temporal clustering by methodological family.

---

### 11 — Macroeconomic Validation
**Purpose:** Computes lead-lag correlations between each Fed uncertainty index and seven macroeconomic variables at horizons of 1, 3, 6, and 12 months.  
**Inputs:** All 19 index pickle files; macroeconomic data downloaded at runtime from FRED  
**Outputs:** `data/macro_benchmark_results.csv`, multiple PNG figures  
**Notes:** Macroeconomic series (GDP growth, unemployment, CPI inflation, industrial production, consumer sentiment, S&P 500 returns, NBER recession indicator) are retrieved directly from the FRED API and require an internet connection. The predictive score used for ranking is the mean absolute correlation across the 3-, 6-, and 12-month forward horizons. Visual inspection plots apply 3-month and 6-month moving averages to the uncertainty index to reduce high-frequency noise; the macro variable is shifted forward by the horizon shown in the figure title.

---

### 12 — Statistical Validation
**Purpose:** Performs formal statistical tests: ROC-AUC classification, controlled OLS forecast regressions with Newey-West standard errors, and Granger causality tests.  
**Inputs:** All 19 index pickle files; macroeconomic data downloaded at runtime from FRED  
**Outputs:** Multiple PNG figures; printed regression and Granger results tables  
**Notes:** Recession and GDP contraction classification targets are defined as binary indicators six months ahead. Forecast regressions include the current level of the macro variable as a control and use HAC standard errors with six lags. Granger tests are run up to 12 lags and report the best (lowest) p-value across all lags. The overall ranking combines recession AUC, GDP contraction AUC, best forecast p-value, and best Granger p-value into a composite rank score.

---

### 13 — Descriptive Figures
**Purpose:** Produces all descriptive visualizations of the corpus and preprocessing pipeline used in the thesis.  
**Inputs:** `data/all_fed_speeches`, `data/corpus.pkl`  
**Outputs:** Multiple PNG figures in `figures/`  
**Notes:** Generates speech frequency plots by quarter (full corpus and Chair-only by Chair), wordclouds before and after preprocessing, speaker-specific wordclouds for Powell and Bernanke, top-20 word frequency tables, and a step-by-step preprocessing pipeline illustration. This script can be run at any point after script 02 and does not depend on any index outputs.

---

## Computational Requirements

| Script | Estimated Runtime | Notes |
|--------|------------------|-------|
| 01 | 20-30 minutes | Network-bound; rate limits may vary |
| 02 | 5-10 minutes | Depends on corpus size and spaCy speed |
| 03 | < 1 minute | |
| 04 | < 1 minute | |
| 05 | 20-30 minutes | CPU-bound; GPU reduces to ~5 minutes |
| 06 | 1 hour | Most expensive; GPU strongly recommended |
| 07 | < 1 minute | Cheap; encoding done in script 06 |
| 08–13 | < 1 minutes each | |

Scripts 05 and 06 are the primary computational bottlenecks. If GPU acceleration is available, set the device in the model loading calls accordingly.

---

## Reproducing Results

```bash
# 1. Clone the repository
git clone [repository-url]
cd fed-uncertainty-thesis

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model
python -m spacy download en_core_web_sm

# 5. Place external data files in data/
#    - Loughran-McDonald_MasterDictionary_1993-2025.xlsx
#    - epu.xlsx
#    - mpu.xlsx
#    - vix.xlsx

# 6. Run the pipeline in order
python src/01_scraping_fed_speeches.py
python src/02_prepare_corpus.py
python src/03_build_dictionary_index.py
python src/04_build_tfidf_index.py
python src/05_build_embedding_index.py
python src/06_build_finbert_encoder_index.py
python src/07_finbert_weight_search.py
python src/08_benchmarking_indices.py
python src/09_benchmark_plots.py
python src/10_historical_validation.py
python src/11_macro_validation.py
python src/12_statistical_validation.py
python src/13_descriptive_figures.py
```

If you wish to skip the scraping step, the raw corpus file `all_fed_speeches` is available upon request or can be reproduced by running script 01 in full.

---

## Citation

If you use this code or the resulting indices in your own work, please cite the thesis:

> Tobella, P. (2026). *Implications of Generative AI for the Work of Economists: Constructing an Uncertainty Index from Federal Reserve Speeches Using Natural Language Processing (NLP)*. Undergraduate thesis, Universitat de Barcelona / Universitat Politècnica de Catalunya.

The scraping framework in script 01 was originally developed by Smith (2020) and substantially adapted for this project:

> Smith, D. (2020). NLP-Fed-Speeches. GitHub repository. https://github.com/davidjsmith44/NLP-Fed-Speeches
