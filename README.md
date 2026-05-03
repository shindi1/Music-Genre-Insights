# Cade's Data Pipeline — Music Sentiment & Genre Classification

Production-grade data scraping, cleaning, matching, and preprocessing pipeline for the
group project **"Sentiment Analysis in Music"** (Sam Shindich, Cade Miller, Larry Li).

This package is **Cade's** deliverable. It produces clean, balanced, train/val/test-split
parquet files that:

- **Sam** consumes (`data/processed/full.parquet`) for unsupervised K-Means clustering.
- **Larry** consumes (`data/processed/{train,val,test}.parquet`) for supervised models
  (logistic/ridge/lasso baseline, XGBoost, KNN, and lyrics-based NLP models).

---

## Why this design

The project proposal originally planned to query the **Spotify Web API** for audio
features and the **Genius API** for lyrics. As of **November 2024**, Spotify deprecated
the Audio Features, Recommendations, Related Artists, and Audio Analysis endpoints for
new apps — and as of February 2026 even Development Mode requires a Premium subscription
on a capped 5-test-user developer account.

Translation: the live-API path is dead for a one-month student project. So this pipeline
takes the workaround the research doc recommends:

- **Audio features** → "Spotify Tracks Dataset" Kaggle CSV (~114K tracks, 125 micro-genres,
  full audio-feature suite — `danceability`, `energy`, `valence`, `tempo`, etc.).
- **Lyrics + genre tags** → "Genius Song Lyrics" Kaggle CSV (~5M songs, full lyrics text,
  Genius `tag` ≈ broad genre).
- **The two are joined** via fuzzy matching on `(artist, title)` using `rapidfuzz` with
  artist-blocking to make the join tractable on a laptop.
- **Genius / Spotify APIs** are still wired in (`src/api_scraper.py`) but scoped to
  endpoints that actually still work — used only for filling underrepresented genres,
  not as the primary source.

The final taxonomy is the **6 canonical genres** the research doc recommends as the
sweet spot for lyric-based classification: `pop`, `hip-hop`, `rock`, `r&b`, `country`,
`electronic`. Lyric-only classification tops out at ~50–65% on this taxonomy and drops
to 30–50% on 10 genres — staying at 6 keeps Larry's models honest.

---

## Project layout

```
cade_data_pipeline/
├── README.md                        # ← you are here
├── Makefile                         # one-liners: make install / test / run
├── requirements.txt                 # pinned deps
├── pyproject.toml                   # `pip install -e .` support
├── config.py                        # all paths, thresholds, genre maps
├── .env.example                     # template for API credentials
├── .gitignore                       # never commit data/ or lyrics
│
├── src/                             # importable library code
│   ├── __init__.py                  # re-exports public API
│   ├── utils.py                     # logging, timer, parquet I/O
│   ├── data_loader.py               # memory-efficient Kaggle CSV loaders
│   ├── lyrics_cleaner.py            # ftfy + regex pipeline for Genius cruft
│   ├── language_filter.py           # langdetect wrapper, English-only
│   ├── genre_mapper.py              # Spotify/Genius micro-genres → 6 canonical
│   ├── dataset_matcher.py           # fuzzy (artist, title) join — the hard one
│   ├── feature_engineering.py       # lyric features + audio scaling
│   ├── balancer.py                  # class balancing (under/cap/none)
│   ├── splitter.py                  # stratified, leakage-safe split
│   ├── api_scraper.py               # Genius + (limited) Spotify clients
│   └── pipeline.py                  # 10-stage orchestrator
│
├── scripts/                         # CLI entrypoints
│   ├── run_pipeline.py              # main: end-to-end build
│   ├── scrape_supplemental.py       # fill underrepresented genres via API
│   ├── validate_output.py           # QA gate: schemas, ranges, leakage
│   └── eda.py                       # generate plots + tables for write-up
│
├── tests/                           # pytest suite
│   ├── conftest.py                  # fixtures (messy lyrics, tiny dfs)
│   ├── test_lyrics_cleaner.py
│   ├── test_dataset_matcher.py
│   └── test_genre_mapper.py
│
├── data/
│   ├── raw/                         # ← put Kaggle CSVs here (gitignored)
│   ├── interim/                     # per-stage intermediate parquets
│   ├── processed/                   # ← final outputs for Sam & Larry
│   └── cache/                       # API response cache
│
└── logs/                            # rotating run logs
```

---

## Setup (5 minutes)

### 1. Install

```bash
make install              # creates .venv (optional) and installs requirements.txt
# or, manually:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ required.

**macOS only:** XGBoost needs OpenMP, which doesn't ship with the pip wheel. Install it once via Homebrew:

```bash
brew install libomp
```

Without this, `import xgboost` will fail with a `Library not loaded: @rpath/libomp.dylib` error.

### 2. Download the two Kaggle datasets

The pipeline expects these two files in `data/raw/`:

| File                         | Source                                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| `data/raw/song_lyrics.csv`   | https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information         |
| `data/raw/dataset.csv`       | https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset                          |

The Genius file is ~9 GB unzipped — the loader streams it in chunks, so don't try to
open it in pandas naively.

### 3. (Optional) Set up API credentials

Only needed if running `scripts/scrape_supplemental.py`. The main pipeline runs entirely
from the local CSVs.

```bash
cp .env.example .env
# then edit .env and fill in:
#   GENIUS_ACCESS_TOKEN=…   (free, takes 5 min: https://genius.com/api-clients)
#   SPOTIPY_CLIENT_ID=…
#   SPOTIPY_CLIENT_SECRET=…
```

---

## Quickstart

### End-to-end smoke test on a sample (~2 minutes)

```bash
make run-sample
# = python scripts/run_pipeline.py --sample 50000
```

This runs all 10 stages on a 50K-row sample of the Genius dataset. Good first sanity
check before kicking off the full ~5M-row run.

### Full build (~30–90 min depending on hardware)

```bash
make run
# = python scripts/run_pipeline.py
```

Output:

```
data/processed/
├── full.parquet           # Sam's input (unsupervised)
├── train.parquet          # Larry's training set
├── val.parquet            # Larry's validation set
├── test.parquet           # Larry's test set
└── metadata.json          # config snapshot, row counts, class distribution
```

### Validate the output

```bash
make validate
# = python scripts/validate_output.py
```

Runs: schema check, null check, class-distribution check, audio-feature range check
(`danceability ∈ [0,1]`, `tempo ∈ [40, 260]`, etc.), match-score distribution, and a
**leakage check** ensuring no `(artist, title)` pair appears in more than one split.
Exits non-zero on failure.

### Generate EDA artifacts

```bash
make eda
# writes to reports/figures/:
#   class_distribution.png
#   audio_features_by_genre.png
#   audio_feature_correlation.png
#   lyric_length_by_genre.png
#   match_score_hist.png
#   top_words_per_genre.csv
#   audio_feature_means_by_genre.csv
```

These figures and tables drop straight into the project write-up.

### Run tests

```bash
make test
# = pytest tests/ -v --cov=src
```

---

## CLI reference

```
python scripts/run_pipeline.py [OPTIONS]

  --sample N                Sample N rows from Genius before processing (debug)
  --output-dir PATH         Override data/processed/
  --no-intermediate         Skip writing intermediate parquets (faster, less debuggable)
  --match-threshold INT     rapidfuzz score cutoff for the join [default: 87]
  --samples-per-class INT   Cap per genre after balancing      [default: 4000]
  --min-words INT           Drop songs shorter than this       [default: 50]
  --max-words INT           Drop songs longer than this        [default: 1500]
  --test-size FLOAT         Test fraction                      [default: 0.15]
  --val-size FLOAT          Val fraction (of original)         [default: 0.15]
  --seed INT                Random seed                        [default: 42]
  --log-level [DEBUG|INFO|WARNING|ERROR]   [default: INFO]
```

---

## The 10 pipeline stages

`src/pipeline.py` is the orchestrator. Each stage is a method on the `Pipeline` class
and writes an intermediate parquet to `data/interim/` so you can resume mid-pipeline
during development.

| #  | Stage              | What it does                                                                                                |
| -- | ------------------ | ----------------------------------------------------------------------------------------------------------- |
| 1  | `load`             | Stream `song_lyrics.csv` and `dataset.csv` with explicit dtypes and optional sampling.                      |
| 2  | `filter_clean`     | English-only filter (langdetect, deterministic seed), then ftfy + regex cleaning of Genius cruft.           |
| 3  | `map_genres`       | Apply `SPOTIFY_GENRE_MAP` (substring match, longest-pattern-first) and `GENIUS_GENRE_MAP` (direct lookup).  |
| 4  | `match`            | Fuzzy join on `(normalized_artist, normalized_title)` via rapidfuzz; artist-blocking for tractability.      |
| 5  | `reconcile_genres` | Resolve genre conflicts between Spotify and Genius labels (Spotify wins; it's the more curated source).    |
| 6  | `engineer_features`| Compute 10 lyric features (vocab diversity, repetition ratio, etc.) and z-scale audio features.             |
| 7  | `balance`          | Cap each canonical genre at `samples_per_class` (default 4000) so Larry's models don't degenerate to "pop".  |
| 8  | `split`            | Dedupe on `(artist, title)` to prevent train/test leakage from studio+live versions, then stratified split.  |
| 9  | `persist`          | Write `full.parquet`, `train.parquet`, `val.parquet`, `test.parquet`.                                       |
| 10 | `metadata`         | Write `metadata.json` with full config snapshot, row counts per stage, and class distribution per split.   |

---

## Output schema

Each row in `full.parquet` / `train.parquet` / `val.parquet` / `test.parquet`:

| Column                     | Type    | Source       | Notes                                              |
| -------------------------- | ------- | ------------ | -------------------------------------------------- |
| `artist`                   | str     | Genius       | Original artist string                             |
| `title`                    | str     | Genius       | Original title string                              |
| `lyrics_clean`             | str     | derived      | After ftfy + regex pipeline                        |
| `genre`                    | category| derived      | One of: pop, hip-hop, rock, r&b, country, electronic |
| `language`                 | str     | Genius       | ISO code; filtered to "en"                         |
| `match_score`              | float   | rapidfuzz    | 0–100; ≥ 87 by default                             |
| **Audio features (raw)**   |         | Spotify CSV  | All native Spotify ranges                          |
| `danceability`             | float   |              | 0–1                                                |
| `energy`                   | float   |              | 0–1                                                |
| `loudness`                 | float   |              | dB, typically -60 to 5                             |
| `speechiness`              | float   |              | 0–1                                                |
| `acousticness`             | float   |              | 0–1                                                |
| `instrumentalness`         | float   |              | 0–1                                                |
| `liveness`                 | float   |              | 0–1                                                |
| `valence`                  | float   |              | 0–1                                                |
| `tempo`                    | float   |              | BPM                                                |
| **Audio features (z)**     |         | derived      | StandardScaler-fit on train, applied everywhere    |
| `danceability_z` … `tempo_z` | float |              | Use these for K-Means / KNN                        |
| **Lyric features**         |         | derived      |                                                    |
| `word_count`               | int     |              |                                                    |
| `unique_word_count`        | int     |              |                                                    |
| `vocab_diversity`          | float   |              | Type-token ratio                                   |
| `line_count`               | int     |              |                                                    |
| `avg_word_length`          | float   |              |                                                    |
| `avg_line_length`          | float   |              |                                                    |
| `exclaim_density`          | float   |              | `!` per word                                       |
| `question_density`         | float   |              | `?` per word                                       |
| `repetition_ratio`         | float   |              | Most-common-word frequency                         |
| `uppercase_ratio`          | float   |              |                                                    |

---

## Team handoff

### For Sam (unsupervised)

```python
import pandas as pd
df = pd.read_parquet("data/processed/full.parquet")

# Use the z-scaled audio features + lyric features for K-Means
feature_cols = [c for c in df.columns if c.endswith("_z")] + [
    "vocab_diversity", "repetition_ratio", "exclaim_density", "question_density",
]
X = df[feature_cols].values
# … fit KMeans, compute silhouette, etc.
```

The z-scaled columns are already on a common scale; mixing in lyric features works
because they're naturally bounded (ratios and densities). Genre is *not* used to fit
the clustering — but it's there as `df["genre"]` for evaluating cluster purity / NMI
afterward.

### For Larry (supervised)

```python
import pandas as pd
train = pd.read_parquet("data/processed/train.parquet")
val   = pd.read_parquet("data/processed/val.parquet")
test  = pd.read_parquet("data/processed/test.parquet")

# Lyrics-based: TF-IDF + LogReg / SBERT + LogReg
X_text = train["lyrics_clean"]
y      = train["genre"]

# Audio-based: XGBoost / KNN on the z-scaled features
audio_cols = [c for c in train.columns if c.endswith("_z")]
X_audio = train[audio_cols].values
```

All three splits share the same StandardScaler (fit on train), so audio features are
directly comparable across splits. The split is stratified on `genre` and deduped on
`(artist, title)` so studio + live versions of the same song can't leak across splits.

---

## Troubleshooting

**`FileNotFoundError: data/raw/song_lyrics.csv`** — Download the Kaggle datasets first
(see Setup §2). The CSV filenames must match exactly: `song_lyrics.csv` and `dataset.csv`.

**Pipeline OOMs on the full Genius dataset** — Use `--sample 500000` for a half-million
row run; that's the sweet spot for a 16 GB laptop. Or process in chunks via
`load_genius_chunked()` in `src/data_loader.py`.

**Match rate looks low** — Lower `--match-threshold` to 82 (default 87). Check
`data/interim/04_matched.parquet` and inspect rows with `match_score` near the cutoff
to tune.

**A canonical genre is missing from the output** — `validate_output.py` will flag this
loudly. Re-run `scripts/scrape_supplemental.py --genre <missing>` to fill the gap from
the Genius API, then re-run the pipeline.

**Spotify scraper errors with "Forbidden"** — That's the November 2024 deprecation
biting. The `SpotifyClient` in `api_scraper.py` is intentionally scoped to endpoints
that still work (`search`, `artist`); audio_features/recommendations/related_artists
are not exposed and will not be added.

**`langdetect` gives different results on different runs** — It's seeded in
`language_filter.py`, but only for that module's import. If you call `langdetect`
elsewhere, seed it yourself with `DetectorFactory.seed = 42`.

---

## Reproducibility

Every random source flows through `--seed` (default 42):
- `langdetect.DetectorFactory.seed`
- `train_test_split(random_state=…)`
- `numpy.random.default_rng(…)` for the balancer

`metadata.json` includes the full effective config and a `pipeline_version` string so
runs are bit-for-bit traceable.

---

## License & ethics

This is coursework, not redistribution. Per the research doc's pitfall section: **never
commit raw lyrics to a public repo.** `.gitignore` enforces this — `data/` is excluded
in its entirety. If you fork this for a public portfolio, ship the *pipeline* and the
*processed feature files* (embeddings, audio features, lyric statistics) but not the
raw lyrics text.
