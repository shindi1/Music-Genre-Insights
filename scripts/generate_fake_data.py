"""Generate synthetic data matching the pipeline output schema for development."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

GENRES = ["pop", "hip-hop", "rock", "r&b", "country", "electronic"]

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

FAKE_LYRICS = [
    "yeah yeah baby love you so much every day night",
    "streets hard grind never stop hustle money power",
    "electric lights city vibes dance floor all night long",
    "heart breaks tears fall down missing you again",
    "guitar strings southern roads pickup truck sunset drive",
    "bass drop beat heavy rhythm pulse feel the wave",
]

ARTISTS = [
    "Taylor Swift", "Drake", "The Beatles", "Beyoncé", "Johnny Cash",
    "Daft Punk", "Kendrick Lamar", "Adele", "Led Zeppelin", "Rihanna",
    "Garth Brooks", "The Weeknd", "Fleetwood Mac", "Jay-Z", "Coldplay",
    "Ariana Grande", "Eminem", "AC/DC", "Whitney Houston", "Alan Jackson",
]

TITLES = [
    "Love Song", "Night Moves", "Better Days", "Electric Feel", "Heartbreak Hotel",
    "Summer Nights", "Rolling On", "Deep Blue", "Golden Hour", "Fire and Rain",
    "Never Enough", "City Lights", "Broken Wings", "Rise Up", "Forever Young",
    "Sweet Melody", "Dark Horse", "Midnight Rain", "Born to Run", "Easy Rider",
]


def _genre_audio_params(genre: str) -> dict:
    """Return mean audio feature values per genre for realistic variation."""
    params = {
        "pop":        dict(dance=0.72, energy=0.68, loud=-5.5, speech=0.07, acoustic=0.18, instru=0.01, live=0.12, valence=0.60, tempo=118),
        "hip-hop":    dict(dance=0.78, energy=0.65, loud=-6.0, speech=0.22, acoustic=0.10, instru=0.02, live=0.10, valence=0.45, tempo=95),
        "rock":       dict(dance=0.52, energy=0.82, loud=-5.0, speech=0.06, acoustic=0.12, instru=0.05, live=0.18, valence=0.48, tempo=128),
        "r&b":        dict(dance=0.70, energy=0.60, loud=-7.0, speech=0.08, acoustic=0.22, instru=0.02, live=0.11, valence=0.52, tempo=105),
        "country":    dict(dance=0.60, energy=0.65, loud=-6.5, speech=0.05, acoustic=0.40, instru=0.01, live=0.13, valence=0.62, tempo=120),
        "electronic": dict(dance=0.75, energy=0.80, loud=-6.0, speech=0.06, acoustic=0.05, instru=0.35, live=0.15, valence=0.40, tempo=128),
    }
    return params[genre]


def generate_rows(n: int, rng: np.random.Generator, genre: str | None = None) -> pd.DataFrame:
    rows = []
    genres = [genre] * n if genre else rng.choice(GENRES, size=n).tolist()

    for i, g in enumerate(genres):
        p = _genre_audio_params(g)
        std = 0.08

        dance = float(np.clip(rng.normal(p["dance"], std), 0, 1))
        energy = float(np.clip(rng.normal(p["energy"], std), 0, 1))
        loud = float(np.clip(rng.normal(p["loud"], 3.0), -60, 0))
        speech = float(np.clip(rng.normal(p["speech"], 0.04), 0, 1))
        acoustic = float(np.clip(rng.normal(p["acoustic"], std), 0, 1))
        instru = float(np.clip(rng.normal(p["instru"], 0.05), 0, 1))
        live = float(np.clip(rng.normal(p["live"], 0.05), 0, 1))
        valence = float(np.clip(rng.normal(p["valence"], std), 0, 1))
        tempo = float(np.clip(rng.normal(p["tempo"], 15), 40, 260))

        wc = int(rng.integers(80, 500))
        uwc = int(rng.integers(40, wc))
        lc = int(rng.integers(10, 60))

        rows.append({
            "artist": rng.choice(ARTISTS),
            "title": f"{rng.choice(TITLES)} {i}",
            "genre": g,
            "language": "en",
            "match_score": float(rng.uniform(87, 100)),
            "danceability": dance,
            "energy": energy,
            "loudness": loud,
            "speechiness": speech,
            "acousticness": acoustic,
            "instrumentalness": instru,
            "liveness": live,
            "valence": valence,
            "tempo": tempo,
            "lyr_word_count": wc,
            "lyr_unique_word_count": uwc,
            "lyr_vocab_diversity": uwc / wc,
            "lyr_line_count": lc,
            "lyr_avg_word_length": float(rng.uniform(3.5, 5.5)),
            "lyr_avg_line_length": float(rng.uniform(4, 10)),
            "lyr_exclaim_density": float(rng.uniform(0, 0.05)),
            "lyr_question_density": float(rng.uniform(0, 0.03)),
            "lyr_repetition_ratio": float(rng.uniform(0.1, 0.6)),
            "lyr_uppercase_ratio": float(rng.uniform(0, 0.1)),
            "lyrics_clean": rng.choice(FAKE_LYRICS),
        })

    df = pd.DataFrame(rows)
    df["genre"] = pd.Categorical(df["genre"], categories=GENRES)

    # Z-scale audio features
    audio_cols = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    for col in audio_cols:
        mu, sigma = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    per_genre_train = 167  # ~1000 total
    per_genre_val = 28     # ~168 total
    per_genre_test = 28    # ~168 total

    splits = {}
    for split_name, per_genre in [("train", per_genre_train), ("val", per_genre_val), ("test", per_genre_test)]:
        parts = [generate_rows(per_genre, rng, genre=g) for g in GENRES]
        df = pd.concat(parts, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        splits[split_name] = df
        out = PROCESSED_DIR / f"{split_name}.parquet"
        df.to_parquet(out, index=False)
        print(f"Wrote {len(df)} rows → {out}")

    full = pd.concat(splits.values(), ignore_index=True)
    full_out = PROCESSED_DIR / "full.parquet"
    full.to_parquet(full_out, index=False)
    print(f"Wrote {len(full)} rows → {full_out}")

    metadata = {
        "note": "synthetic data — replace with real pipeline output",
        "rows": {k: len(v) for k, v in splits.items()},
        "class_distribution": splits["train"]["genre"].value_counts().to_dict(),
        "columns": list(full.columns.tolist()),
    }
    meta_out = PROCESSED_DIR / "metadata.json"
    meta_out.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {meta_out}")


if __name__ == "__main__":
    main()
