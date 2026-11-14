"""Generate a small synthetic catalog fixture for tests (fake artists/titles)."""
import csv
import json
import random

random.seed(3)
GENRES = {
    "Folk": (["indie folk", "chamber folk"], (0.15, 0.45), (0.2, 0.5), (70, 110), 0.85),
    "Electronic": (["house", "deep house"], (0.7, 0.95), (0.5, 0.9), (118, 130), 0.05),
    "Rock": (["indie rock", "garage rock"], (0.6, 0.9), (0.3, 0.7), (110, 160), 0.1),
    "Hip-Hop": (["rap", "conscious hip hop"], (0.5, 0.8), (0.3, 0.7), (80, 100), 0.1),
    "Jazz": (["cool jazz", "bebop"], (0.2, 0.5), (0.3, 0.7), (80, 140), 0.7),
}
ARTISTS = {
    "Folk": ["Willow Creek", "Paper Lanterns"],
    "Electronic": ["Neon Atlas", "Circuit Bloom"],
    "Rock": ["The Static Hearts", "Gravel Kings"],
    "Hip-Hop": ["MC Parallax", "Lyric Theory"],
    "Jazz": ["Blue Hour Trio", "Velvet Quintet"],
}
WORDS = {
    "Folk": ["rain", "river", "autumn", "home", "quiet"],
    "Electronic": ["night", "pulse", "neon", "dance", "lights"],
    "Rock": ["fire", "road", "loud", "city", "run"],
    "Hip-Hop": ["hustle", "block", "dream", "crown", "flow"],
    "Jazz": ["smoke", "midnight", "velvet", "blue", "moon"],
}

rows = []
n = 0
for genre, (niche, e_rng, v_rng, t_rng, acoustic) in GENRES.items():
    for artist in ARTISTS[genre]:
        for k in range(6):
            n += 1
            word = WORDS[genre][k % 5]
            rows.append({
                "id": f"synth{n:04d}",
                "name": f"{word.title()} Song {k + 1}",
                "album_name": f"{artist} LP",
                "artists": json.dumps([artist]),
                "genre": genre,
                "niche_genres": json.dumps(niche),
                "year": 2000 + (n % 24),
                "popularity": random.randint(5, 90),
                "danceability": round(random.uniform(0.2, 0.9), 3),
                "energy": round(random.uniform(*e_rng), 3),
                "valence": round(random.uniform(*v_rng), 3),
                "tempo": round(random.uniform(*t_rng), 1),
                "acousticness": round(min(1, max(0, acoustic + random.uniform(-0.1, 0.1))), 3),
                "instrumentalness": round(random.uniform(0, 0.5), 3),
                "speechiness": round(random.uniform(0.03, 0.3), 3),
                "liveness": round(random.uniform(0.05, 0.3), 3),
                "loudness": round(random.uniform(-14, -4), 2),
                "lyrics_snippet": f"we sing of {word} and {WORDS[genre][(k + 2) % 5]} tonight",
            })

with open("tests/fixtures/catalog_synthetic.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "rows")
