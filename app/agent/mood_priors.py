"""Mood priors for offline semantic scoring."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from core.logging import logger

MOOD_PRIORS_PATH = Path(__file__).parent / "mood_priors.json"


def load_priors() -> Dict[str, Dict[str, Any]]:
    """Load mood priors from JSON file.

    Returns:
        Dict mapping mood names to their priors data
    """
    if not MOOD_PRIORS_PATH.exists():
        logger.warning(f"Mood priors file not found at {MOOD_PRIORS_PATH}")
        return {}

    try:
        with open(MOOD_PRIORS_PATH, encoding='utf-8') as f:
            data = json.load(f)
            return data.get("moods", {})
    except Exception as e:
        logger.error(f"Failed to load mood priors: {e}")
        return {}


def get_relevant_priors(intent_text: str) -> Optional[Dict[str, Any]]:
    """Pick relevant mood priors based on intent text.

    Args:
        intent_text: User intent/prompt text

    Returns:
        Most relevant mood priors dict, or None if no match
    """
    priors = load_priors()
    if not priors:
        return None

    # Simple keyword mapping for moods
    mood_keywords = {
        "sad": ["sad", "heartbreak", "breakup", "depressed", "melancholy", "blue", "down", "cry", "tears"],
        "happy": ["happy", "joy", "cheerful", "upbeat", "positive", "smile", "fun", "party", "celebrate"],
        "chill": ["chill", "relax", "calm", "peaceful", "mellow", "easy", "laid back", "soothing", "ambient"],
        "energetic": ["energetic", "high energy", "pump up", "workout", "dance", "party", "excited", "intense"],
        "romantic": ["romantic", "love", "passion", "intimate", "date", "couple", "affection", "tender"],
        "angry": ["angry", "rage", "furious", "mad", "aggressive", "intense", "hardcore", "metal", "punk"]
    }

    intent_lower = intent_text.lower()

    # Count keyword matches for each mood
    mood_scores = {}
    for mood, keywords in mood_keywords.items():
        score = sum(1 for keyword in keywords if keyword in intent_lower)
        if score > 0:
            mood_scores[mood] = score

    if not mood_scores:
        # Default to chill if no clear match
        logger.info("No mood keywords found in intent, defaulting to chill")
        return priors.get("chill")

    # Return mood with highest score
    best_mood = max(mood_scores, key=mood_scores.get)
    logger.info(f"Selected mood '{best_mood}' for intent: {intent_text}")
    return priors.get(best_mood)


def get_mood_names() -> List[str]:
    """Get list of available mood names."""
    priors = load_priors()
    return list(priors.keys()) if priors else []


def build_priors_from_csv(csv_path: str) -> Dict[str, Any]:
    """Build mood priors from Kaggle songs dataset CSV.

    Args:
        csv_path: Path to the songs CSV file

    Returns:
        Dict with moods data structure
    """
    logger.info(f"Building mood priors from CSV: {csv_path}")

    # Mood definitions with feature thresholds
    mood_thresholds = {
        "sad": {"valence_max": 0.4, "energy_max": 0.5},
        "happy": {"valence_min": 0.6, "energy_min": 0.5},
        "chill": {"energy_max": 0.5, "tempo_max": 120},
        "energetic": {"energy_min": 0.7, "tempo_min": 120},
        "romantic": {"valence_min": 0.4, "valence_max": 0.8, "energy_max": 0.6},
        "angry": {"valence_max": 0.3, "energy_min": 0.8}
    }

    # Try to download NLTK resources if not available
    try:
        stop_words = set(stopwords.words('english'))
        # Test tokenizer
        word_tokenize("test sentence")
    except LookupError:
        logger.warning("NLTK resources not found, downloading...")
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        stop_words = set(stopwords.words('english'))

    # Read CSV in chunks to handle large files
    chunk_size = 10000
    all_chunks = []

    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            all_chunks.append(chunk)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not all_chunks:
        raise ValueError(f"No data found in CSV: {csv_path}")

    df = pd.concat(all_chunks, ignore_index=True)
    logger.info(f"Loaded {len(df)} songs from CSV")

    # Normalize column names (handle case variations)
    df.columns = df.columns.str.lower()

    # Required columns (be robust to missing ones)
    required_cols = ['valence', 'energy', 'tempo']

    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Clean and prepare data
    df = df.dropna(subset=required_cols)  # Remove rows with missing audio features
    df[required_cols] = df[required_cols].clip(0.0, 1.0)  # Ensure features are in valid range

    # Assign moods based on thresholds
    df['mood'] = None
    for mood, thresholds in mood_thresholds.items():
        mask = pd.Series([True] * len(df))
        for feature, threshold in thresholds.items():
            if feature.endswith('_min'):
                feature_name = feature[:-4]  # Remove '_min'
                mask &= (df[feature_name] >= threshold)
            elif feature.endswith('_max'):
                feature_name = feature[:-4]  # Remove '_max'
                mask &= (df[feature_name] <= threshold)

        df.loc[mask & df['mood'].isnull(), 'mood'] = mood

    # Handle unassigned songs (assign to closest mood)
    unassigned = df[df['mood'].isnull()].copy()
    if len(unassigned) > 0:
        logger.info(f"Assigning {len(unassigned)} unassigned songs to closest mood")
        for idx, row in unassigned.iterrows():
            # Calculate distance to each mood centroid
            distances = {}
            for mood, thresholds in mood_thresholds.items():
                distance = 0
                count = 0
                for feature, threshold in thresholds.items():
                    if feature.endswith('_min'):
                        feature_name = feature[:-4]
                        target_val = threshold + 0.1  # Slightly above min
                        distance += abs(row[feature_name] - target_val)
                        count += 1
                    elif feature.endswith('_max'):
                        feature_name = feature[:-4]
                        target_val = threshold - 0.1  # Slightly below max
                        distance += abs(row[feature_name] - target_val)
                        count += 1
                distances[mood] = distance / count if count > 0 else float('inf')

            closest_mood = min(distances, key=distances.get)
            df.at[idx, 'mood'] = closest_mood

    moods_data = {}

    for mood in mood_thresholds.keys():
        mood_df = df[df['mood'] == mood]
        if len(mood_df) == 0:
            logger.warning(f"No songs assigned to mood: {mood}")
            continue

        logger.info(f"Processing mood '{mood}': {len(mood_df)} songs")

        # Top genres
        top_genres = []
        if any(col in df.columns for col in ['genres', 'genre']):
            genre_col = 'genres' if 'genres' in df.columns else 'genre'
            all_genres = []

            for genres_str in mood_df[genre_col].dropna():
                if isinstance(genres_str, str):
                    # Handle various genre formats (comma-separated, pipe-separated, etc.)
                    genres = re.split(r'[|,;]\s*', genres_str)
                    all_genres.extend([g.strip().lower() for g in genres if g.strip()])

            if all_genres:
                from collections import Counter
                genre_counts = Counter(all_genres)
                top_genres = [genre for genre, _ in genre_counts.most_common(30)]

        # Top lyric keywords
        top_keywords = []
        if 'lyrics' in df.columns:
            all_words = []

            for lyrics in mood_df['lyrics'].dropna():
                if isinstance(lyrics, str):
                    # Basic text cleaning
                    lyrics = re.sub(r'[^\w\s]', '', lyrics.lower())
                    words = word_tokenize(lyrics)

                    # Filter stopwords and short words
                    filtered_words = [
                        word for word in words
                        if word not in stop_words and len(word) > 2
                    ]
                    all_words.extend(filtered_words)

            if all_words:
                from collections import Counter
                word_counts = Counter(all_words)
                top_keywords = [word for word, _ in word_counts.most_common(50)]

        # Feature ranges (percentiles)
        feature_ranges = {}
        for feature in required_cols:
            values = mood_df[feature].dropna()
            if len(values) > 0:
                p10 = float(values.quantile(0.1))
                p90 = float(values.quantile(0.9))
                feature_ranges[feature] = {"p10": round(p10, 3), "p90": round(p90, 3)}

        moods_data[mood] = {
            "top_genres": top_genres,
            "top_lyric_keywords": top_keywords,
            "feature_ranges": feature_ranges
        }

    return {"moods": moods_data}
