from transformers import pipeline
from collections import Counter
import torch
import pandas as pd
from scipy.spatial.distance import cosine
import json

# Section-1 Create the personality models and find the scores for the characters

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Initialize Hugging Face pipelines with the appropriate device
sentiment_pipeline = pipeline("sentiment-analysis", device=device)
zero_shot_pipeline = pipeline("zero-shot-classification", device=device)

# Define new traits and their labels
trait_keywords = {
    "humor": ["joke", "funny", "laugh", "hilarious", "sarcasm"],
    "empathy": ["support", "help", "understand", "kindness", "sympathy"]
}

zero_shot_labels = {
    "quirkiness": ["eccentricity", "weirdness", "uniqueness"],
    "ambition": ["ambition", "goal-setting", "career", "dreams"],
    "romanticism": ["love", "romance", "relationships"]
}

# Trait extraction functions
def extract_humor(dialogues):
    """Extract humor based on positive sentiment and keywords."""
    scores = []
    for dialogue in dialogues:
        result = sentiment_pipeline(dialogue)
        if result[0]['label'] == "POSITIVE" and any(
            keyword in dialogue.lower() for keyword in trait_keywords["humor"]
        ):
            scores.append(result[0]['score'])
    return sum(scores) / len(scores) if scores else 0

def extract_empathy(dialogues):
    """Extract empathy based on keyword matches."""
    total_words = sum(len(dialogue.split()) for dialogue in dialogues)
    keyword_count = sum(
        sum(1 for word in dialogue.split() if word.lower() in trait_keywords["empathy"])
        for dialogue in dialogues
    )
    return keyword_count / total_words if total_words > 0 else 0

def extract_zero_shot_trait(dialogues, labels):
    """Extract traits using zero-shot classification."""
    scores = []
    for dialogue in dialogues:
        result = zero_shot_pipeline(dialogue, candidate_labels=labels)
        scores.append(max(result["scores"]))
    return sum(scores) / len(scores) if scores else 0

def calculate_character_traits(dialogues):
    """Calculate all traits for a character."""
    return {
        "humor": extract_humor(dialogues),
        "empathy": extract_empathy(dialogues),
        "quirkiness": extract_zero_shot_trait(dialogues, zero_shot_labels["quirkiness"]),
        "ambition": extract_zero_shot_trait(dialogues, zero_shot_labels["ambition"]),
        "romanticism": extract_zero_shot_trait(dialogues, zero_shot_labels["romanticism"]),
    }

# Normalization remains the same as the GoT implementation
def normalize_traits(character_profiles):
    """Normalize trait scores for consistency."""
    all_traits = {trait: [] for trait in next(iter(character_profiles.values()))}

    # Collect all values for each trait
    for traits in character_profiles.values():
        for trait, value in traits.items():
            all_traits[trait].append(value)

    # Normalize each trait
    normalized_profiles = {}
    for character, traits in character_profiles.items():
        normalized_profiles[character] = {
            trait: (value - min(all_traits[trait])) / (max(all_traits[trait]) - min(all_traits[trait]) + 1e-6)
            for trait, value in traits.items()
        }
    return normalized_profiles

'''
# Example Dataset
dataset = {
    "Chandler": [
        "Could I BE more sarcastic?",
        "I make jokes when I'm uncomfortable.",
        "I'm hopeless with women."
    ],
    "Rachel": [
        "It's like all my life, everyone has told me, 'You're a shoe!'",
        "I got off the plane.",
        "Oh, I forgot, you were the only person on this planet who knew what a hanger was."
    ]
}

# Calculate traits for each character
character_traits = {
    character: calculate_character_traits(dialogues)
    for character, dialogues in dataset.items()
}
'''

# dataset
file_path = 'dataset//friends_scripts//friends_data_cleaned.csv'
dataset = pd.read_csv(file_path)


# Run Trait Calculation actual data
character_personality_profiles = {
    record['Name']: calculate_character_traits(record['Sentence'])
    for index,record in dataset.iterrows()
}

# Normalize the scores
normalized_profiles = normalize_traits(character_personality_profiles)

# Output the results
print("Raw Trait Scores:")
print(character_personality_profiles)
print("\nNormalized Trait Scores:")
print(normalized_profiles)

# Save to a JSON file
with open("char_scores//friends_char_profiles.json", "w") as f:
    json.dump(normalized_profiles, f)