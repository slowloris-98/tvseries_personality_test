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

# Trait keywords for loyalty
trait_keywords = {
    "loyalty": ["family", "honor", "oath", "duty", "friendship", "trust"]
}

# Labels for zero-shot classification
manipulative_labels = ["deception", "manipulation", "power play", "trickery"]
ambitious_labels = ["ambition", "power", "goal-setting", "dreams"]

# Trait extraction functions
def extract_courage(dialogues):
    """Extract courage based on positive sentiment."""
    scores = []
    for dialogue in dialogues:
        result = sentiment_pipeline(dialogue)
        if result[0]['label'] == "POSITIVE":
            scores.append(result[0]['score'])
    return sum(scores) / len(scores) if scores else 0

def extract_loyalty(dialogues):
    """Extract loyalty based on keyword matches."""
    total_words = sum(len(dialogue.split()) for dialogue in dialogues)
    keyword_count = sum(
        sum(1 for word in dialogue.split() if word.lower() in trait_keywords["loyalty"])
        for dialogue in dialogues
    )
    return keyword_count / total_words if total_words > 0 else 0

def extract_trait(dialogues, labels):
    """Extract traits using zero-shot classification."""
    scores = []
    for dialogue in dialogues:
        result = zero_shot_pipeline(dialogue, candidate_labels=labels)
        scores.append(max(result["scores"]))
    return sum(scores) / len(scores) if scores else 0

def calculate_character_traits(dialogues):
    """Calculate all traits for a character."""
    return {
        "courage": extract_courage(dialogues),
        "loyalty": extract_loyalty(dialogues),
        "manipulativeness": extract_trait(dialogues, manipulative_labels),
        "ambition": extract_trait(dialogues, ambitious_labels),
    }

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
# Test Dataset
dataset = {
    "Jon Snow": [
        "The North remembers.",
        "I swore a vow to protect the people.",
        "I don’t want it. I never have."
    ],
    "Cersei Lannister": [
        "When you play the game of thrones, you win or you die.",
        "I choose violence.",
        "Power is power."
    ],
    "Arya Stark": [
        "A girl is Arya Stark of Winterfell. And I'm going home.",
        "I’m going to kill the Queen.",
        "The world doesn’t just let girls decide what they want to be."
    ]
}

# Run Trait Calculation test data
character_personality_profiles = {
    character: calculate_character_traits(dialogues)
    for character, dialogues in dataset.items()
}
'''

# dataset
file_path = 'dataset\got_all_scripts\data_cleaned.csv'
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
with open("character_profiles.json", "w") as f:
    json.dump(character_profiles, f)



# Section-2 Create a function to find the most similar character based on traits

def calculate_similarity(user, character_scores):
    user_vector = [user["positivity"], user["negativity"]]
    character_vector = [character_scores["positivity"], character_scores["negativity"]]
    similarity = 1 - cosine(user_vector, character_vector)  # 1 - cosine gives similarity
    return similarity


def match_user_to_character(user_personality, character_personality_scores):
    similarities = {}
    for character, scores in character_personality_scores.items():
        similarity = calculate_similarity(user_personality, scores)
        similarities[character] = similarity
    
    # Find the character with the highest similarity score
    best_match = max(similarities, key=similarities.get)
    return best_match, similarities[best_match]


best_match, similarity_score = match_user_to_character(user_personality, normalized_profiles)
print(f"Best Match: {best_match} (Similarity: {similarity_score:.2f})")

print(f"You are most like {best_match}!")
print(f"Similarity Score: {similarity_score:.2f}")
print("Why? You share these traits:")
for trait in user_personality:
    print(f"- {trait.capitalize()}: {user_personality[trait]} (User) vs. {normalized_profiles[best_match][trait]} (Character)")
