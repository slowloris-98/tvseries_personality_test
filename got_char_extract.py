from transformers import pipeline
from collections import Counter
import torch
import pandas as pd
from scipy.spatial.distance import cosine
import json
from transformers import BartTokenizer
import re

# Section-1 Create the personality models and find the scores for the characters

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Initialize Hugging Face pipelines with the appropriate device
zero_shot_pipeline = pipeline(model="facebook/bart-large-mnli", device=device)

zero_shot_labels = [
    "deception", "manipulation", "power play", "trickery", # manipulativeness
    "ambition", "goal-setting", "career", "dreams", # ambition
    "faithfulness", "dedication", "allegiance", "trustworthiness", "devotion", "commitment", # loyalty
    "bravery", "valor", "fearlessness", "heroism", "boldness", "daring" # courage
]

def chunk_text_by_sentence(text, max_length=900):
    """
    Splits the text into chunks where each chunk contains multiple sentences,
    but the chunk size does not exceed max_length. Sentences are split by commas.
    Args:
        text (str): The input text to chunk.
        max_length (int): The maximum length of each chunk.
    Returns:
        List[str]: List of text chunks, each with sentences ending at commas.
    """
    sentences = text.split(", ")  # Split text by commas (assuming commas separate sentences)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding the sentence would exceed the max_length
        if len(current_chunk) + len(sentence) + 2 <= max_length:  # +2 for the added comma and space
            if current_chunk:
                current_chunk += ", " + sentence
            else:
                current_chunk = sentence
        else:
            # If adding this sentence would exceed the max length, finalize the current chunk
            chunks.append(current_chunk)
            current_chunk = sentence  # Start a new chunk with the current sentence

    # Append the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def clean_text(input_text):
    pattern = r"[^a-zA-Z0-9,?' ]"
    cleaned_text = re.sub(pattern, '', input_text)
    return cleaned_text

def extract_zeroshot_trait(character, dialogues, labels):
    """Extract traits using zero-shot classification."""
    print(f"Processing character: {character}")

    # Initialize scores dictionary
    scores = {
        "manipulativeness": 0,
        "ambition": 0,
        "loyalty": 0,
        "courage": 0
    }

    all_dialogues = clean_text("".join(dialogues))
    
    # chunking as zero shotpipeine has a limit of 1024 tokens
    chunks = chunk_text_by_sentence(all_dialogues)
    for chunk in chunks:
        result = zero_shot_pipeline(chunk, candidate_labels=labels)
        
        # Aggregate scores for each trait across all chunks
        scores["manipulativeness"] += sum(result["scores"][0:5])
        scores["ambition"] += sum(result["scores"][5:9])
        scores["loyalty"] += sum(result["scores"][9:15])
        scores["courage"] += sum(result["scores"][15:21])
        print(f"character: {character} result: {result}")
    num_chunks = len(chunks)
    for trait in scores:
        scores[trait] /= num_chunks
        print(f"Final scores for {character}: {scores}")
    return scores


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
file_path = 'dataset\got_all_scripts\got_data_cleaned.csv'
dataset = pd.read_csv(file_path)


# Run Trait Calculation actual data
character_personality_profiles = {
    record['Name']: extract_zeroshot_trait(record['Name'], record['Sentence'], zero_shot_labels)
    for index,record in dataset.iterrows()
}

# Normalize the scores
normalized_profiles = normalize_traits(character_personality_profiles)

# Output the results
print("Raw Trait Scores:")
print(character_personality_profiles)
print("\nNormalized Trait Scores:")
print(normalized_profiles)

# Save normalized scores to a JSON file
with open("char_scores\got_char_profiles_norm.json", "w") as f:
    json.dump(normalized_profiles, f)

# Save raw scores to a JSON file
with open("char_scores\got_char_profiles_orig.json", "w") as f:
    json.dump(character_personality_profiles, f)