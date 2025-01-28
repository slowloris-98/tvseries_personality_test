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
#sentiment_pipeline = pipeline("sentiment-analysis", device=device)
zero_shot_pipeline = pipeline(model="facebook/bart-large-mnli", device=device)

zero_shot_labels = [
    "eccentricity", "weirdness", "uniqueness", # quirkiness
    "ambition", "goal-setting", "career", "dreams", "success", "motivation", "goals", # ambition
    "love", "romance", "relationships", # romanticism
    "meticulousness", "perfectionism", "orderliness", "attention to detail", "cleanliness", "structure" # perfectionism
    "sarcasm", "irony", "mocking", "teasing", "sarcastic humor", # sarcasm
    "competitiveness", "winning", "challenge", "rivalry", "competitive" # competitiveness
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
        "quirkiness": 0,
        "ambition": 0,
        "romanticism": 0,
        "perfectionism": 0,
        "sarcasm": 0,
        "competitiveness": 0
    }

    all_dialogues = clean_text("".join(dialogues))
    
    # chunking as zero shotpipeine has a limit of 1024 tokens
    chunks = chunk_text_by_sentence(all_dialogues)
    for chunk in chunks:
        result = zero_shot_pipeline(chunk, candidate_labels=labels)
        
        # Aggregate scores for each trait across all chunks
        scores["quirkiness"] += sum(result["scores"][0:3])
        scores["ambition"] += sum(result["scores"][3:9])
        scores["romanticism"] += sum(result["scores"][9:12])
        scores["perfectionism"] += sum(result["scores"][12:18])
        scores["sarcasm"] += sum(result["scores"][18:24])
        scores["competitiveness"] += sum(result["scores"][24:30])
        print(f"character: {character} result: {result}")
    num_chunks = len(chunks)
    for trait in scores:
        scores[trait] /= num_chunks
        print(f"Final scores for {character}: {scores}")
    return scores

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
file_path = 'dataset//friends_scripts//friends_data_cleaned_core.csv'
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

# Save to a JSON file
with open("char_scores//friends_char_profiles_norm.json", "w") as f:
    json.dump(normalized_profiles, f)

# Save to a JSON file
with open("char_scores//friends_char_profiles_orig.json", "w") as f:
    json.dump(character_personality_profiles, f)