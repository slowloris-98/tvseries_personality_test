import matplotlib.pyplot as plt
import numpy as np
import json
import plotly.express as px
import pandas as pd

# Load character profiles from JSON
with open("char_scores\got_char_profiles_norm.json", "r") as f:
    character_profiles = json.load(f)

'''
# Load character profiles from JSON
with open("char_scores\got_char_test_profiles.json", "r") as f:
    character_profiles = json.load(f)
'''

def plot_interactive_radar_chart(scores):
    """Plot an interactive radar chart with bold character names in hover tips."""
    traits = list(next(iter(scores.values())).keys())
    
    # Transform data for radar chart
    data = []
    for char, values in scores.items():
        row = {"Character": char}
        row.update(values)
        data.append(row)
    
    df = pd.DataFrame(data)
    fig = px.line_polar(
        df.melt(id_vars=["Character"], var_name="Trait", value_name="Score"),
        r="Score",
        theta="Trait",
        color="Character",
        line_close=True,
        hover_data=["Score"],
    )
    # Update hovertemplate to include bolded character name
    fig.update_traces(
        hovertemplate="<b>Character:</b> %{customdata[0]}<br><b>Trait:</b> %{theta}<br><b>Score:</b> %{r:.2f}"
    )
    fig.update_layout(
        title="Character Trait Radar Chart (Interactive)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    fig.show()

# Call the function
plot_interactive_radar_chart(character_profiles)

# Read data from CSV
df = pd.read_csv('dataset\got_all_scripts\got_data_cleaned.csv')

# Capitalize the first and last name of each character
df["Name"] = df["Name"].str.title()

# Create the pie chart
fig = px.pie(
    df, 
    names="Name", 
    values="Sentence_Count", 
    title="Number of Sentences by Each Character", 
    color_discrete_sequence=px.colors.qualitative.D3  # D3 has a better color spread
)

# Update traces to ensure proper styling
fig.update_traces(
    textinfo="percent+label",
    textfont=dict(
        family="Arial, sans-serif",
        size=16,
        color="black"  # Ensures the text is visible
    )
)

# Show the chart
fig.show()
