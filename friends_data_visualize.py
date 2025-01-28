import matplotlib.pyplot as plt
import numpy as np
import json
import plotly.express as px
import pandas as pd

# Load character profiles from JSON
with open("char_scores//friends_char_profiles_norm.json", "r") as f:
    character_profiles = json.load(f)

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