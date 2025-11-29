"""Sponsor type distribution chart"""
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_simple_bar_chart


def create_sponsor_chart(df: pd.DataFrame) -> go.Figure:
    """Create sponsor class distribution chart"""
    
    # Calculate value counts and percentages
    sponsor_counts = df['sponsor_class'].value_counts()
    total_trials = sponsor_counts.sum()
    
    # Custom text for bars including count and percentage
    text_labels = [f'{count:,} ({count/total_trials:.1%})' for count in sponsor_counts.values]

    fig = create_simple_bar_chart(
        df=df,
        column='sponsor_class',
        title='Distribution of Sponsor Types<br><sup>(Note: "Other" often includes academic/government entities, "U.S. Fed" is a subset of NIH)</sup>',
        xaxis_title='Number of Trials',
        yaxis_title='Sponsor Class',
        orientation='h',
        colorscale='Viridis',
        show_percentages=False, # Disable internal percentage calculation
        text=text_labels        # Use custom text labels
    )

    return fig
