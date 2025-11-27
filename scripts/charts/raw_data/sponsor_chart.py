"""Sponsor type distribution chart"""
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_simple_bar_chart


def create_sponsor_chart(df: pd.DataFrame) -> go.Figure:
    """Create sponsor class distribution chart"""
    return create_simple_bar_chart(
        df=df,
        column='sponsor_class',
        title='Distribution of Sponsor Types',
        xaxis_title='Number of Trials',
        yaxis_title='Sponsor Class',
        orientation='h',
        colorscale='Viridis',
        show_percentages=True
    )
