"""Trial phase distribution chart"""
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_simple_bar_chart


def create_phase_chart(df: pd.DataFrame) -> go.Figure:
    """Create phase distribution chart"""
    column = 'phase' if 'phase' in df.columns else 'phases'

    return create_simple_bar_chart(
        df=df,
        column=column,
        title='Distribution of Trial Phases',
        xaxis_title='Phase',
        yaxis_title='Number of Trials',
        orientation='v',
        colorscale='RdYlBu_r'
    )
