"""Trial status distribution chart"""
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_simple_bar_chart


def create_status_chart(df: pd.DataFrame) -> go.Figure:
    """Create trial status distribution chart"""
    return create_simple_bar_chart(
        df=df,
        column='overall_status',
        title='Distribution of Trial Status',
        xaxis_title='Number of Trials',
        yaxis_title='Status',
        orientation='h',
        colorscale='Viridis'
    )
