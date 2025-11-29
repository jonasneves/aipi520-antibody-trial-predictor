"""Trial status overview chart"""
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Import logic from individual charts to reuse
sys.path.insert(0, str(Path(__file__).parent))
from status_chart import create_status_chart

def create_combined_overview_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create trial status overview chart
    Shows the distribution of trial statuses (Completed, Terminated, Withdrawn, Suspended)
    """
    # Just return the status chart with updated title
    fig = create_status_chart(df)
    fig.update_layout(
        title_text="Trial Status Overview",
        height=500,
    )

    return fig
