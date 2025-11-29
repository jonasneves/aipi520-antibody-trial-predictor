"""Trial status distribution chart"""
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_simple_bar_chart


def create_status_chart(df: pd.DataFrame) -> go.Figure:
    """Create trial status distribution chart with percentages"""
    counts = df['overall_status'].value_counts()
    total = counts.sum()

    # Create labels with percentages
    percentages = (counts / total * 100).round(1)
    text_labels = [f'{count:,} ({pct}%)' for count, pct in zip(counts.values, percentages)]

    # Reverse arrays so largest appears at top (plotly renders bottom-to-top)
    y_labels = list(reversed(counts.index))
    x_values = list(reversed(counts.values))
    text_labels = list(reversed(text_labels))

    # Theme colors: success (green) to failure (red/orange)
    status_colors = {
        'Completed': '#2ecc71',      # Green - success
        'Terminated': '#e74c3c',     # Red - failure
        'Withdrawn': '#C84E00',      # Orange - partial failure
        'Suspended': '#f39c12'       # Amber - on hold
    }
    bar_colors = [status_colors.get(label, '#95a5a6') for label in y_labels]

    fig = go.Figure(data=[
        go.Bar(
            y=y_labels,
            x=x_values,
            orientation='h',
            marker=dict(
                color=bar_colors
            ),
            text=text_labels,
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Distribution of Trial Status',
        xaxis_title='Number of Trials',
        yaxis_title='Status',
        height=400,
        template='plotly_white'
    )

    return fig
