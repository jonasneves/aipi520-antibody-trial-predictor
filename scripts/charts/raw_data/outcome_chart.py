"""Trial outcome distribution chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_outcome_bar_chart


def create_outcome_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create outcome distribution chart using a Donut Chart"""
    if 'binary_outcome' in df.columns:
        outcome_col = 'binary_outcome'
        # Explicitly map 0/1 to string labels for the chart
        df_mapped = df.copy()
        df_mapped['outcome_str'] = df_mapped['binary_outcome'].map({0: 'Failure', 1: 'Success'})
        counts = df_mapped['outcome_str'].value_counts()
    elif 'outcome_label' in df.columns:
        counts = df['outcome_label'].value_counts()
    else:
        return None

    # Colors: Green for Success, Red for Failure
    colors = {'Success': '#2ecc71', 'Failure': '#e74c3c'}
    chart_colors = [colors.get(x, '#bdc3c7') for x in counts.index]

    fig = go.Figure(data=[
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.5, # Makes it a Donut chart
            marker_colors=chart_colors,
            textinfo='label+percent+value',
            textfont_size=14
        )
    ])

    fig.update_layout(
        title='Trial Outcome Distribution (Class Balance)',
        annotations=[
            dict(
                text='<b>Success:</b> Completed/Approved<br><b>Failure:</b> Terminated/Withdrawn',
                x=0.5, y=0.5,
                font_size=12,
                showarrow=False,
                xanchor='center'
            )
        ],
        height=450,
        template='plotly_white'
    )

    return fig
