"""Class balance visualization chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def create_class_balance_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create class balance visualization"""
    if 'binary_outcome' not in df.columns:
        return None

    outcome_counts = df['binary_outcome'].value_counts()
    labels = {0: 'Failure', 1: 'Success'}
    outcome_counts.index = outcome_counts.index.map(labels)

    fig = go.Figure(data=[
        go.Bar(
            x=outcome_counts.index,
            y=outcome_counts.values,
            marker_color=['#e74c3c', '#2ecc71'],
            text=outcome_counts.values,
            textposition='auto',
        )
    ])

    total = outcome_counts.sum()
    balance_text = "<br>".join([f"{idx}: {val:,} ({val/total*100:.1f}%)"
                                 for idx, val in outcome_counts.items()])

    fig.update_layout(
        title=f'Class Balance<br><sub>{balance_text}</sub>',
        xaxis_title='Outcome',
        yaxis_title='Number of Samples',
        height=400,
        template='plotly_white'
    )

    return fig
