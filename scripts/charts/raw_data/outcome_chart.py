"""Trial outcome distribution chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def create_outcome_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create outcome distribution chart with class balance information"""
    if 'binary_outcome' in df.columns:
        outcome_col = 'binary_outcome'
        labels = {0: 'Failure', 1: 'Success'}
    elif 'outcome_label' in df.columns:
        outcome_col = 'outcome_label'
        labels = None
    else:
        return None

    outcome_counts = df[outcome_col].value_counts()
    if labels:
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

    # Calculate class balance
    total = outcome_counts.sum()
    balance_text = "<br>".join([f"{idx}: {val:,} ({val/total*100:.1f}%)"
                                 for idx, val in outcome_counts.items()])

    fig.update_layout(
        title=f'Trial Outcome Distribution<br><sub>{balance_text}</sub>',
        xaxis_title='Outcome',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white'
    )

    return fig
