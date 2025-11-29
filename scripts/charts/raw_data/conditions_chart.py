"""Top conditions chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def create_conditions_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create top 20 conditions chart"""
    if 'conditions' not in df.columns:
        return None

    # Extract individual conditions
    all_conditions = []
    for conditions in df['conditions'].dropna():
        # Split by both comma and semicolon, then strip whitespace
        split_conditions = [c.strip() for c in conditions.replace(';', ',').split(',') if c.strip()]
        all_conditions.extend(split_conditions)

    if not all_conditions:
        return None

    condition_counts = pd.Series(all_conditions).value_counts().head(20)

    fig = go.Figure(data=[
        go.Bar(
            y=condition_counts.index[::-1],  # Reverse for better readability
            x=condition_counts.values[::-1],
            orientation='h',
            marker=dict(
                color=condition_counts.values[::-1],
                colorscale='Teal',
                showscale=False
            ),
            text=condition_counts.values[::-1],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title=f'Top 20 Most Common Conditions<br><sub>Total unique conditions: {len(pd.Series(all_conditions).unique()):,}</sub>',
        xaxis_title='Number of Trials',
        yaxis_title='Condition',
        height=600,
        template='plotly_white'
    )

    return fig
