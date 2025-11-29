"""Intervention type distribution chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def create_interventions_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create intervention type distribution chart"""
    if 'intervention_types' not in df.columns:
        return None

    # Extract individual intervention types
    all_interventions = []
    for interventions in df['intervention_types'].dropna():
        all_interventions.extend([i.strip() for i in str(interventions).split(',')])

    if not all_interventions:
        return None

    intervention_counts = pd.Series(all_interventions).value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            y=intervention_counts.index[::-1],  # Reverse for better readability
            x=intervention_counts.values[::-1],
            orientation='h',
            marker=dict(
                color=intervention_counts.values[::-1],
                colorscale='Plasma',
                showscale=False
            ),
            text=intervention_counts.values[::-1],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Intervention Types',
        xaxis_title='Number of Trials',
        yaxis_title='Intervention Type',
        height=500,
        template='plotly_white'
    )

    return fig
