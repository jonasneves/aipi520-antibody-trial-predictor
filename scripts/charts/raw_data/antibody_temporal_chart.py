"""Antibody type temporal evolution chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def create_antibody_temporal_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create antibody type temporal evolution chart.

    Shows how antibody types have evolved over time (murine → chimeric → humanized → fully_human).
    """
    if 'antibody_type' not in df.columns or 'start_date' not in df.columns:
        return None

    # Parse dates and extract year
    df_copy = df.copy()
    df_copy['start_date_parsed'] = pd.to_datetime(df_copy['start_date'], errors='coerce')
    df_copy['start_year'] = df_copy['start_date_parsed'].dt.year

    # Filter out invalid years and incomplete future years
    df_copy = df_copy.dropna(subset=['start_year'])
    # Exclude years 2024+ as they contain incomplete data
    df_copy = df_copy[(df_copy['start_year'] >= 1990) & (df_copy['start_year'] <= 2023)]

    if len(df_copy) == 0:
        return None

    # Count trials by year and antibody type
    temporal_data = df_copy.groupby(['start_year', 'antibody_type']).size().reset_index(name='count')

    # Pivot for easier plotting
    pivot_data = temporal_data.pivot(index='start_year', columns='antibody_type', values='count').fillna(0)

    # Define antibody type order and colors
    antibody_order = ['murine', 'chimeric', 'humanized', 'fully_human', 'unknown']
    colors = {
        'murine': '#e74c3c',       # Red (oldest)
        'chimeric': '#f39c12',     # Orange
        'humanized': '#3498db',    # Blue
        'fully_human': '#2ecc71',  # Green (most modern)
        'unknown': '#95a5a6'       # Gray
    }

    fig = go.Figure()

    # Add traces for each antibody type (in order)
    for ab_type in antibody_order:
        if ab_type in pivot_data.columns:
            fig.add_trace(go.Scatter(
                name=ab_type.replace('_', ' ').title(),
                x=pivot_data.index.tolist(),
                y=pivot_data[ab_type].tolist(),
                mode='lines+markers',
                line=dict(width=2, color=colors.get(ab_type, '#95a5a6')),
                marker=dict(size=6),
                stackgroup='one',  # For stacked area chart
            ))

    fig.update_layout(
        title='Temporal Evolution of Antibody Types (1990-2023)',
        xaxis_title='Year',
        yaxis_title='Number of Trials',
        height=450,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                text='Note: Years 2024+ excluded due to incomplete data',
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=10, color='gray'),
                xanchor='center'
            )
        ]
    )

    return fig
