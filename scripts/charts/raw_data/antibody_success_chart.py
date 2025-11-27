"""Antibody type vs success rate chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def create_antibody_success_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create antibody type vs success rate chart.

    Shows success rate for each antibody type (murine, chimeric, humanized, fully_human).
    """
    if 'antibody_type' not in df.columns or 'binary_outcome' not in df.columns:
        return None

    # Calculate success rate by antibody type
    ab_stats = df.groupby('antibody_type').agg({
        'binary_outcome': ['sum', 'count', 'mean']
    }).reset_index()

    ab_stats.columns = ['antibody_type', 'successes', 'total', 'success_rate']
    ab_stats['success_rate_pct'] = ab_stats['success_rate'] * 100

    # Sort by success rate descending
    ab_stats = ab_stats.sort_values('success_rate_pct', ascending=False)

    # Create grouped bar chart
    fig = go.Figure()

    # Success rate bars
    fig.add_trace(go.Bar(
        name='Success Rate (%)',
        x=ab_stats['antibody_type'],
        y=ab_stats['success_rate_pct'],
        marker_color='#2ecc71',
        text=[f"{rate:.1f}%" for rate in ab_stats['success_rate_pct']],
        textposition='auto',
        yaxis='y',
    ))

    # Total trials bars (secondary axis)
    fig.add_trace(go.Bar(
        name='Total Trials',
        x=ab_stats['antibody_type'],
        y=ab_stats['total'],
        marker_color='#3498db',
        text=ab_stats['total'].values,
        textposition='auto',
        yaxis='y2',
        opacity=0.6,
    ))

    fig.update_layout(
        title='Antibody Type Success Rate Analysis',
        xaxis_title='Antibody Type',
        yaxis_title='Success Rate (%)',
        yaxis2=dict(
            title='Total Trials',
            overlaying='y',
            side='right'
        ),
        height=450,
        template='plotly_white',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
