"""Antibody type by therapeutic area chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def create_antibody_by_area_chart(df: pd.DataFrame, top_n: int = 10) -> Optional[go.Figure]:
    """
    Create antibody type by therapeutic area heatmap.

    Shows which antibody types are used for different therapeutic areas/conditions.

    Args:
        df: DataFrame with trial data
        top_n: Number of top therapeutic areas to show (default: 10)
    """
    if 'antibody_type' not in df.columns or 'conditions' not in df.columns:
        return None

    # Extract primary condition from conditions (semicolon-separated)
    df_copy = df.copy()
    df_copy['primary_condition'] = df_copy['conditions'].apply(
        lambda x: str(x).split(';')[0].strip() if pd.notna(x) else 'Unknown'
    )

    # Get top N conditions by trial count
    top_conditions = df_copy['primary_condition'].value_counts().head(top_n).index.tolist()

    # Filter to top conditions
    df_filtered = df_copy[df_copy['primary_condition'].isin(top_conditions)]

    if len(df_filtered) == 0:
        return None

    # Create cross-tabulation
    cross_tab = pd.crosstab(
        df_filtered['primary_condition'],
        df_filtered['antibody_type']
    )

    # Sort by total trials
    cross_tab['total'] = cross_tab.sum(axis=1)
    cross_tab = cross_tab.sort_values('total', ascending=False).drop(columns='total')

    # Define antibody type order
    antibody_order = ['murine', 'chimeric', 'humanized', 'fully_human', 'unknown']
    available_types = [ab for ab in antibody_order if ab in cross_tab.columns]

    # Reorder columns
    cross_tab = cross_tab[available_types]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cross_tab.values.tolist(),
        x=[col.replace('_', ' ').title() for col in cross_tab.columns],
        y=cross_tab.index.tolist(),
        colorscale='Blues',
        text=cross_tab.values.tolist(),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Trials"),
        hovertemplate=(
            '<b>Condition:</b> %{y}<br>' +
            '<b>Antibody Type:</b> %{x}<br>' +
            '<b>Trials:</b> %{z}<br>' +
            '<b>Percentage for this Condition:</b> %{customdata:.1f}%<extra></extra>'
        ),
        customdata=(cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100).values.tolist()
    ))

    fig.update_layout(
        title=f'Antibody Type Usage by Top {top_n} Therapeutic Areas',
        xaxis_title='Antibody Type',
        yaxis_title='Therapeutic Area (Primary Condition)',
        height=500,
        template='plotly_white',
        margin=dict(l=250),  # More space for condition names
    )

    return fig
