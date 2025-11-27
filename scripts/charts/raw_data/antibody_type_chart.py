"""Antibody type distribution chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def create_antibody_type_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create antibody type distribution chart"""
    if 'antibody_type' not in df.columns:
        return None

    ab_counts = df['antibody_type'].value_counts()

    # Calculate percentage for unknown category
    total = ab_counts.sum()
    unknown_pct = (ab_counts.get('unknown', 0) / total * 100) if 'unknown' in ab_counts.index else 0

    # Color bars differently for unknown vs known types
    colors = ['lightgray' if idx == 'unknown' else 'mediumpurple' for idx in ab_counts.index]

    fig = go.Figure(data=[
        go.Bar(
            x=ab_counts.index,
            y=ab_counts.values,
            marker_color=colors,
            text=ab_counts.values,
            textposition='auto',
        )
    ])

    annotations = []
    if unknown_pct > 0:
        annotations.append(
            dict(
                text=f'Note: "Unknown" category ({unknown_pct:.1f}%) indicates incomplete antibody classification data',
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=10, color='gray'),
                xanchor='center'
            )
        )

    fig.update_layout(
        title='Distribution of Antibody Types',
        xaxis_title='Antibody Type',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white',
        annotations=annotations
    )

    return fig
