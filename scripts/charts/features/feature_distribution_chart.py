"""Feature distribution charts"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def create_feature_distribution_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create distribution charts for key numeric features"""
    # Select a few important numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['binary_outcome', 'outcome_label']]

    # Pick top 6 by variance
    if len(feature_cols) > 6:
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(6).index.tolist()

    n_features = len(feature_cols)
    if n_features == 0:
        return None

    rows = (n_features + 2) // 3
    cols = min(3, n_features)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=feature_cols,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    for idx, col in enumerate(feature_cols):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1

        # Add histogram
        fig.add_trace(
            go.Histogram(x=df[col].dropna().tolist(), nbinsx=30, marker_color='lightblue', name=col, showlegend=False),
            row=row, col=col_pos
        )

        # Add median line
        median_val = df[col].median()
        fig.add_vline(
            x=median_val,
            line_dash="dash",
            line_color="red",
            opacity=0.7,
            annotation_text=f"Median: {median_val:.1f}",
            annotation_position="top",
            annotation_font_size=9,
            row=row, col=col_pos
        )

    fig.update_layout(
        title_text='Distribution of Key Features (with Median Lines)',
        height=rows * 300,
        template='plotly_white'
    )

    return fig
