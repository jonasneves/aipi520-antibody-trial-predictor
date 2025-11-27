"""Correlation heatmap for numeric features"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional


def create_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create correlation heatmap for numeric features"""
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude target variable
    feature_cols = [col for col in numeric_cols if col not in ['binary_outcome', 'outcome_label']]

    if len(feature_cols) == 0:
        return None

    # Compute correlation matrix (limit to top 30 features to keep readable)
    if len(feature_cols) > 30:
        # Use variance to select top features
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(30).index.tolist()

    corr_matrix = df[feature_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values.tolist(),
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2).tolist(),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=f'Feature Correlation Heatmap (Top {len(feature_cols)} Features)',
        height=max(600, len(feature_cols) * 20),
        template='plotly_white'
    )

    return fig
