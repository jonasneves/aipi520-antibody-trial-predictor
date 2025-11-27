"""Feature importance chart using Random Forest"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional


def create_feature_importance_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create quick feature importance using Random Forest"""
    from sklearn.ensemble import RandomForestClassifier

    # Prepare data
    if 'binary_outcome' not in df.columns:
        return None

    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['binary_outcome', 'outcome_label']]

    if len(feature_cols) == 0:
        return None

    X = df[feature_cols].fillna(0)
    y = df['binary_outcome']

    # Quick Random Forest
    print("  Computing feature importance (this may take a minute)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get top 20 features
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    fig = go.Figure(data=[
        go.Bar(
            y=importances['feature'][::-1],  # Reverse for better readability
            x=importances['importance'][::-1],
            orientation='h',
            marker_color='steelblue',
            text=importances['importance'][::-1].round(4),
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Top 20 Feature Importances (Random Forest)',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        template='plotly_white'
    )

    return fig
