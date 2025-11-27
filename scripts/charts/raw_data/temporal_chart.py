"""Temporal trends chart"""
import pandas as pd
import plotly.graph_objects as go


def create_temporal_chart(df: pd.DataFrame) -> go.Figure:
    """Create temporal trends chart showing trials started by year"""
    # Parse start dates
    df = df.copy()
    df['start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['start_year'] = df['start_date_parsed'].dt.year

    trials_by_year = df['start_year'].value_counts().sort_index()

    fig = go.Figure(data=[
        go.Scatter(
            x=trials_by_year.index.tolist(),  # Convert to list
            y=trials_by_year.values.tolist(),  # Convert to list
            mode='lines+markers',
            line=dict(width=2, color='steelblue'),
            marker=dict(size=8),
        )
    ])

    fig.update_layout(
        title='Trials Started by Year',
        xaxis_title='Year',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white'
    )

    return fig
