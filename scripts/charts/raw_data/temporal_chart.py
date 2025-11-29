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
    
    # Calculate missing stats for subtitle
    missing_count = df['start_year'].isna().sum()
    total_count = len(df)
    missing_pct = (missing_count / total_count) * 100

    fig = go.Figure(data=[
        go.Scatter(
            x=trials_by_year.index.tolist(),  # Convert to list
            y=trials_by_year.values.tolist(),  # Convert to list
            mode='lines+markers',
            line=dict(width=2, color='steelblue'),
            marker=dict(size=8),
            hovertemplate='<b>Year:</b> %{x}<br><b>Trials:</b> %{y:,}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=f'Trials Started by Year<br><sup>(Excludes {missing_count:,} trials ({missing_pct:.1f}%) with missing start dates)</sup>',
        xaxis_title='Year',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig
