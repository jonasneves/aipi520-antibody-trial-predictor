"""Missing values analysis chart"""
import pandas as pd
import plotly.graph_objects as go


def create_missing_values_chart(df: pd.DataFrame) -> go.Figure:
    """Create missing values chart"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values found!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="green")
        )
        fig.update_layout(
            title='Missing Values Analysis',
            height=300,
            template='plotly_white'
        )
        return fig

    missing_pct = (missing / len(df) * 100).round(2)

    fig = go.Figure(data=[
        go.Bar(
            y=missing.index,
            x=missing_pct.values,
            orientation='h',
            marker_color='coral',
            text=[f'{val}%' for val in missing_pct.values],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Missing Values by Feature',
        xaxis_title='Percentage Missing',
        yaxis_title='Feature',
        height=max(300, len(missing) * 25),
        template='plotly_white'
    )

    return fig
