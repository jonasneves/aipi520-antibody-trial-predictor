"""Trial status distribution chart"""
import pandas as pd
import plotly.graph_objects as go


def create_status_chart(df: pd.DataFrame) -> go.Figure:
    """Create trial status distribution chart"""
    status_counts = df['overall_status'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            y=status_counts.index,
            x=status_counts.values,
            orientation='h',
            marker=dict(
                color=status_counts.values,
                colorscale='Viridis',
                showscale=False
            ),
            text=status_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Trial Status',
        xaxis_title='Number of Trials',
        yaxis_title='Status',
        height=400,
        template='plotly_white'
    )

    return fig
