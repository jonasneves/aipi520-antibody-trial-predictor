"""Sponsor type distribution chart"""
import pandas as pd
import plotly.graph_objects as go


def create_sponsor_chart(df: pd.DataFrame) -> go.Figure:
    """Create sponsor class distribution chart"""
    sponsor_counts = df['sponsor_class'].value_counts()

    # Calculate percentages for labels
    total = sponsor_counts.sum()
    percentages = (sponsor_counts / total * 100).round(1)
    text_labels = [f'{count:,} ({pct}%)' for count, pct in zip(sponsor_counts.values, percentages)]

    fig = go.Figure(data=[
        go.Bar(
            y=sponsor_counts.index,
            x=sponsor_counts.values,
            orientation='h',
            text=text_labels,
            textposition='auto',
            marker=dict(
                color=sponsor_counts.values,
                colorscale='Viridis',
                showscale=False
            ),
        )
    ])

    fig.update_layout(
        title='Distribution of Sponsor Types',
        xaxis_title='Number of Trials',
        yaxis_title='Sponsor Class',
        height=400,
        template='plotly_white',
        showlegend=False
    )

    return fig
