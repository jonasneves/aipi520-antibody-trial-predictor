"""Trial phase distribution chart"""
import pandas as pd
import plotly.graph_objects as go


def create_phase_chart(df: pd.DataFrame) -> go.Figure:
    """Create phase distribution chart"""
    phase_counts = df['phase'].value_counts() if 'phase' in df.columns else df['phases'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            x=phase_counts.index,
            y=phase_counts.values,
            marker=dict(
                color=phase_counts.values,
                colorscale='RdYlBu_r',
                showscale=False
            ),
            text=phase_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Trial Phases',
        xaxis_title='Phase',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white'
    )

    return fig
