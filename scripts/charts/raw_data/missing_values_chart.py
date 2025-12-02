"""Missing values analysis chart for raw data"""
import pandas as pd
import plotly.graph_objects as go


def create_missing_values_chart(df: pd.DataFrame) -> go.Figure:
    """Create missing values chart showing fields with missing data"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values found",
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

    # Calculate percentages
    missing_pct = (missing / len(df) * 100).round(1)

    # Take top 15 to avoid overcrowding
    missing = missing.head(15)
    missing_pct = missing_pct.head(15)

    fig = go.Figure(data=[
        go.Bar(
            y=missing.index[::-1],  # Reverse to show highest at top
            x=missing_pct.values[::-1],
            orientation='h',
            marker_color='coral',
            text=[f'{val}%' for val in missing_pct.values[::-1]],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Missing: %{x:.1f}%<br>Count: %{customdata:,}<extra></extra>',
            customdata=missing.values[::-1]
        )
    ])

    fig.update_layout(
        title=f'Missing Values by Field (Top {len(missing)})',
        xaxis_title='Percentage Missing (%)',
        yaxis_title='',
        height=max(400, len(missing) * 25),
        template='plotly_white',
        showlegend=False,
        margin=dict(l=150)
    )

    return fig
