"""Phase vs Status cross-analysis heatmap"""
import pandas as pd
import plotly.graph_objects as go


def create_phase_status_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create phase vs status cross-analysis heatmap"""
    phase_col = 'phase' if 'phase' in df.columns else 'phases'

    # Create crosstab with percentages
    crosstab = pd.crosstab(df[phase_col], df['overall_status'], normalize='index') * 100

    # Round for display
    crosstab_display = crosstab.round(1)

    fig = go.Figure(data=go.Heatmap(
        z=crosstab.values.tolist(),  # Convert to list
        x=crosstab.columns.tolist(),  # Convert to list
        y=crosstab.index.tolist(),  # Convert to list
        colorscale='YlOrRd',
        text=crosstab_display.values.tolist(),  # Convert to list
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title='Percentage')
    ))

    fig.update_layout(
        title='Clinical Trial Status by Phase (%)',
        xaxis_title='Status',
        yaxis_title='Phase',
        height=500,
        template='plotly_white'
    )

    return fig
