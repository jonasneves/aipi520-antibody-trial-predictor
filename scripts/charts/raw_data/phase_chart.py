"""Trial phase distribution chart"""
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_simple_bar_chart


def create_phase_chart(df: pd.DataFrame) -> go.Figure:
    """Create phase distribution chart with logical sorting"""
    column = 'phase' if 'phase' in df.columns else 'phases'
    
    counts = df[column].value_counts()
    
    # Define logical order for phases
    phase_order = [
        'Early Phase 1', 
        'Phase 1', 
        'Phase 1/Phase 2', 
        'Phase 2', 
        'Phase 2/Phase 3', 
        'Phase 3', 
        'Phase 4'
    ]
    
    # Filter and sort based on the logical order
    sorted_phases = [p for p in phase_order if p in counts.index]
    # Add any remaining phases that weren't in our standard list at the end
    remaining = [p for p in counts.index if p not in sorted_phases]
    final_order = sorted_phases + remaining
    
    sorted_counts = counts.reindex(final_order).fillna(0)

    fig = go.Figure(data=[
        go.Bar(
            x=sorted_counts.index,
            y=sorted_counts.values,
            marker_color=sorted_counts.values,
            marker_colorscale='RdYlBu_r',
            text=sorted_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Trial Phases (Logical Order)',
        xaxis_title='Phase',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig
