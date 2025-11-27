"""Enrollment distribution chart"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_enrollment_chart(df: pd.DataFrame) -> go.Figure:
    """Create enrollment distribution chart with linear scale, log scale, and box plot"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Enrollment Distribution', 'Log-Scale Enrollment', 'Box Plot Summary'),
        column_widths=[0.35, 0.35, 0.3]
    )

    enrollment_data = df['enrollment'].dropna()

    # Raw enrollment histogram
    fig.add_trace(
        go.Histogram(x=enrollment_data.tolist(), nbinsx=50, marker_color='lightblue', name='Enrollment'),
        row=1, col=1
    )

    # Log-scale enrollment histogram
    fig.add_trace(
        go.Histogram(x=np.log1p(enrollment_data).tolist(), nbinsx=50, marker_color='lightcoral', name='Log Enrollment'),
        row=1, col=2
    )

    # Box plot
    fig.add_trace(
        go.Box(
            y=enrollment_data.tolist(),
            name='Enrollment',
            marker_color='lightseagreen',
            boxmean='sd',  # Show mean and standard deviation
            showlegend=False
        ),
        row=1, col=3
    )

    # Update axes
    fig.update_xaxes(title_text="Enrollment Count", row=1, col=1)
    fig.update_xaxes(title_text="Log(Enrollment + 1)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Enrollment Count", row=1, col=3)

    # Calculate statistics for annotation
    median_val = enrollment_data.median()
    mean_val = enrollment_data.mean()
    q1 = enrollment_data.quantile(0.25)
    q3 = enrollment_data.quantile(0.75)

    fig.update_layout(
        title_text='Enrollment Analysis',
        height=400,
        showlegend=False,
        template='plotly_white',
        annotations=[
            dict(
                text=f'Median: {median_val:.0f} | Mean: {mean_val:.0f} | IQR: [{q1:.0f}, {q3:.0f}]',
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.12,
                showarrow=False,
                font=dict(size=11, color='dimgray'),
                xanchor='center'
            )
        ]
    )

    return fig
