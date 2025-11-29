"""Utility functions for creating charts with common patterns"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any
def create_simple_bar_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    orientation: str = 'h',
    colorscale: str = 'Viridis',
    show_percentages: bool = False,
    height: int = 400,
    template: str = 'plotly_white',
    custom_colors: Optional[list] = None,
    **kwargs
) -> go.Figure:
    """
    Create a simple bar chart from value counts

    Args:
        df: DataFrame containing the data
        column: Column name to create value counts from
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        orientation: 'h' for horizontal, 'v' for vertical
        colorscale: Plotly colorscale name
        show_percentages: Whether to show percentages in text labels
        height: Chart height in pixels
        template: Plotly template name
        custom_colors: List of colors to use instead of colorscale
        **kwargs: Additional arguments passed to go.Bar(). Can include 'text' to override defaults.

    Returns:
        Plotly Figure object
    """
    counts = df[column].value_counts()

    # Generate default text labels if 'text' is not provided in kwargs
    if 'text' not in kwargs:
        if show_percentages:
            total = counts.sum()
            percentages = (counts / total * 100).round(1)
            kwargs['text'] = [f'{count:,} ({pct}%)' for count, pct in zip(counts.values, percentages)]
        else:
            kwargs['text'] = counts.values.astype(str) # Ensure text is string for display

    # Prepare marker configuration
    marker_config = {}
    if custom_colors:
        marker_config['color'] = custom_colors
    else:
        # Use theme-friendly gradient instead of Viridis
        # Duke colors: blues and teals
        theme_colorscale = [
            [0.0, '#005587'],   # Primary blue
            [0.5, '#339898'],   # Primary teal
            [1.0, '#1D6363']    # Primary cyan
        ]
        marker_config = {
            'color': counts.values,
            'colorscale': theme_colorscale,
            'showscale': False
        }

    # For horizontal bars, reverse the data so largest appears at top
    if orientation == 'h':
        y_data = list(reversed(counts.index))
        x_data = list(reversed(counts.values))
        # Also reverse text if it was provided
        if 'text' in kwargs:
            kwargs['text'] = list(reversed(kwargs['text']))
        # Update marker colors for reversed data
        if 'color' in marker_config and not isinstance(marker_config['color'], str):
            marker_config['color'] = list(reversed(marker_config['color']))

        bar_data = go.Bar(
            y=y_data,
            x=x_data,
            orientation='h',
            marker=marker_config,
            textposition='auto',
            **kwargs
        )
    else:
        bar_data = go.Bar(
            x=counts.index,
            y=counts.values,
            marker=marker_config,
            textposition='auto',
            **kwargs
        )

    fig = go.Figure(data=[bar_data])

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        template=template,
        showlegend=False
    )

    return fig


def create_outcome_bar_chart(
    df: pd.DataFrame,
    outcome_column: str,
    title: str,
    labels: Optional[Dict[Any, str]] = None,
    colors: Optional[list] = None,
    show_balance: bool = True,
    height: int = 400,
) -> go.Figure:
    """
    Create a bar chart for outcome/class balance visualization

    Args:
        df: DataFrame containing the data
        outcome_column: Column name for outcomes
        title: Base title for the chart
        labels: Optional mapping of outcome values to display labels
        colors: Custom colors for bars (default: red/green for binary)
        show_balance: Whether to show percentage balance in title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    outcome_counts = df[outcome_column].value_counts()

    # Apply labels if provided
    if labels:
        outcome_counts.index = outcome_counts.index.map(labels)

    # Use default red/green colors for binary outcomes if not specified
    if colors is None:
        colors = ['#e74c3c', '#2ecc71']

    fig = go.Figure(data=[
        go.Bar(
            x=outcome_counts.index,
            y=outcome_counts.values,
            marker_color=colors,
            text=outcome_counts.values,
            textposition='auto',
        )
    ])

    # Add balance information to title if requested
    if show_balance:
        total = outcome_counts.sum()
        balance_text = "<br>".join([f"{idx}: {val:,} ({val/total*100:.1f}%)"
                                     for idx, val in outcome_counts.items()])
        full_title = f'{title}<br><sub>{balance_text}</sub>'
    else:
        full_title = title

    fig.update_layout(
        title=full_title,
        xaxis_title='Outcome',
        yaxis_title='Number of Trials' if 'Trial' in title else 'Number of Samples',
        height=height,
        template='plotly_white'
    )

    return fig
