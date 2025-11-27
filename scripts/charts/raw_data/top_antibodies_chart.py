"""Top antibody drugs chart"""
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

# Add src to path for antibody_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))


def create_top_antibodies_chart(df: pd.DataFrame, top_n: int = 15) -> Optional[go.Figure]:
    """
    Create top antibody drugs chart.

    Shows the most frequently tested antibody interventions with success rates.

    Args:
        df: DataFrame with trial data
        top_n: Number of top antibodies to show (default: 15)
    """
    if 'intervention_names' not in df.columns:
        return None

    # Extract individual antibody names from intervention_names
    # intervention_names can be semicolon-separated for combination therapies
    from antibody_utils import is_antibody_intervention

    antibody_list = []
    for idx, names_str in df['intervention_names'].items():
        if pd.isna(names_str):
            continue

        # Split by semicolon for combination therapies
        names = [n.strip() for n in str(names_str).split(';')]

        # Filter to only antibodies
        for name in names:
            if is_antibody_intervention(name):
                # Get outcome if available
                outcome = df.loc[idx, 'binary_outcome'] if 'binary_outcome' in df.columns else None
                antibody_list.append({
                    'antibody': name,
                    'outcome': outcome
                })

    if not antibody_list:
        return None

    ab_df = pd.DataFrame(antibody_list)

    # Count trials per antibody
    ab_counts = ab_df.groupby('antibody').agg({
        'outcome': ['count', 'sum', 'mean']
    }).reset_index()

    ab_counts.columns = ['antibody', 'total_trials', 'successes', 'success_rate']
    ab_counts['success_rate_pct'] = ab_counts['success_rate'] * 100

    # Sort by total trials and take top N
    ab_counts = ab_counts.sort_values('total_trials', ascending=False).head(top_n)

    # Sort by total trials for plotting (descending)
    ab_counts = ab_counts.sort_values('total_trials', ascending=True)

    # Create horizontal bar chart with color based on success rate
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=ab_counts['antibody'],
        x=ab_counts['total_trials'],
        orientation='h',
        marker=dict(
            color=ab_counts['success_rate_pct'],
            colorscale='RdYlGn',  # Red-Yellow-Green
            showscale=True,
            colorbar=dict(title="Success<br>Rate (%)")
        ),
        text=[f"{trials} trials ({rate:.0f}% success)"
              for trials, rate in zip(ab_counts['total_trials'], ab_counts['success_rate_pct'])],
        textposition='outside',
        textfont=dict(size=10),
    ))

    fig.update_layout(
        title=f'Top {top_n} Most Tested Antibody Drugs',
        xaxis_title='Number of Trials',
        yaxis_title='Antibody Drug',
        height=500,
        template='plotly_white',
        margin=dict(l=200),  # More space for antibody names
    )

    return fig
