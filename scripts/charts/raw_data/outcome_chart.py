"""Trial outcome distribution chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_outcome_bar_chart


def create_outcome_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create outcome distribution chart with class balance information"""
    if 'binary_outcome' in df.columns:
        outcome_col = 'binary_outcome'
        labels = {0: 'Failure', 1: 'Success'}
    elif 'outcome_label' in df.columns:
        outcome_col = 'outcome_label'
        labels = None
    else:
        return None

    return create_outcome_bar_chart(
        df=df,
        outcome_column=outcome_col,
        title='Trial Outcome Distribution',
        labels=labels,
        show_balance=True
    )
