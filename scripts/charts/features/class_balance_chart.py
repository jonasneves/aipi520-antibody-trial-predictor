"""Class balance visualization chart"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_outcome_bar_chart


def create_class_balance_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create class balance visualization"""
    if 'binary_outcome' not in df.columns:
        return None

    labels = {0: 'Failure', 1: 'Success'}

    return create_outcome_bar_chart(
        df=df,
        outcome_column='binary_outcome',
        title='Class Balance',
        labels=labels,
        show_balance=True
    )
