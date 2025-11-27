"""
Antibody Classification Utilities

Utilities for detecting and classifying monoclonal antibody interventions
based on nomenclature patterns (INN naming conventions).
"""

import pandas as pd
from typing import Tuple


# Antibody Detection Constants
ANTIBODY_SUFFIXES = ['mab', 'zumab', 'umab', 'ximab', 'tumumab', 'omab']
ANTIBODY_KEYWORDS = ['antibody', 'monoclonal', 'immunoglobulin']

# Antibody type mapping by suffix
ANTIBODY_TYPE_MAP = {
    'omab': 'murine',       # Mouse antibody (rare, older generation)
    'ximab': 'chimeric',    # ~75% human, 25% mouse
    'zumab': 'humanized',   # ~90-95% human
    'umab': 'fully_human',  # 100% human (most modern)
    'tumumab': 'fully_human'  # Tumor-targeting human antibody
}


def analyze_antibody(intervention_name: str) -> Tuple[bool, str]:
    """
    Detect if an intervention is a monoclonal antibody and classify its type.

    Args:
        intervention_name: Name of the intervention/drug

    Returns:
        Tuple of (is_antibody: bool, antibody_type: str)

    Examples:
        >>> analyze_antibody("Pembrolizumab")
        (True, 'fully_human')
        >>> analyze_antibody("Keytruda")
        (False, 'not_antibody')
    """
    if pd.isna(intervention_name) or not isinstance(intervention_name, str):
        return False, 'not_antibody'

    name_lower = intervention_name.lower().strip()

    # Check if name ends with antibody suffix
    for suffix, ab_type in ANTIBODY_TYPE_MAP.items():
        if name_lower.endswith(suffix):
            return True, ab_type

    # Check if name contains antibody keywords
    if any(keyword in name_lower for keyword in ANTIBODY_KEYWORDS):
        return True, 'unknown'

    return False, 'not_antibody'


def is_antibody_intervention(intervention_name: str) -> bool:
    """Legacy function for backwards compatibility."""
    is_ab, _ = analyze_antibody(intervention_name)
    return is_ab


def get_antibody_type(intervention_name: str) -> str:
    """Legacy function for backwards compatibility."""
    _, ab_type = analyze_antibody(intervention_name)
    return ab_type
