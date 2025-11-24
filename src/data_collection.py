"""
Clinical Trials Data Collection Module

This module provides functions to download and process clinical trial data
from ClinicalTrials.gov API, with specific focus on monoclonal antibody trials.
"""

import requests
import pandas as pd
import time
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path


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


# Legacy compatibility functions
def is_antibody_intervention(intervention_name: str) -> bool:
    """Legacy function for backwards compatibility."""
    is_ab, _ = analyze_antibody(intervention_name)
    return is_ab


def get_antibody_type(intervention_name: str) -> str:
    """Legacy function for backwards compatibility."""
    _, ab_type = analyze_antibody(intervention_name)
    return ab_type


class ClinicalTrialsAPI:
    """
    Interface to ClinicalTrials.gov API v2
    API Documentation: https://clinicaltrials.gov/data-api/api
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self, output_dir: str = "../data"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def search_studies(self,
                      query_params: Optional[Dict] = None,
                      max_studies: int = 1000,
                      page_size: int = 100) -> List[Dict]:
        """
        Search for clinical trials studies

        Args:
            query_params: Dictionary of query parameters
            max_studies: Maximum number of studies to retrieve
            page_size: Number of studies per page (max 100)

        Returns:
            List of study dictionaries
        """
        endpoint = f"{self.BASE_URL}/studies"

        # Set up query parameters
        query_params = query_params or {}
        query_params["format"] = "json"
        query_params["pageSize"] = min(page_size, 100)

        all_studies = []
        page_token = None

        print(f"Fetching studies from ClinicalTrials.gov...")

        while len(all_studies) < max_studies:
            if page_token:
                query_params["pageToken"] = page_token

            try:
                response = requests.get(endpoint, params=query_params, timeout=30)
                response.raise_for_status()
                data = response.json()
                studies = data.get("studies", [])

                if not studies:
                    print("No more studies found.")
                    break

                all_studies.extend(studies)
                print(f"Retrieved {len(all_studies)} studies so far...")

                page_token = data.get("nextPageToken")
                if not page_token:
                    print("Reached last page.")
                    break

                time.sleep(0.5)  # Rate limiting

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break

        return all_studies[:max_studies]

    def get_study_details(self, nct_id: str) -> Optional[Dict]:
        """
        Get detailed information for a specific study by NCT ID

        Args:
            nct_id: The NCT identifier for the study

        Returns:
            Study details dictionary or None if not found
        """
        endpoint = f"{self.BASE_URL}/studies/{nct_id}"

        try:
            response = requests.get(endpoint, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching study {nct_id}: {e}")
            return None

    def filter_by_phase(self, studies: List[Dict], phases: List[str]) -> List[Dict]:
        """Filter studies by clinical trial phase"""
        return [
            study for study in studies
            if any(phase in study.get('protocolSection', {}).get('designModule', {}).get('phases', [])
                   for phase in phases)
        ]

    def extract_basic_features(self, study: Dict) -> Dict:
        """
        Extract basic features from a study for initial analysis

        Args:
            study: Study dictionary from API

        Returns:
            Dictionary of extracted features
        """
        protocol = study.get('protocolSection', {})

        # Helper to safely get nested values
        def get_nested(d, *keys, default=''):
            for key in keys:
                d = d.get(key, {})
                if not isinstance(d, dict):
                    return d if d else default
            return default or ''

        # Identification
        id_module = protocol.get('identificationModule', {})
        nct_id = id_module.get('nctId', '')
        brief_title = id_module.get('briefTitle', '')

        # Status
        status_module = protocol.get('statusModule', {})
        overall_status = status_module.get('overallStatus', '')
        why_stopped = status_module.get('whyStopped', '')
        start_date = get_nested(status_module, 'startDateStruct', 'date')
        completion_date = get_nested(status_module, 'completionDateStruct', 'date')

        # Design
        design_module = protocol.get('designModule', {})
        study_type = design_module.get('studyType', '')
        phases = design_module.get('phases', [])
        enrollment = get_nested(design_module, 'enrollmentInfo', 'count') or 0

        # Conditions
        conditions = protocol.get('conditionsModule', {}).get('conditions', [])

        # Interventions
        arms_module = protocol.get('armsInterventionsModule', {})
        interventions = arms_module.get('interventions', [])
        intervention_types = [inv.get('type', '') for inv in interventions]
        intervention_names = [inv.get('name', '') for inv in interventions]

        # Antibody detection
        antibody_names = [name for name in intervention_names if is_antibody_intervention(name)]
        primary_antibody = antibody_names[0] if antibody_names else ''
        is_antibody, antibody_type = analyze_antibody(primary_antibody) if primary_antibody else (False, 'not_antibody')

        # Sponsor
        sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
        lead_sponsor = sponsor_module.get('leadSponsor', {})
        sponsor_name = lead_sponsor.get('name', '')
        sponsor_class = lead_sponsor.get('class', '')

        # Results
        has_results = 'resultsSection' in study

        return {
            'nct_id': nct_id,
            'brief_title': brief_title,
            'overall_status': overall_status,
            'why_stopped': why_stopped,
            'start_date': start_date,
            'completion_date': completion_date,
            'study_type': study_type,
            'phases': ','.join(phases) if phases else '',
            'enrollment': enrollment,
            'conditions': ','.join(conditions) if conditions else '',
            'intervention_types': ','.join(set(intervention_types)) if intervention_types else '',
            'intervention_names': ','.join(intervention_names) if intervention_names else '',
            'is_antibody': is_antibody,
            'antibody_name': primary_antibody,
            'antibody_type': antibody_type,
            'sponsor_name': sponsor_name,
            'sponsor_class': sponsor_class,
            'has_results': has_results
        }

    def save_studies_to_csv(self, studies: List[Dict], filename: str = "clinical_trials.csv") -> pd.DataFrame:
        """Extract features from studies and save to CSV"""
        output_path = Path(self.output_dir) / filename

        print(f"Extracting features from {len(studies)} studies...")
        features = [self.extract_basic_features(study) for study in studies]
        df = pd.DataFrame(features)

        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} studies to {output_path}")

        return df

    def save_raw_json(self, studies: List[Dict], filename: str = "clinical_trials_raw.json"):
        """Save raw study data as JSON for later processing"""
        output_path = Path(self.output_dir) / filename

        with open(output_path, 'w') as f:
            json.dump(studies, f, indent=2)

        print(f"Saved raw data for {len(studies)} studies to {output_path}")


def main():
    """Main function to demonstrate data collection for antibody trials"""
    api = ClinicalTrialsAPI(output_dir="../data")

    print("\n=== Collecting Monoclonal Antibody Phase 2/3 Trials ===")
    query_params = {
        "query.term": (
            "(AREA[Phase]PHASE2 OR AREA[Phase]PHASE3) AND "
            "AREA[StudyType]INTERVENTIONAL AND "
            "(AREA[InterventionName]*mab OR AREA[InterventionName]*zumab OR "
            "AREA[InterventionName]*umab OR AREA[InterventionName]*ximab OR "
            "AREA[InterventionName]antibody OR AREA[InterventionName]monoclonal) AND "
            "AREA[OverallStatus]COMPLETED"
        )
    }

    studies = api.search_studies(query_params=query_params, max_studies=5000)

    if studies:
        print(f"Found {len(studies)} antibody trials")

        df = api.save_studies_to_csv(studies, "antibody_trials.csv")
        api.save_raw_json(studies, "antibody_trials_raw.json")

        print("\n=== Antibody Trial Summary Statistics ===")
        print(f"Total antibody trials: {len(df)}")
        print(f"\nAntibody type distribution:")
        print(df['antibody_type'].value_counts())
        print(f"\nPhase distribution:")
        print(df['phases'].value_counts())
        print(f"\nSponsor class distribution:")
        print(df['sponsor_class'].value_counts())
        print(f"\nTop 10 antibodies:")
        print(df['antibody_name'].value_counts().head(10))


if __name__ == "__main__":
    main()
