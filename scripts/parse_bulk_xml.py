#!/usr/bin/env python3
"""
Parse bulk XML data and filter for Phase 2/3 antibody trials

This script:
1. Reads all XML files from data/xml_raw/
2. Parses each file and extracts trial information
3. Filters for Phase 2/3 antibody trials that are completed/terminated
4. Saves results in the same format expected by the pipeline
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xml_parser import (
    parse_clinical_trial_xml,
    filter_antibody_phase23_trial,
)


def find_all_xml_files(xml_dir: Path) -> List[Path]:
    """Find all XML files recursively in the extracted directory"""
    xml_files = []

    # The ClinicalTrials.gov bulk download has a nested structure like:
    # NCT0000xxxx/NCT00001234.xml
    for xml_file in xml_dir.rglob("*.xml"):
        xml_files.append(xml_file)

    return xml_files


def parse_and_filter_trials(xml_dir: Path, output_file: Path, progress_interval: int = 1000):
    """
    Parse all XML files and filter for antibody trials.

    Args:
        xml_dir: Directory containing extracted XML files
        output_file: Output JSON file path
        progress_interval: Print progress every N files
    """
    print(f"Finding XML files in {xml_dir}...")
    xml_files = find_all_xml_files(xml_dir)
    print(f"Found {len(xml_files):,} XML files")

    filtered_trials = []
    parse_errors = 0

    print("\nParsing XML files and filtering for Phase 2/3 antibody trials...")
    for i, xml_file in enumerate(xml_files, 1):
        if i % progress_interval == 0:
            print(f"  Processed {i:,}/{len(xml_files):,} files "
                  f"({i/len(xml_files)*100:.1f}%) - Found {len(filtered_trials):,} matching trials")

        trial_data = parse_clinical_trial_xml(xml_file)

        if trial_data is None:
            parse_errors += 1
            continue

        # Filter for antibody trials
        if filter_antibody_phase23_trial(trial_data):
            filtered_trials.append(trial_data)

    print(f"\n✓ Parsing complete!")
    print(f"  Total files processed: {len(xml_files):,}")
    print(f"  Parse errors: {parse_errors:,}")
    print(f"  Phase 2/3 antibody trials found: {len(filtered_trials):,}")

    # Save to JSON in raw format (same as API collection)
    print(f"\nSaving to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(filtered_trials, f, indent=2)

    print(f"✓ Saved {len(filtered_trials):,} trials to {output_file}")

    return filtered_trials


def convert_to_csv_format(trials: List[Dict], csv_file: Path):
    """
    Convert parsed trials to CSV format expected by data_collection.py

    This creates the same format as data_collection.save_studies_to_csv()
    """
    import pandas as pd
    from data_collection import ClinicalTrialsAPI

    print(f"\nConverting to CSV format for pipeline compatibility...")

    # Create API instance just for the extract_basic_features method
    api = ClinicalTrialsAPI()

    # The trials from XML are already in a flat format, but we need to
    # convert them to match the protocolSection format that extract_basic_features expects

    # Actually, let's just create the CSV directly since we already have flat data
    records = []
    for trial in trials:
        # Extract antibody type and other features
        intervention_names = trial.get("intervention_names", "")
        is_antibody, antibody_type = api._classify_antibody_type(intervention_names)

        record = {
            "nct_id": trial.get("nct_id", ""),
            "brief_title": trial.get("brief_title", ""),
            "official_title": trial.get("official_title", ""),
            "overall_status": trial.get("overall_status", ""),
            "why_stopped": trial.get("why_stopped", ""),
            "phase": trial.get("phase", ""),
            "study_type": trial.get("study_type", ""),
            "enrollment": trial.get("enrollment", 0),
            "conditions": trial.get("conditions", ""),
            "intervention_names": intervention_names,
            "is_antibody": is_antibody,
            "antibody_type": antibody_type,
            "sponsor_class": trial.get("sponsor_class", ""),
            "sponsor_name": trial.get("sponsor_name", ""),
            "start_date": trial.get("start_date", ""),
            "completion_date": trial.get("completion_date", ""),
            "primary_completion_date": trial.get("primary_completion_date", ""),
            "has_results": trial.get("has_results", False),
            "allocation": trial.get("allocation", ""),
            "intervention_model": trial.get("intervention_model", ""),
            "primary_purpose": trial.get("primary_purpose", ""),
            "masking": trial.get("masking", ""),
            "gender": trial.get("gender", ""),
            "minimum_age": trial.get("minimum_age", ""),
            "maximum_age": trial.get("maximum_age", ""),
            "num_locations": trial.get("num_locations", 0),
            "num_countries": trial.get("num_countries", 0),
            "countries": trial.get("countries", ""),
            "keywords": trial.get("keywords", ""),
        }
        records.append(record)

    df = pd.DataFrame(records)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved {len(df)} trials to {csv_file}")

    return df


def main():
    """Main entry point"""
    # Paths
    xml_dir = Path("data/xml_raw")
    raw_json_file = Path("data/completed_phase2_3_trials_raw.json")
    csv_file = Path("data/completed_phase2_3_trials.csv")

    if not xml_dir.exists():
        print(f"Error: XML directory not found at {xml_dir}")
        print("Please extract the bulk XML file first:")
        print("  unzip data/ctg-public-xml.zip -d data/xml_raw/")
        sys.exit(1)

    # Parse and filter
    trials = parse_and_filter_trials(xml_dir, raw_json_file)

    # Convert to CSV format
    convert_to_csv_format(trials, csv_file)

    print("\n" + "=" * 60)
    print("SUCCESS! Bulk XML parsing complete")
    print("=" * 60)
    print(f"Raw JSON: {raw_json_file}")
    print(f"CSV file: {csv_file}")
    print(f"Total Phase 2/3 antibody trials: {len(trials):,}")


if __name__ == "__main__":
    main()
