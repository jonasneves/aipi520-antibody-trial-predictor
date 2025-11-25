#!/usr/bin/env python3
"""
Parse bulk XML data and filter for Phase 2/3 antibody trials

This script:
1. Streams XML files directly from ZIP archive (no full extraction needed)
2. Parses each file and extracts trial information
3. Filters for Phase 2/3 antibody trials that are completed/terminated
4. Saves results in the same format expected by the pipeline

This approach saves ~11GB of disk space by not extracting the full archive.
"""

import sys
import json
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict
from xml.etree import ElementTree as ET

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xml_parser import (
    parse_clinical_trial_xml,
    parse_clinical_trial_xml_from_element,
    filter_antibody_phase23_trial,
)


def parse_and_filter_trials_from_zip(zip_path: Path, output_file: Path, progress_interval: int = 1000):
    """
    Parse all XML files directly from ZIP archive and filter for antibody trials.

    This streams files from the ZIP without extracting the full archive,
    saving ~11GB of disk space.

    Args:
        zip_path: Path to the ZIP file containing XML files
        output_file: Output JSON file path
        progress_interval: Print progress every N files
    """
    print(f"Opening ZIP archive: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / (1024**3):.2f} GB")

    filtered_trials = []
    parse_errors = 0
    total_files = 0

    print("\nStreaming and parsing XML files from ZIP archive...")
    print("(This avoids extracting ~11GB to disk)")

    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        # Get list of XML files in the archive
        xml_entries = [name for name in zip_file.namelist() if name.endswith('.xml')]
        total_files = len(xml_entries)
        print(f"Found {total_files:,} XML files in archive")

        for i, xml_entry in enumerate(xml_entries, 1):
            if i % progress_interval == 0:
                print(f"  Processed {i:,}/{total_files:,} files "
                      f"({i/total_files*100:.1f}%) - Found {len(filtered_trials):,} matching trials")

            try:
                # Read XML content from ZIP (in memory, no disk extraction)
                with zip_file.open(xml_entry) as xml_file:
                    xml_content = xml_file.read()

                # Parse XML from string
                try:
                    root = ET.fromstring(xml_content)

                    # Extract trial data using a modified version of parse_clinical_trial_xml
                    trial_data = parse_clinical_trial_xml_from_element(root)

                    if trial_data is None:
                        parse_errors += 1
                        continue

                    # Filter for antibody trials
                    if filter_antibody_phase23_trial(trial_data):
                        filtered_trials.append(trial_data)

                except ET.ParseError as e:
                    parse_errors += 1
                    continue

            except Exception as e:
                parse_errors += 1
                continue

    print(f"\n✓ Parsing complete!")
    print(f"  Total files processed: {total_files:,}")
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
    from data_collection import analyze_antibody

    print(f"\nConverting to CSV format for pipeline compatibility...")

    # The trials from XML are already in a flat format, we just need to
    # classify antibody types and create the CSV

    records = []
    filtered_count = 0

    for trial in trials:
        # Extract antibody type and other features
        # intervention_names is semicolon-separated, analyze each one
        intervention_names = trial.get("intervention_names", "")

        # Split by semicolon and analyze each intervention separately
        is_antibody = False
        antibody_type = "not_antibody"

        if intervention_names:
            interventions = [i.strip() for i in intervention_names.split(";")]
            for intervention in interventions:
                ab_check, ab_type = analyze_antibody(intervention)
                if ab_check:
                    is_antibody = True
                    antibody_type = ab_type
                    break  # Found an antibody, use it

        # Filter: Only include trials where we found a true antibody intervention
        if not is_antibody:
            filtered_count += 1
            continue

        # Note: is_antibody is always True at this point, so we don't include it
        # This removes a zero-variance feature from the dataset
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
            "antibody_type": antibody_type,  # Keep this - has variance!
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

    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} trials without confirmed antibody interventions")
        print(f"  Remaining: {len(records)} trials with true antibody interventions")

    df = pd.DataFrame(records)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved {len(df)} antibody trials to {csv_file}")

    return df


def main():
    """Main entry point"""
    # Paths
    zip_file = Path("data/ctg-public-xml.zip")
    raw_json_file = Path("data/completed_phase2_3_trials_raw.json")
    csv_file = Path("data/completed_phase2_3_trials.csv")

    if not zip_file.exists():
        print(f"Error: ZIP file not found at {zip_file}")
        print("Please download the bulk XML file first from S3")
        sys.exit(1)

    # Parse and filter (streaming from ZIP, no extraction needed)
    trials = parse_and_filter_trials_from_zip(zip_file, raw_json_file)

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
