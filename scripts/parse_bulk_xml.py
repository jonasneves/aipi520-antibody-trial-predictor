#!/usr/bin/env python3
"""
Parse bulk XML data and filter for Phase 2/3 antibody trials

This script parses the extracted AllPublicXML files and filters for:
- Phase 2 or Phase 3 trials
- Completed, terminated, withdrawn, or suspended status
- Interventional studies
- Contains antibody interventions
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from xml_parser import xmlfile2results
from data_collection import is_antibody_intervention


def is_phase2_or_3(phase_str):
    """Check if trial is Phase 2 or Phase 3"""
    if not phase_str:
        return False
    phase_lower = phase_str.lower()
    return 'phase 2' in phase_lower or 'phase 3' in phase_lower or 'phase2' in phase_lower or 'phase3' in phase_lower


def is_completed_status(status):
    """Check if trial has a completed/terminal status"""
    if not status:
        return False
    status_upper = status.upper()
    return status_upper in [
        'COMPLETED',
        'TERMINATED',
        'WITHDRAWN',
        'SUSPENDED',
        'APPROVED_FOR_MARKETING',
        'AVAILABLE',
        'NO_LONGER_AVAILABLE'
    ]


def has_antibody_intervention(intervention_names):
    """Check if any intervention is an antibody"""
    if not intervention_names:
        return False

    # intervention_names is a comma-separated string
    names = intervention_names.split(',')
    return any(is_antibody_intervention(name.strip()) for name in names if name.strip())


def main():
    print("=" * 80)
    print("Parsing Bulk XML for Phase 2/3 Antibody Trials")
    print("=" * 80)
    print()

    xml_dir = Path("data/raw_xml")
    if not xml_dir.exists():
        print(f"Error: XML directory not found: {xml_dir}")
        print("Please extract AllPublicXML.zip first")
        sys.exit(1)

    # Find all XML files
    xml_files = list(xml_dir.rglob("NCT*.xml"))
    print(f"Found {len(xml_files)} XML files")
    print()

    # Parse and filter trials
    filtered_trials = []
    total_processed = 0

    for i, xml_file in enumerate(xml_files):
        if i % 5000 == 0 and i > 0:
            print(f"Processed {i}/{len(xml_files)} files... ({len(filtered_trials)} matches so far)")

        try:
            data = xmlfile2results(str(xml_file))
            total_processed += 1

            # Apply filters
            if not is_phase2_or_3(data.get('phases', '')):
                continue

            if not is_completed_status(data.get('overall_status', '')):
                continue

            if data.get('study_type', '').upper() != 'INTERVENTIONAL':
                continue

            if not has_antibody_intervention(data.get('intervention_names', '')):
                continue

            # Add antibody detection
            intervention_names = data.get('intervention_names', '').split(',')
            antibody_names = [name.strip() for name in intervention_names if is_antibody_intervention(name.strip())]

            data['is_antibody'] = len(antibody_names) > 0
            data['antibody_name'] = antibody_names[0] if antibody_names else ''

            filtered_trials.append(data)

        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            continue

    print()
    print(f"✓ Processed {total_processed} files")
    print(f"✓ Found {len(filtered_trials)} Phase 2/3 antibody trials")
    print()

    # Save to JSON
    output_file = "data/completed_phase2_3_trials_raw.json"
    with open(output_file, 'w') as f:
        json.dump(filtered_trials, f, indent=2)

    print(f"✓ Saved filtered trials to {output_file}")

    # Print summary statistics
    print()
    print("Summary Statistics:")
    print(f"  - Total trials processed: {total_processed}")
    print(f"  - Antibody trials found: {len(filtered_trials)}")

    if filtered_trials:
        phases = {}
        statuses = {}
        for trial in filtered_trials:
            phase = trial.get('phases', 'Unknown')
            status = trial.get('overall_status', 'Unknown')
            phases[phase] = phases.get(phase, 0) + 1
            statuses[status] = statuses.get(status, 0) + 1

        print()
        print("Phase distribution:")
        for phase, count in sorted(phases.items(), key=lambda x: -x[1]):
            print(f"  - {phase}: {count}")

        print()
        print("Status distribution:")
        for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
            print(f"  - {status}: {count}")


if __name__ == "__main__":
    main()
