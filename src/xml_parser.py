"""
XML Parser for ClinicalTrials.gov Bulk Data

This module parses XML files from the AllPublicXML.zip bulk download
and extracts detailed clinical trial information including clinical results,
adverse events, and outcome measures.
"""

from xml.etree import ElementTree as ET
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd


def parse_clinical_results(root: ET.Element) -> Dict:
    """
    Parse the clinical results section from XML and return structured data.

    This includes:
    - Participant flow (enrollment, completion)
    - Baseline characteristics
    - Outcome measures (primary and secondary)
    - Adverse events (serious and other)
    """
    clinical_results = {}

    # Find clinical_results section
    results_section = root.find('clinical_results')
    if results_section is None:
        return clinical_results

    # Parse participant flow
    participant_flow = {}
    flow_section = results_section.find('participant_flow')
    if flow_section is not None:
        # Parse groups
        groups = []
        group_list = flow_section.find('group_list')
        if group_list is not None:
            for group in group_list.findall('group'):
                group_data = {
                    'group_id': group.get('group_id', ''),
                    'title': group.find('title').text if group.find('title') is not None else '',
                    'description': group.find('description').text if group.find('description') is not None else ''
                }
                groups.append(group_data)

        participant_flow['groups'] = groups

    # Parse adverse events
    reported_events = {}
    events_section = results_section.find('reported_events')
    if events_section is not None:
        reported_events = {
            'time_frame': events_section.find('time_frame').text if events_section.find('time_frame') is not None else '',
            'serious_events': [],
            'other_events': []
        }

        # Parse serious events
        serious_events = events_section.find('serious_events')
        if serious_events is not None:
            category_list = serious_events.find('category_list')
            if category_list is not None:
                for category in category_list.findall('category'):
                    category_data = {
                        'title': category.find('title').text if category.find('title') is not None else '',
                        'event_count': 0
                    }

                    event_list = category.find('event_list')
                    if event_list is not None:
                        category_data['event_count'] = len(event_list.findall('event'))

                    reported_events['serious_events'].append(category_data)

    clinical_results['reported_events'] = reported_events

    return clinical_results


def xmlfile2results(xml_file: str) -> Dict:
    """
    Parse clinical trial XML file and return a dictionary with extracted data.

    Args:
        xml_file: Path to XML file

    Returns:
        Dictionary with extracted trial data
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Basic study identifiers
    nctid = root.find('id_info/nct_id').text if root.find('id_info/nct_id') is not None else ''
    url = root.find('required_header/url').text if root.find('required_header/url') is not None else ''

    # Titles
    brief_title = root.find('brief_title').text if root.find('brief_title') is not None else ''
    official_title = root.find('official_title').text if root.find('official_title') is not None else ''

    # Sponsors
    lead_sponsor = ''
    sponsor_class = ''
    sponsors = root.find('sponsors')
    if sponsors is not None:
        lead_sponsor_elem = sponsors.find('lead_sponsor/agency')
        lead_class_elem = sponsors.find('lead_sponsor/agency_class')
        if lead_sponsor_elem is not None:
            lead_sponsor = lead_sponsor_elem.text
        if lead_class_elem is not None:
            sponsor_class = lead_class_elem.text

    # Study type and phase
    study_type = root.find('study_type').text if root.find('study_type') is not None else ''
    phase = root.find('phase').text if root.find('phase') is not None else ''

    # Status and dates
    overall_status = root.find('overall_status').text if root.find('overall_status') is not None else ''
    why_stopped = root.find('why_stopped').text if root.find('why_stopped') is not None else ''

    # Dates
    start_date = ''
    start_date_elem = root.find('start_date')
    if start_date_elem is not None:
        start_date = start_date_elem.text if start_date_elem.text else ''

    completion_date = ''
    completion_date_elem = root.find('completion_date')
    if completion_date_elem is None:
        completion_date_elem = root.find('primary_completion_date')
    if completion_date_elem is not None:
        completion_date = completion_date_elem.text if completion_date_elem.text else ''

    # Interventions
    interventions = []
    intervention_names = []
    for intervention in root.findall('intervention'):
        intervention_type_elem = intervention.find('intervention_type')
        intervention_name_elem = intervention.find('intervention_name')

        if intervention_name_elem is not None and intervention_name_elem.text:
            intervention_names.append(intervention_name_elem.text)
            interventions.append({
                'type': intervention_type_elem.text if intervention_type_elem is not None else '',
                'name': intervention_name_elem.text
            })

    # Conditions
    conditions = [condition.text.strip() for condition in root.findall('condition') if condition.text]

    # Enrollment
    enrollment = 0
    enrollment_elem = root.find('enrollment')
    if enrollment_elem is not None and enrollment_elem.text:
        try:
            enrollment = int(enrollment_elem.text)
        except ValueError:
            enrollment = 0

    # Study design
    study_design_info = {}
    sdi = root.find('study_design_info')
    if sdi is not None:
        allocation = sdi.find('allocation')
        intervention_model = sdi.find('intervention_model')
        masking = sdi.find('masking')
        primary_purpose = sdi.find('primary_purpose')

        if allocation is not None and allocation.text:
            study_design_info['allocation'] = allocation.text
        if intervention_model is not None and intervention_model.text:
            study_design_info['intervention_model'] = intervention_model.text
        if masking is not None and masking.text:
            study_design_info['masking'] = masking.text
        if primary_purpose is not None and primary_purpose.text:
            study_design_info['primary_purpose'] = primary_purpose.text

    # Primary outcomes
    primary_outcomes = []
    for po in root.findall('primary_outcome'):
        measure_elem = po.find('measure')
        if measure_elem is not None and measure_elem.text:
            primary_outcomes.append({'measure': measure_elem.text})

    # Clinical Results
    clinical_results = parse_clinical_results(root)
    has_results = len(clinical_results) > 0

    # Assemble data
    data = {
        'nct_id': nctid,
        'url': url,
        'brief_title': brief_title,
        'official_title': official_title,
        'sponsor_name': lead_sponsor,
        'sponsor_class': sponsor_class,
        'study_type': study_type,
        'phases': phase,
        'overall_status': overall_status,
        'why_stopped': why_stopped,
        'start_date': start_date,
        'completion_date': completion_date,
        'intervention_names': ','.join(intervention_names),
        'intervention_types': ','.join(set(i['type'] for i in interventions if i.get('type'))),
        'conditions': ','.join(conditions),
        'enrollment': enrollment,
        'study_design_info': study_design_info,
        'primary_outcomes': primary_outcomes,
        'has_results': has_results,
        'clinical_results': clinical_results
    }

    return data


def parse_xml_directory(xml_dir: str, filter_func=None) -> pd.DataFrame:
    """
    Parse all XML files in a directory and return as DataFrame.

    Args:
        xml_dir: Directory containing XML files
        filter_func: Optional function to filter trials (returns True to include)

    Returns:
        DataFrame with parsed trial data
    """
    xml_path = Path(xml_dir)
    xml_files = list(xml_path.rglob("NCT*.xml"))

    print(f"Found {len(xml_files)} XML files")

    trials = []
    for i, xml_file in enumerate(xml_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(xml_files)} files...")

        try:
            data = xmlfile2results(str(xml_file))

            # Apply filter if provided
            if filter_func is None or filter_func(data):
                trials.append(data)

        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            continue

    print(f"✓ Parsed {len(trials)} trials")

    return pd.DataFrame(trials)
