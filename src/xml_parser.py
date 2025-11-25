"""
XML Parser for ClinicalTrials.gov Bulk Data

This module parses XML files from the bulk download (ctg-public-xml.zip)
and extracts detailed clinical trial information.
"""

from xml.etree import ElementTree as ET
from typing import Dict, List, Optional
from pathlib import Path
import json


def safe_find_text(element: Optional[ET.Element], path: str, default: str = "") -> str:
    """Safely find text in XML element"""
    if element is None:
        return default
    found = element.find(path)
    return found.text if found is not None and found.text else default


def safe_findall(element: Optional[ET.Element], path: str) -> List[ET.Element]:
    """Safely find all matching elements"""
    if element is None:
        return []
    return element.findall(path) or []


def parse_clinical_trial_xml(xml_file: Path) -> Dict:
    """
    Parse a single ClinicalTrials.gov XML file and extract key information.

    Args:
        xml_file: Path to the XML file

    Returns:
        Dictionary with extracted trial information
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Basic identification
        nct_id = safe_find_text(root, "id_info/nct_id")

        # Study information
        brief_title = safe_find_text(root, "brief_title")
        official_title = safe_find_text(root, "official_title")
        overall_status = safe_find_text(root, "overall_status")
        why_stopped = safe_find_text(root, "why_stopped")

        # Phase
        phase_elem = root.find("phase")
        phase = phase_elem.text if phase_elem is not None and phase_elem.text else ""

        # Study type
        study_type = safe_find_text(root, "study_type")

        # Enrollment
        enrollment_elem = root.find("enrollment")
        enrollment = 0
        if enrollment_elem is not None:
            try:
                enrollment = int(enrollment_elem.text) if enrollment_elem.text else 0
            except (ValueError, TypeError):
                enrollment = 0

        # Conditions
        conditions = [cond.text for cond in safe_findall(root, "condition") if cond.text]
        conditions_str = "; ".join(conditions) if conditions else ""

        # Interventions
        interventions = []
        intervention_names = []
        for intervention in safe_findall(root, "intervention"):
            intervention_type = safe_find_text(intervention, "intervention_type")
            intervention_name = safe_find_text(intervention, "intervention_name")
            if intervention_name:
                interventions.append({
                    "type": intervention_type,
                    "name": intervention_name
                })
                intervention_names.append(intervention_name)

        intervention_names_str = "; ".join(intervention_names) if intervention_names else ""

        # Sponsor
        sponsor_class = safe_find_text(root, "sponsors/lead_sponsor/agency_class")
        sponsor_name = safe_find_text(root, "sponsors/lead_sponsor/agency")

        # Dates
        start_date = safe_find_text(root, "start_date")
        completion_date = safe_find_text(root, "completion_date")
        primary_completion_date = safe_find_text(root, "primary_completion_date")

        # Results availability
        has_results_elem = root.find("clinical_results")
        has_results = has_results_elem is not None

        # Study design
        study_design_info = root.find("study_design_info")
        allocation = safe_find_text(study_design_info, "allocation")
        intervention_model = safe_find_text(study_design_info, "intervention_model")
        primary_purpose = safe_find_text(study_design_info, "primary_purpose")
        masking = safe_find_text(study_design_info, "masking")

        # Eligibility
        eligibility = root.find("eligibility")
        gender = safe_find_text(eligibility, "gender")
        minimum_age = safe_find_text(eligibility, "minimum_age")
        maximum_age = safe_find_text(eligibility, "maximum_age")

        # Locations (count)
        locations = safe_findall(root, "location")
        num_locations = len(locations)

        # Country information
        countries = list(set([safe_find_text(loc, "country") for loc in locations if safe_find_text(loc, "country")]))
        num_countries = len(countries)

        # Detailed description
        detailed_description = safe_find_text(root, "detailed_description/textblock")
        brief_summary = safe_find_text(root, "brief_summary/textblock")

        # Keywords
        keywords = [kw.text for kw in safe_findall(root, "keyword") if kw.text]
        keywords_str = "; ".join(keywords) if keywords else ""

        # MeSH terms
        condition_mesh = [mesh.text for mesh in safe_findall(root, "condition_browse/mesh_term") if mesh.text]
        intervention_mesh = [mesh.text for mesh in safe_findall(root, "intervention_browse/mesh_term") if mesh.text]

        return {
            "nct_id": nct_id,
            "brief_title": brief_title,
            "official_title": official_title,
            "overall_status": overall_status,
            "why_stopped": why_stopped,
            "phase": phase,
            "study_type": study_type,
            "enrollment": enrollment,
            "conditions": conditions_str,
            "intervention_names": intervention_names_str,
            "interventions": interventions,  # Full list with types
            "sponsor_class": sponsor_class,
            "sponsor_name": sponsor_name,
            "start_date": start_date,
            "completion_date": completion_date,
            "primary_completion_date": primary_completion_date,
            "has_results": has_results,
            "allocation": allocation,
            "intervention_model": intervention_model,
            "primary_purpose": primary_purpose,
            "masking": masking,
            "gender": gender,
            "minimum_age": minimum_age,
            "maximum_age": maximum_age,
            "num_locations": num_locations,
            "num_countries": num_countries,
            "countries": "; ".join(countries) if countries else "",
            "detailed_description": detailed_description,
            "brief_summary": brief_summary,
            "keywords": keywords_str,
            "condition_mesh_terms": condition_mesh,
            "intervention_mesh_terms": intervention_mesh,
        }

    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None


def is_phase_2_or_3(phase: str) -> bool:
    """Check if trial is Phase 2 or Phase 3"""
    if not phase:
        return False
    phase_lower = phase.lower()
    return "phase 2" in phase_lower or "phase 3" in phase_lower


def is_completed_or_terminated(status: str) -> bool:
    """Check if trial status qualifies for analysis"""
    if not status:
        return False
    status_lower = status.lower()
    return status_lower in ["completed", "terminated", "withdrawn", "suspended"]


def is_interventional(study_type: str) -> bool:
    """Check if study is interventional"""
    if not study_type:
        return False
    return study_type.lower() == "interventional"


def contains_antibody(intervention_names: str) -> bool:
    """
    Check if any intervention name suggests an antibody therapy.

    Common antibody suffixes: -mab, -zumab, -mumab, -ximab, -umab, -tuzumab, etc.
    """
    if not intervention_names:
        return False

    intervention_lower = intervention_names.lower()

    # Check for common antibody patterns
    antibody_patterns = [
        "mab",  # monoclonal antibody
        "antibody",
        "antibodies",
        "-mab",
        "-zumab",
        "-mumab",
        "-ximab",
        "-umab",
        "-tuzumab",
        "immunoglobulin",
    ]

    return any(pattern in intervention_lower for pattern in antibody_patterns)


def filter_antibody_phase23_trial(trial_data: Dict) -> bool:
    """
    Filter for Phase 2/3 antibody trials that are completed/terminated.

    Returns True if trial meets criteria:
    - Phase 2 or Phase 3
    - Completed, terminated, withdrawn, or suspended
    - Interventional study
    - Contains antibody intervention
    """
    if not trial_data:
        return False

    return (
        is_phase_2_or_3(trial_data.get("phase", ""))
        and is_completed_or_terminated(trial_data.get("overall_status", ""))
        and is_interventional(trial_data.get("study_type", ""))
        and contains_antibody(trial_data.get("intervention_names", ""))
    )
