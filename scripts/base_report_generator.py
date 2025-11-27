#!/usr/bin/env python3
"""
Base class for generating EDA HTML reports

Provides common functionality for loading data, generating HTML reports,
and handling templates.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from jinja2 import Template
from site_config import SITE_CONFIG


class BaseReportGenerator(ABC):
    """Base class for EDA report generation"""

    def __init__(self, template_name: str = 'dashboard_template.html'):
        """
        Initialize the report generator

        Args:
            template_name: Name of the HTML template file in templates/
        """
        self.template_path = Path(__file__).parent / 'templates' / template_name

    def load_data(self, csv_path: Path) -> pd.DataFrame:
        """
        Load and prepare data from CSV

        Args:
            csv_path: Path to the CSV file

        Returns:
            Loaded DataFrame
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df):,} rows with {len(df.columns)} columns")
        return df

    @abstractmethod
    def get_charts(self, df: pd.DataFrame) -> Dict:
        """
        Create all charts for the report

        Args:
            df: DataFrame containing the data

        Returns:
            Dictionary mapping chart names to Plotly Figure objects
        """
        pass

    @abstractmethod
    def build_summary_stats(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Build summary statistics for the report header

        Args:
            df: DataFrame containing the data

        Returns:
            List of (label, value) tuples for summary cards
        """
        pass

    @abstractmethod
    def get_chart_sections(self) -> List[Tuple[str, List[str]]]:
        """
        Define chart sections and their organization

        Returns:
            List of (section_title, chart_names) tuples
        """
        pass

    @abstractmethod
    def get_report_title(self) -> str:
        """
        Get the report title

        Returns:
            Report title string
        """
        pass

    @abstractmethod
    def get_nav_active_states(self) -> Dict[str, str]:
        """
        Get navigation active states

        Returns:
            Dictionary with 'overview', 'raw', 'features' keys and 'active' or '' values
        """
        pass

    def build_summary_cards_html(self, stats: List[Tuple[str, str]]) -> str:
        """
        Build HTML for summary statistics cards

        Args:
            stats: List of (label, value) tuples

        Returns:
            HTML string for summary cards
        """
        cards = []
        for label, value in stats:
            cards.extend([
                '<div class="stat-card">',
                f'    <div class="stat-label">{label}</div>',
                f'    <div class="stat-value">{value}</div>',
                '</div>',
            ])

        return '<div class="summary-stats">\n' + '\n'.join(cards) + '\n</div>'

    def build_content_html(self, charts: Dict, sections: Optional[List[Tuple[str, List[str]]]] = None) -> str:
        """
        Build HTML for chart content

        Args:
            charts: Dictionary of chart name -> Figure
            sections: Optional list of (section_title, chart_names) tuples for organization

        Returns:
            HTML string for chart content
        """
        content_parts = []

        if sections:
            # Organize charts by section
            for section_title, chart_names in sections:
                section_charts = {name: charts.get(name) for name in chart_names if name in charts}

                if not section_charts:
                    continue

                content_parts.append(f'<h2>{section_title}</h2>')

                for name, fig in section_charts.items():
                    chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'chart-{name}')
                    content_parts.append(f'<div class="chart">{chart_html}</div>')
        else:
            # No sections, just add all charts
            for name, fig in charts.items():
                chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'chart-{name}')
                content_parts.append(f'<div class="chart">{chart_html}</div>')

        return '\n'.join(content_parts)

    def generate_html_report(self, df: pd.DataFrame, output_path: Path, metadata: dict):
        """
        Generate complete HTML report

        Args:
            df: DataFrame containing the data
            output_path: Path where the HTML report will be saved
            metadata: Dictionary containing report metadata (timestamp, data_source, workflow_url)
        """
        print("\nGenerating HTML report...")

        # Load template
        with open(self.template_path, 'r') as f:
            template = Template(f.read())

        # Create all charts
        charts = self.get_charts(df)

        # Remove None charts
        charts = {k: v for k, v in charts.items() if v is not None}

        # Build summary stats
        summary_stats = self.build_summary_stats(df)
        summary_cards_html = self.build_summary_cards_html(summary_stats)

        # Build content HTML
        sections = self.get_chart_sections()
        content_html = self.build_content_html(charts, sections)

        # Get navigation states (now just need to know which page is active)
        nav_states = self.get_nav_active_states()
        active_page = next((k for k, v in nav_states.items() if v == 'active'), None)

        # Render template with Jinja2
        html = template.render(
            config=SITE_CONFIG,
            title=self.get_report_title(),
            active_page=active_page,
            metadata={
                'timestamp': metadata['timestamp'],
                'data_source': metadata.get('data_source', SITE_CONFIG['defaults']['data_source']),
                'workflow_url': metadata.get('workflow_url', '#'),
            },
            summary_cards=summary_cards_html,
            content=content_html,
            insights=metadata.get('insights', ''),
        )

        # Write HTML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"✓ Report saved to {output_path}")

    def run(self, input_csv: str, output_html: str, **metadata_kwargs):
        """
        Main entry point to generate report

        Args:
            input_csv: Path to input CSV file
            output_html: Path to output HTML file
            **metadata_kwargs: Additional metadata (timestamp, data_source, workflow_url, etc.)
        """
        # Load data
        df = self.load_data(Path(input_csv))

        # Set default metadata
        metadata = {
            'timestamp': metadata_kwargs.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')),
            'data_source': metadata_kwargs.get('data_source', 'ClinicalTrials.gov API'),
            'workflow_url': metadata_kwargs.get('workflow_url', '#'),
            'insights': metadata_kwargs.get('insights', ''),
        }

        # Generate report
        self.generate_html_report(df, Path(output_html), metadata)

        print("\n" + "="*60)
        print(f"✓ {self.get_report_title()} Generated Successfully")
        print("="*60)
