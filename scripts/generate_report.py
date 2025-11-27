#!/usr/bin/env python3
"""
Generate HTML report with interactive visualizations for ML pipeline results.

Usage:
    python scripts/generate_report.py <results_dir> <csv_output> <html_output> [options]

Options:
    --timestamp TIMESTAMP       Timestamp when results were generated
    --workflow-url URL         GitHub Actions workflow run URL
    --repo REPO                GitHub repository (e.g., owner/repo)
"""

import sys
import json
from pathlib import Path
import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Professional color palette
COLORS = {
    'primary': ['#2563eb', '#0891b2', '#06b6d4', '#f97316', '#10b981'],
    'blue': '#2563eb',
    'teal': '#0891b2',
    'cyan': '#06b6d4',
    'orange': '#f97316',
    'colorscale': [
        [0.0, '#f1f5f9'],
        [0.3, '#a5d8dd'],
        [0.6, '#0891b2'],
        [0.8, '#06b6d4'],
        [1.0, '#2563eb']
    ]
}


def aggregate_results(results_dir, csv_output):
    """Aggregate all results_*.json files into a single CSV."""
    results_path = Path(results_dir)
    results = [json.load(open(f)) for f in sorted(results_path.glob('results_*.json'))]

    if not results:
        print(f"No results_*.json files found in {results_dir}")
        sys.exit(1)

    df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_output, index=False)

    print(f"\n{'='*80}")
    print("MODEL COMPARISON (Sorted by ROC AUC)")
    print('='*80)
    print(df[['model_name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    print(f"\nBest Model: {df.iloc[0]['model_name']}")
    print(f"Best ROC AUC: {df.iloc[0]['roc_auc']:.4f}")
    print(f"\nCSV saved to: {csv_output}")
    print('='*80)

    return df


def load_results(results_dir):
    """Load all model results from JSON files."""
    results = [json.load(open(f)) for f in sorted(Path(results_dir).glob('results_*.json'))]
    if not results:
        raise FileNotFoundError(f"No results_*.json files found in {results_dir}")
    return sorted(results, key=lambda x: x['roc_auc'], reverse=True)


def calculate_dataset_statistics(results):
    """Calculate dataset statistics from model results."""
    if not results:
        return None

    # Use actual sizes from results if available, otherwise calculate from confusion matrix
    if 'total_samples' in results[0] and 'train_size' in results[0] and 'test_size' in results[0]:
        total_samples = results[0]['total_samples']
        train_size = results[0]['train_size']
        test_size = results[0]['test_size']
    else:
        # Fallback: calculate from confusion matrix (assumes 80/20 split)
        cm = results[0]['confusion_matrix']
        test_size = sum(sum(row) for row in cm)
        total_samples = int(test_size / 0.2)
        train_size = total_samples - test_size

    cm = results[0]['confusion_matrix']
    class_0_count = cm[0][0] + cm[0][1]
    class_1_count = cm[1][0] + cm[1][1]

    # Calculate actual failure rate
    failure_rate = (class_0_count / test_size) * 100

    # Get feature count from best model if available
    feature_count = 'TBD'
    if results[0].get('feature_importance'):
        feature_data = results[0]['feature_importance']
        if 'feature_names' in feature_data:
            feature_count = len(feature_data['feature_names'])

    # Calculate actual train/test percentages
    train_pct = (train_size / total_samples * 100) if total_samples > 0 else 0
    test_pct = (test_size / total_samples * 100) if total_samples > 0 else 0

    return {
        'total_samples': total_samples,
        'train_size': train_size,
        'test_size': test_size,
        'train_pct': train_pct,
        'test_pct': test_pct,
        'class_0_count': class_0_count,
        'class_1_count': class_1_count,
        'class_0_pct': (class_0_count / test_size) * 100,
        'class_1_pct': (class_1_count / test_size) * 100,
        'failure_rate': failure_rate,
        'feature_count': feature_count
    }


def create_plot(plot_func, div_id):
    """Helper to create plot HTML with consistent settings."""
    fig = plot_func()
    return fig.to_html(include_plotlyjs='cdn' if div_id == 'metrics-comparison' else False, div_id=div_id)


def create_metrics_comparison(results):
    """Create bar chart comparing all metrics across models."""
    models = [r['model_name'] for r in results]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'CV ROC AUC'),
        specs=[[{'type': 'bar'}] * 3, [{'type': 'bar'}] * 3]
    )

    metrics = [('accuracy', 1, 1), ('precision', 1, 2), ('recall', 1, 3),
               ('f1', 2, 1), ('roc_auc', 2, 2), ('cv_roc_auc_mean', 2, 3)]

    for metric_name, row, col in metrics:
        values = [r[metric_name] for r in results]
        fig.add_trace(
            go.Bar(
                x=models, y=values,
                marker_color=COLORS['primary'],
                text=[f'{v:.3f}' for v in values],
                textposition='outside',
                showlegend=False,
                hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
        fig.update_yaxes(range=[0, 1.1], row=row, col=col)

    fig.update_layout(
        height=600,
        title_text="Model Performance Comparison",
        title_font_size=20,
        showlegend=False,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color='#1e293b')
    )

    return fig


def create_confusion_matrices(results):
    """Create heatmap visualizations for all confusion matrices."""
    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[r['model_name'] for r in results],
        specs=[[{'type': 'heatmap'} for _ in range(cols)] for _ in range(rows)],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    annotations = []

    for idx, result in enumerate(results):
        subplot_row = idx // cols + 1
        subplot_col = idx % cols + 1
        cm = result['confusion_matrix']
        total = sum(sum(row) for row in cm)
        cm_pct = [[val/total*100 for val in row] for row in cm]

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale=COLORS['colorscale'],
                showscale=(idx == 0),
                hovertemplate='%{y} / %{x}<br>Count: %{z}<extra></extra>'
            ),
            row=subplot_row, col=subplot_col
        )

        # Add text annotations
        max_val = max(max(row) for row in cm)
        threshold = max_val * 0.5
        x_labels = ['Predicted 0', 'Predicted 1']
        y_labels = ['Actual 0', 'Actual 1']

        for i in range(2):
            for j in range(2):
                # Use white text for dark backgrounds, dark text for light backgrounds
                text_color = 'white' if cm[i][j] > threshold else '#1e293b'
                # Calculate subplot position for axis references
                # Plotly subplot axes: first is 'x'/'y', rest are 'x2'/'y2', 'x3'/'y3', etc.
                axis_num = subplot_row * cols - cols + subplot_col
                x_axis = 'x' if axis_num == 1 else f'x{axis_num}'
                y_axis = 'y' if axis_num == 1 else f'y{axis_num}'
                annotations.append(dict(
                    x=x_labels[j], y=y_labels[i],
                    text=f'<b>{cm[i][j]}</b><br>({cm_pct[i][j]:.1f}%)',
                    xref=x_axis, yref=y_axis,
                    showarrow=False,
                    font=dict(color=text_color, size=13, family='Inter, sans-serif'),
                    xanchor='center', yanchor='middle'
                ))

    # Preserve existing annotations (subplot titles) and add our annotations
    existing_annotations = list(fig.layout.annotations)
    all_annotations = existing_annotations + annotations

    fig.update_layout(
        height=300 * rows,
        title_text="Confusion Matrices",
        title_font_size=20,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color='#1e293b'),
        annotations=all_annotations
    )

    return fig


def create_cv_scores_plot(results):
    """Create error bar plot for cross-validation scores."""
    models = [r['model_name'] for r in results]
    means = [r['cv_roc_auc_mean'] for r in results]
    stds = [r['cv_roc_auc_std'] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=means,
        error_y=dict(type='data', array=stds),
        marker_color=COLORS['primary'],
        text=[f'{m:.3f}±{s:.3f}' for m, s in zip(means, stds)],
        textposition='outside',
        hovertemplate='%{x}<br>Mean: %{y:.4f}<br>Std: %{error_y.array:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='Cross-Validation ROC AUC Scores (3-Fold)',
        xaxis_title='Model',
        yaxis_title='ROC AUC',
        yaxis_range=[0, 1.1],
        height=400,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(family="Inter, sans-serif", color='#1e293b')
    )

    return fig


def create_feature_importance_plot(best_model):
    """Create horizontal bar chart for feature importance from best model."""
    if not best_model.get('feature_importance'):
        return None

    feature_data = best_model['feature_importance']
    feature_names = feature_data['feature_names'][:15][::-1]
    importances = feature_data['importances'][:15][::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importances, y=feature_names,
        orientation='h',
        marker=dict(
            color=importances,
            colorscale=[[0.0, '#a5d8dd'], [0.5, '#0891b2'], [1.0, '#2563eb']],
            showscale=False
        ),
        text=[f'{imp:.3f}' for imp in importances],
        textposition='outside',
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Top 15 Most Important Features - {best_model["model_name"]}',
        xaxis_title='Feature Importance Score',
        yaxis_title='',
        height=max(400, len(feature_names) * 30),
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color='#1e293b'),
        margin=dict(l=200, r=100, t=60, b=60)
    )

    return fig


def create_results_table(results):
    """Create formatted HTML table of results."""
    rows = []
    for idx, r in enumerate(results):
        is_best = idx == 0
        badge = '<span class="badge">Best</span>' if is_best else ''
        time_str = f"{r.get('training_time', 0):.2f}s"

        rows.append(f"""
        <tr class="{'best-model' if is_best else ''}">
            <td><strong>{r['model_name']}</strong>{badge}</td>
            <td>{r['accuracy']:.4f}</td>
            <td>{r['precision']:.4f}</td>
            <td>{r['recall']:.4f}</td>
            <td>{r['f1']:.4f}</td>
            <td><strong>{r['roc_auc']:.4f}</strong></td>
            <td>{r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}</td>
            <td>{time_str}</td>
        </tr>
        """)

    return '\n'.join(rows)


def build_content_sections(results, dataset_stats, best_model):
    """Build all content sections for the report."""
    sections = []

    # Project Overview section
    sections.append(f'''
        <section class="section animate-in">
            <div class="section-header">
                <h2 class="section-title">Project Overview</h2>
                <p class="section-description">Real-world ML challenge: predicting clinical trial success for antibody therapeutics</p>
            </div>
            <div class="card">
                <div class="info-grid">
                    <div class="info-block">
                        <h3>Problem Statement
                            <span class="info-tooltip">
                                <span class="info-icon">i</span>
                                <span class="tooltip-content">
                                    <strong>Why Antibody Trials?</strong><br>Antibodies represent ~30% of the pharmaceutical development pipeline with high commercial value ($237B+ market), but face historically high failure rates.
                                </span>
                            </span>
                        </h3>
                        <p>Predict probability of success for Phase 2 & 3 antibody clinical trials. With {dataset_stats['failure_rate']:.1f}% failure rate observed in our dataset and costs of $10-100M+ per trial, early prediction can save pharmaceutical companies significant resources.</p>
                    </div>
                    <div class="info-block">
                        <h3>Data Source</h3>
                        <p><strong>ClinicalTrials.gov Bulk XML Download</strong></p>
                        <p>Streaming parse from S3-hosted bulk XML archive</p>
                        <p style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">Phase 2/3 interventional antibody trials (completed/terminated/withdrawn status)</p>
                    </div>                    <div class="info-block">
                        <h3>Success Labeling Strategy</h3>
                        <p><strong>Refined Binary Classification:</strong></p>
                        <ul>
                            <li><strong>Success (1):</strong> COMPLETED, APPROVED_FOR_MARKETING, AVAILABLE</li>
                            <li><strong>Failure (0):</strong> TERMINATED/WITHDRAWN/SUSPENDED for efficacy or safety reasons</li>
                            <li><strong>Excluded:</strong> Trials terminated for administrative reasons (funding, enrollment, business decisions)</li>
                        </ul>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: var(--text-secondary);">This approach focuses on efficacy outcomes rather than administrative completion.</p>
                    </div>
                </div>
            </div>
    ''')

    # Dataset statistics if available
    if dataset_stats:
        sections.append(f'''
            <div class="card">
                <h3 style="margin-bottom: 1rem; color: var(--primary-cyan); font-size: 1.25rem; font-weight: 600;">Dataset Statistics</h3>
                <div class="info-grid">
                    <div class="info-block">
                        <h3>Sample Counts</h3>
                        <ul>
                            <li><strong>Total Samples:</strong> {dataset_stats['total_samples']:,} trials</li>
                            <li><strong>Training Set:</strong> {dataset_stats['train_size']:,} samples ({dataset_stats['train_pct']:.1f}%)</li>
                            <li><strong>Test Set:</strong> {dataset_stats['test_size']:,} samples ({dataset_stats['test_pct']:.1f}%)</li>
                        </ul>
                    </div>
                    <div class="info-block">
                        <h3>Class Distribution (Test Set)</h3>
                        <ul>
                            <li><strong>Failure (Class 0):</strong> {dataset_stats['class_0_count']:,} trials ({dataset_stats['class_0_pct']:.1f}%)</li>
                            <li><strong>Success (Class 1):</strong> {dataset_stats['class_1_count']:,} trials ({dataset_stats['class_1_pct']:.1f}%)</li>
                        </ul>
                    </div>
                    <div class="info-block">
                        <h3>Data Quality</h3>
                        <ul>
                            <li><strong>Source:</strong> ClinicalTrials.gov Bulk XML (S3)</li>
                            <li><strong>Validation:</strong> Random stratified train/test split (80/20)</li>
                            <li><strong>Features:</strong> 32 domain-specific engineered features</li>
                        </ul>
                    </div>
                </div>
            </div>
        ''')

    # Feature engineering card
    sections.append(f'''
            <div class="card">
                <h3 style="margin-bottom: 0.75rem; color: var(--primary-cyan); font-size: 1.25rem; font-weight: 600;">
                    Feature Engineering - 32 Features
                </h3>
                <p style="color: var(--text-body); margin-bottom: 1rem;">Domain-specific feature set:</p>
                <div class="feature-pills">
                    <span class="pill">Antibody Type & Mechanism (12 features)</span>
                    <span class="pill">Trial Design & Enrollment (8 features)</span>
                    <span class="pill">Disease Categories (8 features)</span>
                    <span class="pill">Sponsor Characteristics (3 features)</span>
                    <span class="pill">Temporal Features (4 features + has_start_date flag)</span>
                    <span class="pill">Trial Metadata (1 feature)</span>
                </div>
                <p style="margin-top: 1rem; font-size: 0.9rem; color: var(--text-secondary);">
                    <strong>Antibody-specific features:</strong> Type classification (murine/chimeric/humanized/fully_human), target mechanisms (checkpoint inhibitors, growth factors, cytokines, CD markers), biomarker selection, combination therapy indicators
                    <br><br>
                    <strong>Methodology:</strong> Post-2024 trials filtered (22), text features excluded, has_start_date flag for missing dates (45%), stratified random split
                </p>
            </div>
        </section>
    ''')

    # Model performance table
    sections.append(f'''
        <section class="section animate-in">
            <div class="section-header">
                <h2 class="section-title">Model Performance Summary</h2>
                <p class="section-description">Comprehensive comparison of all 5 models across key metrics</p>
            </div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                            <th>ROC AUC</th>
                            <th>CV ROC AUC (Mean ± Std)</th>
                            <th>Training Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {create_results_table(results)}
                    </tbody>
                </table>
            </div>
        </section>

        <section class="section animate-in">
            <div class="section-header">
                <h2 class="section-title">Performance Metrics Comparison</h2>
                <p class="section-description">Interactive visualization of model performance across all evaluation metrics</p>
            </div>
            <div class="chart-container">
                {create_plot(lambda: create_metrics_comparison(results), 'metrics-comparison')}
            </div>
        </section>

        <section class="section animate-in">
            <div class="section-header">
                <h2 class="section-title">Cross-Validation Results</h2>
                <p class="section-description">Model consistency evaluated through 3-fold stratified cross-validation</p>
            </div>
            <div class="chart-container">
                {create_plot(lambda: create_cv_scores_plot(results), 'cv-scores')}
            </div>
        </section>
    ''')

    # Feature importance if available
    feature_plot = create_feature_importance_plot(best_model)
    if feature_plot:
        sections.append(f'''
        <section class="section animate-in">
            <div class="section-header">
                <h2 class="section-title">Feature Importance Analysis</h2>
                <p class="section-description">Most influential features for predicting clinical trial success using the best model ({best_model['model_name']})</p>
            </div>
            <div class="chart-container">
                {feature_plot.to_html(include_plotlyjs=False, div_id='feature-importance')}
            </div>
        </section>
        ''')

    # Confusion matrices
    sections.append(f'''
        <section class="section animate-in">
            <div class="section-header">
                <h2 class="section-title">Confusion Matrices</h2>
                <p class="section-description">Detailed breakdown of predictions vs. actual outcomes for each model</p>
            </div>
            <div class="chart-container">
                {create_plot(lambda: create_confusion_matrices(results), 'confusion-matrices')}
            </div>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 1rem; text-align: center;">
                Values show absolute counts with percentages. Class 0 = Failure | Class 1 = Success
            </p>
        </section>
    ''')

    return '\n'.join(sections)


def generate_html_report(results, output_file, timestamp=None, workflow_url=None, github_repo=None):
    """Generate complete HTML report with all visualizations."""
    best_model = results[0]
    dataset_stats = calculate_dataset_statistics(results)

    # Load template
    template_path = Path(__file__).parent / 'templates' / 'dashboard_template.html'
    with open(template_path) as f:
        template = f.read()

    # Build content
    content = build_content_sections(results, dataset_stats, best_model)

    # Build summary cards (metrics grid)
    summary_cards_html = f'''
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 2rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08); position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--primary-blue);"></div>
            <div style="font-size: 0.875rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">Best Model</div>
            <div style="font-size: 2.5rem; font-weight: 700; line-height: 1; color: var(--primary-blue);">{best_model['model_name']}</div>
        </div>
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 2rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08); position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--primary-teal);"></div>
            <div style="font-size: 0.875rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">ROC AUC Score</div>
            <div style="font-size: 2.5rem; font-weight: 700; line-height: 1; color: var(--primary-teal);">{best_model['roc_auc']:.4f}</div>
        </div>
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 2rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08); position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--accent-orange);"></div>
            <div style="font-size: 0.875rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">F1 Score</div>
            <div style="font-size: 2.5rem; font-weight: 700; line-height: 1; color: var(--accent-orange);">{best_model['f1']:.4f}</div>
        </div>
    </div>
    '''

    # Render template using replace (like EDA scripts)
    html = template.replace('{{ title }}', 'Clinical Trial Outcome Prediction')
    html = html.replace('{{ header_icon }}', '')
    html = html.replace('{{ raw_active }}', '')
    html = html.replace('{{ features_active }}', '')
    html = html.replace('{{ metadata.timestamp }}', timestamp or 'Not specified')
    html = html.replace('{{ metadata.data_source }}', 'ClinicalTrials.gov API')
    html = html.replace('{{ metadata.workflow_url }}', workflow_url or '#')
    html = html.replace('{{ summary_cards }}', summary_cards_html)
    html = html.replace('{{ content }}', content)
    html = html.replace('{{ insights }}', '')

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ HTML report generated: {output_path}")
    print(f"✓ Best model: {best_model['model_name']} (ROC AUC: {best_model['roc_auc']:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate model comparison CSV and HTML dashboard for ML pipeline results.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('results_dir', help='Directory containing results_*.json files')
    parser.add_argument('csv_output', help='Output CSV file path')
    parser.add_argument('html_output', help='Output HTML file path')
    parser.add_argument('--timestamp', help='Timestamp when results were generated')
    parser.add_argument('--workflow-url', help='GitHub Actions workflow run URL')
    parser.add_argument('--repo', help='GitHub repository (owner/repo format)')

    args = parser.parse_args()

    try:
        aggregate_results(args.results_dir, args.csv_output)
        results = load_results(args.results_dir)
        generate_html_report(
            results,
            args.html_output,
            timestamp=args.timestamp,
            workflow_url=args.workflow_url,
            github_repo=args.repo
        )
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
