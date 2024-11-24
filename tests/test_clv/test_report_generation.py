import matplotlib.pyplot as plt
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

class ModelStatisticalAnalyzer:
    def run_statistical_tests(self):
        # Add test implementation
        return {
            'convergence_tests': {
                'r_hat': {'trend_statistic': 0.5, 'trend_pvalue': 0.04},
                'ess': {'autocorr_statistic': 0.3, 'autocorr_pvalue': 0.06}
            },
            'stationarity_tests': {
                'duration': {'pvalue': 0.03}
            }
        }
    
    def get_performance_metrics(self):
        # Add test implementation
        return [0.8, 0.85, 0.9, 0.88, 0.92]
    
    def get_convergence_metrics(self):
        # Add test implementation
        return [1.5, 1.3, 1.2, 1.15, 1.08]

class ModelMetadataAnalyzer:
    def analyze_runs(self):
        # Add test implementation
        class Analysis:
            def __init__(self):
                self.run_summary = {
                    'total_runs': [100],
                    'successful_runs': [95],
                    'avg_duration': [2.5],
                    'avg_r_hat': [1.05]
                }
                self.recommendations = [
                    "Sample recommendation 1",
                    "Sample recommendation 2"
                ]
        return Analysis()

class ModelAnalysisReport:
    """Generate PDF reports for model analysis"""
    
    def __init__(
        self,
        stat_analyzer: ModelStatisticalAnalyzer,
        metadata_analyzer: ModelMetadataAnalyzer
    ):
        if not stat_analyzer or not metadata_analyzer:
            raise ValueError("Both stat_analyzer and metadata_analyzer must be provided")
            
        self.stat_analyzer = stat_analyzer
        self.metadata_analyzer = metadata_analyzer
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(
            ParagraphStyle(
                name='Header',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
        )
        self.styles.add(
            ParagraphStyle(
                name='SubHeader',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceAfter=20
            )
        )
        self.styles.add(
            ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=12
            )
        )
    
    def generate_report(self, output_dir: Path):
        """Generate comprehensive PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = output_dir / f'model_analysis_report_{timestamp}.pdf'
        
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build report content
        story = []
        
        # Title
        story.append(Paragraph("Model Analysis Report", self.styles['Header']))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.extend(self._create_executive_summary())
        
        # Statistical Analysis
        story.extend(self._create_statistical_analysis())
        
        # Performance Analysis
        story.extend(self._create_performance_analysis())
        
        # Convergence Analysis
        story.extend(self._create_convergence_analysis())
        
        # Recommendations
        story.extend(self._create_recommendations())
        
        # Build and save PDF
        doc.build(story)
        return pdf_path
    
    def _create_executive_summary(self) -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SubHeader']))
        
        # Get key metrics
        analysis = self.metadata_analyzer.analyze_runs()
        stats = self.stat_analyzer.run_statistical_tests()
        
        # Updated to handle potential None/missing values
        total_runs = analysis.run_summary.get('total_runs', [0])[0]
        successful_runs = analysis.run_summary.get('successful_runs', [0])[0]
        avg_duration = analysis.run_summary.get('avg_duration', [0.0])[0]
        avg_r_hat = analysis.run_summary.get('avg_r_hat', [0.0])[0]
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Runs', str(total_runs)],
            ['Success Rate', f"{successful_runs/total_runs:.1%}" if total_runs > 0 else "0.0%"],
            ['Average Duration', f"{avg_duration:.2f}s"],
            ['Average R-hat', f"{avg_r_hat:.3f}"]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_statistical_analysis(self) -> List:
        """Create statistical analysis section"""
        elements = []
        
        elements.append(Paragraph("Statistical Analysis", self.styles['SubHeader']))
        
        # Add statistical test results
        stats = self.stat_analyzer.run_statistical_tests()
        
        # Create statistical summary table
        stat_data = [
            ['Test', 'Statistic', 'p-value', 'Significant'],
            ['R-hat Trend', 
             f"{stats['convergence_tests']['r_hat']['trend_statistic']:.3f}",
             f"{stats['convergence_tests']['r_hat']['trend_pvalue']:.3f}",
             '✓' if stats['convergence_tests']['r_hat']['trend_pvalue'] < 0.05 else '✗'],
            ['ESS Autocorrelation',
             f"{stats['convergence_tests']['ess']['autocorr_statistic']:.3f}",
             f"{stats['convergence_tests']['ess']['autocorr_pvalue']:.3f}",
             '✓' if stats['convergence_tests']['ess']['autocorr_pvalue'] < 0.05 else '✗']
        ]
        
        stat_table = Table(stat_data)
        stat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(stat_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_performance_analysis(self) -> List:
        """Create performance analysis section"""
        elements = []
        
        elements.append(Paragraph("Performance Analysis", self.styles['SubHeader']))
        
        # Add performance plots
        performance_plot = io.BytesIO()
        self._plot_performance_summary(performance_plot)
        elements.append(Image(performance_plot, width=6*inch, height=4*inch))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_convergence_analysis(self) -> List:
        """Create convergence analysis section"""
        elements = []
        
        elements.append(Paragraph("Convergence Analysis", self.styles['SubHeader']))
        
        # Add convergence plots
        convergence_plot = io.BytesIO()
        self._plot_convergence_summary(convergence_plot)
        elements.append(Image(convergence_plot, width=6*inch, height=4*inch))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_recommendations(self) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("Recommendations", self.styles['SubHeader']))
        
        # Get recommendations from analyzers
        stat_results = self.stat_analyzer.run_statistical_tests()
        metadata_analysis = self.metadata_analyzer.analyze_runs()
        
        recommendations = []
        
        # Add statistical-based recommendations
        if stat_results['convergence_tests']['r_hat']['trend_pvalue'] < 0.05:
            recommendations.append(
                "Consider increasing the number of samples or tuning steps "
                "to improve convergence"
            )
            
        if stat_results['stationarity_tests']['duration']['pvalue'] < 0.05:
            recommendations.append(
                "Performance metrics show non-stationarity. "
                "Consider investigating system stability"
            )
            
        # Add metadata-based recommendations
        recommendations.extend(metadata_analysis.recommendations)
        
        # Create recommendations list
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", self.styles['BodyText']))
        
        return elements
    
    def _plot_performance_summary(self, output):
        """Create performance summary plot"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Get performance data
            perf_data = self.stat_analyzer.get_performance_metrics()
            
            # Plot performance metrics
            if perf_data:
                x = range(len(perf_data))
                plt.plot(x, perf_data, 'b-', label='Performance')
                plt.xlabel('Run Number')
                plt.ylabel('Performance Metric')
                plt.title('Model Performance Over Time')
                plt.legend()
            
            plt.savefig(output, format='png')
            plt.close()
        except Exception as e:
            print(f"Error generating performance plot: {str(e)}")
            # Create a simple placeholder plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Performance data unavailable', 
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig(output, format='png')
            plt.close()
    
    def _plot_convergence_summary(self, output):
        """Create convergence summary plot"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Get convergence data
            conv_data = self.stat_analyzer.get_convergence_metrics()
            
            # Plot convergence metrics
            if conv_data:
                x = range(len(conv_data))
                plt.plot(x, conv_data, 'r-', label='R-hat')
                plt.axhline(y=1.1, color='k', linestyle='--', label='Convergence Threshold')
                plt.xlabel('Iteration')
                plt.ylabel('R-hat Value')
                plt.title('Model Convergence Analysis')
                plt.legend()
            
            plt.savefig(output, format='png')
            plt.close()
        except Exception as e:
            print(f"Error generating convergence plot: {str(e)}")
            # Create a simple placeholder plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Convergence data unavailable',
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig(output, format='png')
            plt.close() 