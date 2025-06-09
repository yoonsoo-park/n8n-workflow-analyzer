"""
Visualization module for n8n workflow analysis.

This module provides visualization capabilities for workflows, patterns,
networks, and analysis results.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import base64
from datetime import datetime

# Data processing
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# For interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

try:
    from ..models import N8nWorkflow, WorkflowCollection, WorkflowNode, WorkflowConnection
    from ..analysis.workflow_analyzer import WorkflowAnalyzer
    from ..mining.pattern_miner import PatternMiner
    from ..network.network_analyzer import NetworkAnalyzer
    from ..config import get_analysis_config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import N8nWorkflow, WorkflowCollection, WorkflowNode, WorkflowConnection
    from analysis.workflow_analyzer import WorkflowAnalyzer
    from mining.pattern_miner import PatternMiner
    from network.network_analyzer import NetworkAnalyzer
    from config import get_analysis_config


logger = logging.getLogger(__name__)


class VisualizationManager:
    """Main class for managing visualizations of workflow analysis."""
    
    def __init__(self, output_dir: str = "data/visualizations"):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "workflows").mkdir(exist_ok=True)
        (self.output_dir / "patterns").mkdir(exist_ok=True)
        (self.output_dir / "networks").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        (self.output_dir / "dashboard").mkdir(exist_ok=True)
        (self.output_dir / "collection").mkdir(exist_ok=True)
        
        # Set up visualization style
        self._setup_visualization_style()
        
        self.logger = logging.getLogger(__name__)
    
    def visualize_workflow(self, workflow: N8nWorkflow, 
                          output_format: str = "html", 
                          include_details: bool = True) -> str:
        """
        Create a visual representation of a workflow.
        
        Args:
            workflow: The workflow to visualize
            output_format: Output format (html, png, svg)
            include_details: Whether to include detailed node information
            
        Returns:
            Path to the generated visualization file
        """
        self.logger.info(f"Visualizing workflow: {workflow.name}")
        
        if output_format == "html":
            return self._create_interactive_workflow_diagram(workflow, include_details)
        else:
            return self._create_static_workflow_diagram(workflow, output_format, include_details)
    
    def visualize_workflow_collection(self, collection: WorkflowCollection,
                                     output_format: str = "html") -> str:
        """
        Create a visual overview of a workflow collection.
        
        Args:
            collection: The workflow collection to visualize
            output_format: Output format (html, png, svg)
            
        Returns:
            Path to the generated visualization file
        """
        self.logger.info(f"Visualizing collection with {collection.total_workflows} workflows")
        
        if output_format == "html":
            return self._create_interactive_collection_overview(collection)
        else:
            return self._create_static_collection_overview(collection, output_format)
    
    def visualize_pattern_analysis(self, collection: WorkflowCollection,
                                  patterns: List[Any] = None,
                                  rules: List[Any] = None,
                                  output_format: str = "html") -> str:
        """
        Visualize pattern analysis results.
        
        Args:
            collection: The workflow collection analyzed
            patterns: List of patterns found (optional)
            rules: List of association rules found (optional)
            output_format: Output format (html, png, svg)
            
        Returns:
            Path to the generated visualization file
        """
        self.logger.info("Visualizing pattern analysis results")
        
        # If patterns and rules not provided, run analysis
        if patterns is None or rules is None:
            miner = PatternMiner()
            patterns = miner.mine_frequent_patterns(collection)
            rules = miner.generate_association_rules(patterns)
        
        if output_format == "html":
            return self._create_interactive_pattern_visualization(collection, patterns, rules)
        else:
            return self._create_static_pattern_visualization(collection, patterns, rules, output_format)
    
    def visualize_network_analysis(self, collection: WorkflowCollection,
                                  workflow_id: Optional[str] = None,
                                  output_format: str = "html") -> str:
        """
        Visualize network analysis results.
        
        Args:
            collection: The workflow collection analyzed
            workflow_id: ID of specific workflow to visualize (optional)
            output_format: Output format (html, png, svg)
            
        Returns:
            Path to the generated visualization file
        """
        self.logger.info("Visualizing network analysis results")
        
        analyzer = NetworkAnalyzer()
        
        if workflow_id:
            # Visualize specific workflow
            workflow = collection.get_workflow_by_id(workflow_id)
            if not workflow:
                self.logger.error(f"Workflow with ID {workflow_id} not found")
                return ""
            
            graph = analyzer.create_workflow_graph(workflow)
            
            if output_format == "html":
                return self._create_interactive_network_visualization(graph, workflow.name)
            else:
                return self._create_static_network_visualization(graph, workflow.name, output_format)
        else:
            # Visualize collection network
            graph = analyzer.create_collection_graph(collection)
            
            if output_format == "html":
                return self._create_interactive_collection_network(graph)
            else:
                return self._create_static_collection_network(graph, output_format)
    
    def create_statistical_dashboard(self, collection: WorkflowCollection,
                                    output_format: str = "html") -> str:
        """
        Create a statistical dashboard for the workflow collection.
        
        Args:
            collection: The workflow collection to analyze
            output_format: Output format (html, png, svg)
            
        Returns:
            Path to the generated dashboard file
        """
        self.logger.info("Creating statistical dashboard")
        
        if output_format == "html":
            return self._create_interactive_statistical_dashboard(collection)
        else:
            return self._create_static_statistical_dashboard(collection, output_format)
    
    def create_comprehensive_report(self, collection: WorkflowCollection) -> str:
        """
        Create a comprehensive HTML report with all analysis results.
        
        Args:
            collection: The workflow collection to analyze
            
        Returns:
            Path to the generated report file
        """
        self.logger.info("Creating comprehensive analysis report")
        
        # Generate all visualizations
        workflow_viz = {}
        for workflow in collection.workflows:
            workflow_viz[workflow.id or workflow.name] = self.visualize_workflow(
                workflow, output_format="html", include_details=True
            )
        
        collection_viz = self.visualize_workflow_collection(collection)
        pattern_viz = self.visualize_pattern_analysis(collection)
        network_viz = self.visualize_network_analysis(collection)
        stats_dashboard = self.create_statistical_dashboard(collection)
        
        # Create report HTML
        report_path = self.output_dir / "dashboard" / "comprehensive_report.html"
        
        # Combine all visualizations into a single report
        self._create_html_report(
            report_path, collection, workflow_viz, 
            collection_viz, pattern_viz, network_viz, stats_dashboard
        )
        
        return str(report_path)
    
    def export_visualizations(self, collection: WorkflowCollection, 
                             formats: List[str] = ["html", "png", "svg"]) -> Dict[str, List[str]]:
        """
        Export all visualizations in multiple formats.
        
        Args:
            collection: The workflow collection to visualize
            formats: List of output formats
            
        Returns:
            Dictionary mapping visualization types to lists of file paths
        """
        self.logger.info(f"Exporting visualizations in formats: {formats}")
        
        exports = {
            "workflows": [],
            "collection": [],
            "patterns": [],
            "networks": [],
            "statistics": []
        }
        
        for fmt in formats:
            # Export workflow visualizations
            for workflow in collection.workflows:
                path = self.visualize_workflow(workflow, output_format=fmt)
                exports["workflows"].append(path)
            
            # Export collection overview
            path = self.visualize_workflow_collection(collection, output_format=fmt)
            exports["collection"].append(path)
            
            # Export pattern analysis
            path = self.visualize_pattern_analysis(collection, output_format=fmt)
            exports["patterns"].append(path)
            
            # Export network analysis
            path = self.visualize_network_analysis(collection, output_format=fmt)
            exports["networks"].append(path)
            
            # Export statistical dashboard
            path = self.create_statistical_dashboard(collection, output_format=fmt)
            exports["statistics"].append(path)
        
        # Create comprehensive report (HTML only)
        report_path = self.create_comprehensive_report(collection)
        exports["report"] = [report_path]
        
        return exports
    
    # Helper methods for visualization
    
    def _setup_visualization_style(self):
        """Set up the visualization style."""
        # Set Seaborn style
        sns.set_theme(style="whitegrid")
        
        # Set Matplotlib style
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12
        
        # Set Plotly template
        pio.templates.default = "plotly_white"
    
    def _create_interactive_workflow_diagram(self, workflow: N8nWorkflow, 
                                           include_details: bool) -> str:
        """Create an interactive HTML diagram of a workflow."""
        # Create a NetworkX graph
        analyzer = NetworkAnalyzer()
        graph = analyzer.create_workflow_graph(workflow)
        
        # Create a Plotly figure
        fig = go.Figure()
        
        # Node positions using spring layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge details
            edge_data = graph.get_edge_data(edge[0], edge[1])
            edge_text.append(f"From: {graph.nodes[edge[0]].get('name', edge[0])}<br>"
                            f"To: {graph.nodes[edge[1]].get('name', edge[1])}<br>"
                            f"Type: {edge_data.get('output_type', 'main')}")
        
        # Add edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        fig.add_trace(edge_trace)
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        # Color map for node types
        node_types = set(graph.nodes[node]['type'] for node in graph.nodes)
        color_map = {}
        colors = px.colors.qualitative.Plotly
        for i, node_type in enumerate(node_types):
            color_map[node_type] = colors[i % len(colors)]
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node details
            node_data = graph.nodes[node]
            node_name = node_data.get('name', node)
            node_type = node_data.get('type', 'unknown')
            
            # Node text for hover
            text = f"Name: {node_name}<br>Type: {node_type}"
            if include_details:
                text += f"<br>ID: {node}"
                if node_data.get('has_notes', False):
                    text += "<br>Has notes: Yes"
                if node_data.get('disabled', False):
                    text += "<br>Disabled: Yes"
            
            node_text.append(text)
            
            # Node color based on type
            node_color.append(color_map.get(node_type, '#000'))
            
            # Node size based on connections
            size = 15 + 5 * (graph.degree(node))
            node_size.append(size)
        
        # Add node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=2, color='white')
            )
        )
        
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=f"Workflow: {workflow.name}",
            title_font=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=800
        )
        
        # Save to HTML file
        output_path = self.output_dir / "workflows" / f"{workflow.id or 'workflow'}_diagram.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def _create_static_workflow_diagram(self, workflow: N8nWorkflow, 
                                      output_format: str,
                                      include_details: bool) -> str:
        """Create a static diagram of a workflow."""
        # Create a NetworkX graph
        analyzer = NetworkAnalyzer()
        graph = analyzer.create_workflow_graph(workflow)
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Node positions using spring layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Node colors based on type
        node_types = set(graph.nodes[node]['type'] for node in graph.nodes)
        color_map = {}
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, node_type in enumerate(node_types):
            color_map[node_type] = colors[i % len(colors)]
        
        node_colors = [color_map.get(graph.nodes[node]['type'], 'gray') for node in graph.nodes]
        
        # Node sizes based on connections
        node_sizes = [300 + 100 * graph.degree(node) for node in graph.nodes]
        
        # Draw the graph
        nx.draw_networkx(
            graph, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            arrows=True,
            alpha=0.8
        )
        
        # Add title
        plt.title(f"Workflow: {workflow.name}", fontsize=16)
        plt.axis('off')
        
        # Save to file
        output_path = self.output_dir / "workflows" / f"{workflow.id or 'workflow'}_diagram.{output_format}"
        plt.savefig(output_path, format=output_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _create_interactive_collection_overview(self, collection: WorkflowCollection) -> str:
        """Create an interactive overview of a workflow collection."""
        # Create a Plotly figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Workflow Size Distribution",
                "Node Type Distribution",
                "Complexity Metrics",
                "Workflow Activity"
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "domain"}]
            ]
        )
        
        # 1. Workflow Size Distribution
        workflow_sizes = [len(w.nodes) for w in collection.workflows]
        workflow_names = [w.name for w in collection.workflows]
        
        fig.add_trace(
            go.Bar(
                x=workflow_names,
                y=workflow_sizes,
                text=workflow_sizes,
                textposition='auto',
                name="Node Count"
            ),
            row=1, col=1
        )
        
        # 2. Node Type Distribution
        node_types = {}
        for workflow in collection.workflows:
            for node in workflow.nodes:
                node_type = node.type
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(node_types.keys()),
                values=list(node_types.values()),
                hole=0.3,
                name="Node Types"
            ),
            row=1, col=2
        )
        
        # 3. Complexity Metrics
        analyzer = WorkflowAnalyzer()
        complexity_metrics = []
        workflow_names = []
        
        for workflow in collection.workflows:
            result = analyzer.analyze_workflow(workflow)
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'complexity'):
                complexity = result.metrics.complexity
                complexity_metrics.append([
                    getattr(complexity, 'structural_complexity', 0),
                    getattr(complexity, 'cognitive_complexity', 0),
                    getattr(complexity, 'cyclomatic_complexity', 0)
                ])
                workflow_names.append(workflow.name)
        
        if complexity_metrics:
            complexity_metrics = np.array(complexity_metrics)
            
            fig.add_trace(
                go.Bar(
                    x=workflow_names,
                    y=complexity_metrics[:, 0],
                    name="Structural Complexity"
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=workflow_names,
                    y=complexity_metrics[:, 1],
                    name="Cognitive Complexity"
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=workflow_names,
                    y=complexity_metrics[:, 2],
                    name="Cyclomatic Complexity"
                ),
                row=2, col=1
            )
        
        # 4. Workflow Activity
        active_count = sum(1 for w in collection.workflows if w.active)
        inactive_count = collection.total_workflows - active_count
        
        fig.add_trace(
            go.Pie(
                labels=["Active", "Inactive"],
                values=[active_count, inactive_count],
                hole=0.5,
                marker=dict(colors=['#2ca02c', '#d62728'])
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Workflow Collection Overview ({collection.total_workflows} workflows)",
            height=800,
            width=1200,
            showlegend=True
        )
        
        # Save to HTML file
        output_path = self.output_dir / "collection" / "collection_overview.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def _create_static_collection_overview(self, collection: WorkflowCollection, 
                                         output_format: str) -> str:
        """Create a static overview of a workflow collection."""
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Workflow Size Distribution
        workflow_sizes = [len(w.nodes) for w in collection.workflows]
        workflow_names = [w.name for w in collection.workflows]
        
        axs[0, 0].bar(workflow_names, workflow_sizes)
        axs[0, 0].set_title("Workflow Size Distribution")
        axs[0, 0].set_ylabel("Node Count")
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Node Type Distribution
        node_types = {}
        for workflow in collection.workflows:
            for node in workflow.nodes:
                node_type = node.type.split('.')[-1]  # Use short name
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        axs[0, 1].pie(
            list(node_types.values()),
            labels=list(node_types.keys()),
            autopct='%1.1f%%',
            startangle=90
        )
        axs[0, 1].set_title("Node Type Distribution")
        
        # 3. Complexity Metrics
        analyzer = WorkflowAnalyzer()
        complexity_metrics = []
        workflow_names = []
        
        for workflow in collection.workflows:
            result = analyzer.analyze_workflow(workflow)
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'complexity'):
                complexity = result.metrics.complexity
                complexity_metrics.append([
                    getattr(complexity, 'structural_complexity', 0),
                    getattr(complexity, 'cognitive_complexity', 0),
                    getattr(complexity, 'cyclomatic_complexity', 0)
                ])
                workflow_names.append(workflow.name)
        
        if complexity_metrics:
            complexity_metrics = np.array(complexity_metrics)
            
            x = np.arange(len(workflow_names))
            width = 0.25
            
            axs[1, 0].bar(x - width, complexity_metrics[:, 0], width, label='Structural')
            axs[1, 0].bar(x, complexity_metrics[:, 1], width, label='Cognitive')
            axs[1, 0].bar(x + width, complexity_metrics[:, 2], width, label='Cyclomatic')
            
            axs[1, 0].set_title("Complexity Metrics")
            axs[1, 0].set_xticks(x)
            axs[1, 0].set_xticklabels(workflow_names, rotation=45)
            axs[1, 0].legend()
        
        # 4. Workflow Activity
        active_count = sum(1 for w in collection.workflows if w.active)
        inactive_count = collection.total_workflows - active_count
        
        axs[1, 1].pie(
            [active_count, inactive_count],
            labels=["Active", "Inactive"],
            autopct='%1.1f%%',
            colors=['#2ca02c', '#d62728'],
            startangle=90
        )
        axs[1, 1].set_title("Workflow Activity")
        
        # Adjust layout
        plt.suptitle(f"Workflow Collection Overview ({collection.total_workflows} workflows)", fontsize=16)
        plt.tight_layout()
        
        # Save to file
        output_path = self.output_dir / "collection" / f"collection_overview.{output_format}"
        plt.savefig(output_path, format=output_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _create_interactive_pattern_visualization(self, collection: WorkflowCollection,
                                                patterns: List[Any],
                                                rules: List[Any]) -> str:
        """Create an interactive visualization of pattern analysis results."""
        # Create a Plotly figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Top Frequent Patterns",
                "Association Rules Network",
                "Pattern Support Distribution",
                "Rule Metrics"
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Top Frequent Patterns
        if patterns:
            # Sort patterns by support
            sorted_patterns = sorted(patterns, key=lambda p: p.support, reverse=True)[:15]
            
            pattern_labels = [f"{', '.join(list(p.items)[:3])}" for p in sorted_patterns]
            pattern_supports = [p.support for p in sorted_patterns]
            
            fig.add_trace(
                go.Bar(
                    x=pattern_labels,
                    y=pattern_supports,
                    text=[f"{s:.2f}" for s in pattern_supports],
                    textposition='auto',
                    name="Pattern Support"
                ),
                row=1, col=1
            )
        
        # 2. Association Rules Network
        if rules:
            # Create a network of association rules
            rule_graph = nx.DiGraph()
            
            # Add top rules
            top_rules = sorted(rules, key=lambda r: r.lift, reverse=True)[:20]
            
            # Add nodes and edges
            for rule in top_rules:
                for item in rule.antecedent:
                    if item not in rule_graph:
                        rule_graph.add_node(item)
                
                for item in rule.consequent:
                    if item not in rule_graph:
                        rule_graph.add_node(item)
                
                for a_item in rule.antecedent:
                    for c_item in rule.consequent:
                        rule_graph.add_edge(
                            a_item, c_item, 
                            weight=rule.lift,
                            confidence=rule.confidence
                        )
            
            # Create network visualization
            pos = nx.spring_layout(rule_graph, seed=42)
            
            # Add edges
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in rule_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                edge_data = rule_graph.get_edge_data(edge[0], edge[1])
                edge_text.append(f"Rule: {edge[0]} -> {edge[1]}<br>"
                                f"Lift: {edge_data['weight']:.2f}<br>"
                                f"Confidence: {edge_data['confidence']:.2f}")
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='text',
                text=edge_text,
                mode='lines'
            )
            
            fig.add_trace(edge_trace, row=1, col=2)
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            
            for node in rule_graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"Item: {node}")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=False,
                    color='#1f77b4',
                    size=10,
                    line=dict(width=2, color='white')
                )
            )
            
            fig.add_trace(node_trace, row=1, col=2)
        
        # 3. Pattern Support Distribution
        if patterns:
            supports = [p.support for p in patterns]
            
            fig.add_trace(
                go.Histogram(
                    x=supports,
                    nbinsx=20,
                    name="Support Distribution"
                ),
                row=2, col=1
            )
        
        # 4. Rule Metrics
        if rules:
            # Create scatter plot of confidence vs lift
            confidences = [r.confidence for r in rules]
            lifts = [r.lift for r in rules]
            
            fig.add_trace(
                go.Scatter(
                    x=confidences,
                    y=lifts,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=lifts,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Lift")
                    ),
                    name="Rules"
                ),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Confidence", row=2, col=2)
            fig.update_yaxes(title_text="Lift", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Pattern Analysis Results",
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Save to HTML file
        output_path = self.output_dir / "patterns" / "pattern_analysis.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def _create_static_pattern_visualization(self, collection: WorkflowCollection,
                                           patterns: List[Any],
                                           rules: List[Any],
                                           output_format: str) -> str:
        """Create a static visualization of pattern analysis results."""
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Top Frequent Patterns
        if patterns:
            # Sort patterns by support
            sorted_patterns = sorted(patterns, key=lambda p: p.support, reverse=True)[:10]
            
            pattern_labels = [f"{', '.join(list(p.items)[:2])}" for p in sorted_patterns]
            pattern_supports = [p.support for p in sorted_patterns]
            
            axs[0, 0].bar(pattern_labels, pattern_supports)
            axs[0, 0].set_title("Top Frequent Patterns")
            axs[0, 0].set_ylabel("Support")
            axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Association Rules Network
        if rules:
            # Create a network of association rules
            rule_graph = nx.DiGraph()
            
            # Add top rules
            top_rules = sorted(rules, key=lambda r: r.lift, reverse=True)[:15]
            
            # Add nodes and edges
            for rule in top_rules:
                for item in rule.antecedent:
                    if item not in rule_graph:
                        rule_graph.add_node(item)
                
                for item in rule.consequent:
                    if item not in rule_graph:
                        rule_graph.add_node(item)
                
                for a_item in rule.antecedent:
                    for c_item in rule.consequent:
                        rule_graph.add_edge(
                            a_item, c_item, 
                            weight=rule.lift
                        )
            
            # Draw the graph
            pos = nx.spring_layout(rule_graph, seed=42)
            nx.draw_networkx(
                rule_graph, pos,
                ax=axs[0, 1],
                with_labels=True,
                node_color='skyblue',
                node_size=500,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                arrows=True,
                alpha=0.8
            )
            
            axs[0, 1].set_title("Association Rules Network")
            axs[0, 1].axis('off')
        
        # 3. Pattern Support Distribution
        if patterns:
            supports = [p.support for p in patterns]
            
            axs[1, 0].hist(supports, bins=20)
            axs[1, 0].set_title("Pattern Support Distribution")
            axs[1, 0].set_xlabel("Support")
            axs[1, 0].set_ylabel("Frequency")
        
        # 4. Rule Metrics
        if rules:
            # Create scatter plot of confidence vs lift
            confidences = [r.confidence for r in rules]
            lifts = [r.lift for r in rules]
            
            scatter = axs[1, 1].scatter(
                confidences, lifts,
                c=lifts, cmap='viridis',
                alpha=0.7
            )
            
            axs[1, 1].set_title("Rule Metrics")
            axs[1, 1].set_xlabel("Confidence")
            axs[1, 1].set_ylabel("Lift")
            
            plt.colorbar(scatter, ax=axs[1, 1], label="Lift")
        
        # Adjust layout
        plt.suptitle("Pattern Analysis Results", fontsize=16)
        plt.tight_layout()
        
        # Save to file
        output_path = self.output_dir / "patterns" / f"pattern_analysis.{output_format}"
        plt.savefig(output_path, format=output_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _create_interactive_network_visualization(self, graph: nx.Graph, title: str) -> str:
        """Create an interactive visualization of a network graph."""
        # Create a Plotly figure
        fig = go.Figure()
        
        # Node positions using spring layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge details
            edge_data = graph.get_edge_data(edge[0], edge[1])
            edge_text.append(f"From: {edge[0]}<br>To: {edge[1]}")
        
        # Add edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        fig.add_trace(edge_trace)
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node details
            node_data = graph.nodes[node]
            node_text.append(f"Node: {node}")
            
            # Node size based on degree
            size = 10 + 5 * graph.degree(node)
            node_size.append(size)
        
        # Add node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=[graph.degree(node) for node in graph.nodes()],
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            )
        )
        
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=title,
            title_font=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800
        )
        
        # Save to HTML file
        output_path = self.output_dir / "networks" / f"{title.replace(' ', '_').lower()}_network.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def _create_static_network_visualization(self, graph: nx.Graph, title: str, 
                                           output_format: str) -> str:
        """Create a static visualization of a network graph."""
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Node positions using spring layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Node colors based on degree
        node_degrees = dict(graph.degree())
        
        # Draw the graph
        nodes = nx.draw_networkx_nodes(
            graph, pos,
            node_size=[300 + 100 * d for d in node_degrees.values()],
            node_color=list(node_degrees.values()),
            cmap=plt.cm.YlGnBu
        )
        
        edges = nx.draw_networkx_edges(
            graph, pos,
            edge_color='gray',
            alpha=0.5
        )
        
        labels = nx.draw_networkx_labels(
            graph, pos,
            font_size=8,
            font_weight='bold'
        )
        
        # Add colorbar
        plt.colorbar(nodes, label='Node Degree')
        
        # Add title
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        # Save to file
        output_path = self.output_dir / "networks" / f"{title.replace(' ', '_').lower()}_network.{output_format}"
        plt.savefig(output_path, format=output_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _create_interactive_collection_network(self, graph: nx.Graph) -> str:
        """Create an interactive visualization of a collection network."""
        # Create a Plotly figure
        fig = go.Figure()
        
        # Node positions using spring layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge details
            edge_data = graph.get_edge_data(edge[0], edge[1])
            similarity = edge_data.get('similarity', 0)
            edge_text.append(f"Workflows: {edge[0]} - {edge[1]}<br>Similarity: {similarity:.2f}")
        
        # Add edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        fig.add_trace(edge_trace)
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node details
            node_data = graph.nodes[node]
            node_name = node_data.get('name', node)
            node_count = node_data.get('node_count', 0)
            active = node_data.get('active', False)
            tags = node_data.get('tags', [])
            
            node_text.append(f"Workflow: {node_name}<br>"
                            f"Nodes: {node_count}<br>"
                            f"Active: {'Yes' if active else 'No'}<br>"
                            f"Tags: {', '.join(tags)}")
            
            # Node size based on node count
            size = 15 + node_count
            node_size.append(size)
            
            # Node color based on active status
            node_color.append('#2ca02c' if active else '#d62728')
        
        # Add node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[graph.nodes[node].get('name', node) for node in graph.nodes()],
            textposition="top center",
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=2, color='white')
            )
        )
        
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title="Workflow Collection Network",
            title_font=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800
        )
        
        # Save to HTML file
        output_path = self.output_dir / "networks" / "collection_network.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def _create_static_collection_network(self, graph: nx.Graph, output_format: str) -> str:
        """Create a static visualization of a collection network."""
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Node positions using spring layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Node colors based on active status
        node_colors = ['#2ca02c' if graph.nodes[node].get('active', False) else '#d62728' 
                      for node in graph.nodes()]
        
        # Node sizes based on node count
        node_sizes = [300 + 50 * graph.nodes[node].get('node_count', 0) 
                     for node in graph.nodes()]
        
        # Draw the graph
        nx.draw_networkx(
            graph, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            alpha=0.8
        )
        
        # Add title
        plt.title("Workflow Collection Network", fontsize=16)
        plt.axis('off')
        
        # Save to file
        output_path = self.output_dir / "networks" / f"collection_network.{output_format}"
        plt.savefig(output_path, format=output_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _create_interactive_statistical_dashboard(self, collection: WorkflowCollection) -> str:
        """Create an interactive statistical dashboard."""
        # Create a Plotly figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Workflow Size Distribution",
                "Node Type Distribution",
                "Complexity Metrics",
                "Connection Types",
                "Error Handling Coverage",
                "Workflow Tags"
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Workflow Size Distribution
        workflow_sizes = [len(w.nodes) for w in collection.workflows]
        workflow_names = [w.name for w in collection.workflows]
        
        fig.add_trace(
            go.Bar(
                x=workflow_names,
                y=workflow_sizes,
                text=workflow_sizes,
                textposition='auto',
                name="Node Count"
            ),
            row=1, col=1
        )
        
        # 2. Node Type Distribution
        node_types = {}
        for workflow in collection.workflows:
            for node in workflow.nodes:
                node_type = node.type.split('.')[-1]  # Use short name
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(node_types.keys()),
                values=list(node_types.values()),
                hole=0.3,
                name="Node Types"
            ),
            row=1, col=2
        )
        
        # 3. Complexity Metrics
        analyzer = WorkflowAnalyzer()
        complexity_metrics = []
        workflow_names = []
        
        for workflow in collection.workflows:
            result = analyzer.analyze_workflow(workflow)
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'complexity'):
                complexity = result.metrics.complexity
                complexity_metrics.append([
                    getattr(complexity, 'structural_complexity', 0),
                    getattr(complexity, 'cognitive_complexity', 0),
                    getattr(complexity, 'cyclomatic_complexity', 0)
                ])
                workflow_names.append(workflow.name)
        
        if complexity_metrics:
            complexity_metrics = np.array(complexity_metrics)
            
            fig.add_trace(
                go.Bar(
                    x=workflow_names,
                    y=complexity_metrics[:, 0],
                    name="Structural Complexity"
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=workflow_names,
                    y=complexity_metrics[:, 1],
                    name="Cognitive Complexity"
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=workflow_names,
                    y=complexity_metrics[:, 2],
                    name="Cyclomatic Complexity"
                ),
                row=2, col=1
            )
        
        # 4. Connection Types
        connection_types = {}
        for workflow in collection.workflows:
            for conn in workflow.connections:
                conn_type = conn.source_output or "main"
                connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
        
        fig.add_trace(
            go.Bar(
                x=list(connection_types.keys()),
                y=list(connection_types.values()),
                name="Connection Types"
            ),
            row=2, col=2
        )
        
        # 5. Error Handling Coverage
        error_handling = []
        for workflow in collection.workflows:
            result = analyzer.analyze_workflow(workflow)
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'structure'):
                structure = result.metrics.structure
                error_handling.append(getattr(structure, 'error_handling_coverage', 0) * 100)
            else:
                error_handling.append(0)
        
        fig.add_trace(
            go.Bar(
                x=workflow_names,
                y=error_handling,
                name="Error Handling Coverage (%)"
            ),
            row=3, col=1
        )
        
        # 6. Workflow Tags
        tags = {}
        for workflow in collection.workflows:
            for tag in workflow.tags:
                tags[tag] = tags.get(tag, 0) + 1
        
        fig.add_trace(
            go.Bar(
                x=list(tags.keys()),
                y=list(tags.values()),
                name="Workflow Tags"
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Workflow Analysis Dashboard",
            height=1000,
            width=1200,
            showlegend=True
        )
        
        # Save to HTML file
        output_path = self.output_dir / "dashboard" / "statistical_dashboard.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def _create_static_statistical_dashboard(self, collection: WorkflowCollection, 
                                           output_format: str) -> str:
        """Create a static statistical dashboard."""
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(14, 16))
        
        # 1. Workflow Size Distribution
        workflow_sizes = [len(w.nodes) for w in collection.workflows]
        workflow_names = [w.name for w in collection.workflows]
        
        axs[0, 0].bar(workflow_names, workflow_sizes)
        axs[0, 0].set_title("Workflow Size Distribution")
        axs[0, 0].set_ylabel("Node Count")
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Node Type Distribution
        node_types = {}
        for workflow in collection.workflows:
            for node in workflow.nodes:
                node_type = node.type.split('.')[-1]  # Use short name
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        axs[0, 1].pie(
            list(node_types.values()),
            labels=list(node_types.keys()),
            autopct='%1.1f%%',
            startangle=90
        )
        axs[0, 1].set_title("Node Type Distribution")
        
        # 3. Complexity Metrics
        analyzer = WorkflowAnalyzer()
        complexity_metrics = []
        workflow_names = []
        
        for workflow in collection.workflows:
            result = analyzer.analyze_workflow(workflow)
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'complexity'):
                complexity = result.metrics.complexity
                complexity_metrics.append([
                    getattr(complexity, 'structural_complexity', 0),
                    getattr(complexity, 'cognitive_complexity', 0),
                    getattr(complexity, 'cyclomatic_complexity', 0)
                ])
                workflow_names.append(workflow.name)
        
        if complexity_metrics:
            complexity_metrics = np.array(complexity_metrics)
            
            x = np.arange(len(workflow_names))
            width = 0.25
            
            axs[1, 0].bar(x - width, complexity_metrics[:, 0], width, label='Structural')
            axs[1, 0].bar(x, complexity_metrics[:, 1], width, label='Cognitive')
            axs[1, 0].bar(x + width, complexity_metrics[:, 2], width, label='Cyclomatic')
            
            axs[1, 0].set_title("Complexity Metrics")
            axs[1, 0].set_xticks(x)
            axs[1, 0].set_xticklabels(workflow_names, rotation=45)
            axs[1, 0].legend()
        
        # 4. Connection Types
        connection_types = {}
        for workflow in collection.workflows:
            for conn in workflow.connections:
                conn_type = conn.source_output or "main"
                connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
        
        axs[1, 1].bar(list(connection_types.keys()), list(connection_types.values()))
        axs[1, 1].set_title("Connection Types")
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        # 5. Error Handling Coverage
        error_handling = []
        for workflow in collection.workflows:
            result = analyzer.analyze_workflow(workflow)
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'structure'):
                structure = result.metrics.structure
                error_handling.append(getattr(structure, 'error_handling_coverage', 0) * 100)
            else:
                error_handling.append(0)
        
        axs[2, 0].bar(workflow_names, error_handling)
        axs[2, 0].set_title("Error Handling Coverage (%)")
        axs[2, 0].tick_params(axis='x', rotation=45)
        
        # 6. Workflow Tags
        tags = {}
        for workflow in collection.workflows:
            for tag in workflow.tags:
                tags[tag] = tags.get(tag, 0) + 1
        
        axs[2, 1].bar(list(tags.keys()), list(tags.values()))
        axs[2, 1].set_title("Workflow Tags")
        axs[2, 1].tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.suptitle("Workflow Analysis Dashboard", fontsize=16)
        plt.tight_layout()
        
        # Save to file
        output_path = self.output_dir / "dashboard" / f"statistical_dashboard.{output_format}"
        plt.savefig(output_path, format=output_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(output_path)
    
    def _create_html_report(self, report_path: Path, collection: WorkflowCollection,
                          workflow_viz: Dict[str, str], collection_viz: str,
                          pattern_viz: str, network_viz: str, stats_dashboard: str) -> None:
        """Create a comprehensive HTML report."""
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>n8n Workflow Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }}
                .workflow-list {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .workflow-card {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    width: 300px;
                    background-color: white;
                }}
                .viz-container {{
                    margin-top: 20px;
                }}
                iframe {{
                    border: none;
                    width: 100%;
                    height: 600px;
                }}
                .tabs {{
                    display: flex;
                    margin-bottom: 10px;
                }}
                .tab {{
                    padding: 10px 20px;
                    background-color: #eee;
                    cursor: pointer;
                    border: 1px solid #ddd;
                    border-bottom: none;
                    border-radius: 5px 5px 0 0;
                    margin-right: 5px;
                }}
                .tab.active {{
                    background-color: #fff;
                    border-bottom: 1px solid #fff;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 0 5px 5px 5px;
                }}
                .tab-content.active {{
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>n8n Workflow Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Collection Overview</h2>
                    <p>Total workflows: {collection.total_workflows}</p>
                    <div class="viz-container">
                        <iframe src="{os.path.basename(collection_viz)}"></iframe>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Workflow Analysis</h2>
                    <div class="tabs">
        """
        
        # Add tabs for each workflow
        for i, workflow in enumerate(collection.workflows):
            workflow_id = workflow.id or workflow.name
            active_class = "active" if i == 0 else ""
            html_content += f'<div class="tab {active_class}" onclick="openTab(event, \'workflow-{i}\')">{workflow.name}</div>\n'
        
        html_content += """
                    </div>
        """
        
        # Add tab content for each workflow
        for i, workflow in enumerate(collection.workflows):
            workflow_id = workflow.id or workflow.name
            active_class = "active" if i == 0 else ""
            viz_path = os.path.basename(workflow_viz.get(workflow_id, ""))
            
            html_content += f"""
                    <div id="workflow-{i}" class="tab-content {active_class}">
                        <h3>{workflow.name}</h3>
                        <p>Nodes: {len(workflow.nodes)}, Connections: {len(workflow.connections)}</p>
                        <p>Tags: {', '.join(workflow.tags)}</p>
                        <div class="viz-container">
                            <iframe src="{viz_path}"></iframe>
                        </div>
                    </div>
            """
        
        html_content += f"""
                </div>
                
                <div class="section">
                    <h2>Pattern Analysis</h2>
                    <div class="viz-container">
                        <iframe src="{os.path.basename(pattern_viz)}"></iframe>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Network Analysis</h2>
                    <div class="viz-container">
                        <iframe src="{os.path.basename(network_viz)}"></iframe>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Statistical Dashboard</h2>
                    <div class="viz-container">
                        <iframe src="{os.path.basename(stats_dashboard)}"></iframe>
                    </div>
                </div>
            </div>
            
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    
                    // Hide all tab content
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                    }}
                    
                    // Remove active class from all tabs
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    
                    // Show the current tab and add active class
                    document.getElementById(tabName).className += " active";
                    evt.currentTarget.className += " active";
                }}
            </script>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Copy referenced files to the dashboard directory
        for viz_path in workflow_viz.values():
            if viz_path:
                src = Path(viz_path)
                dst = self.output_dir / "dashboard" / src.name
                if src.exists() and src != dst:
                    with open(src, 'rb') as src_file:
                        with open(dst, 'wb') as dst_file:
                            dst_file.write(src_file.read())
        
        for viz_path in [collection_viz, pattern_viz, network_viz, stats_dashboard]:
            if viz_path:
                src = Path(viz_path)
                dst = self.output_dir / "dashboard" / src.name
                if src.exists() and src != dst:
                    with open(src, 'rb') as src_file:
                        with open(dst, 'wb') as dst_file:
                            dst_file.write(src_file.read())

