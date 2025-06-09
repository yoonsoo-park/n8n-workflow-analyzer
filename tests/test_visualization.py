#!/usr/bin/env python3
"""
Test script for visualization functionality.

This script tests the visualization capabilities of the n8n workflow analyzer.
"""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import N8nWorkflow, WorkflowCollection
from parser import N8nWorkflowParser
from analysis.workflow_analyzer import WorkflowAnalyzer
from mining.pattern_miner import PatternMiner
from network.network_analyzer import NetworkAnalyzer
from visualization.visualization_manager import VisualizationManager
from config import get_config_manager


def test_workflow_visualization(collection, viz_manager):
    """Test workflow visualization."""
    print("\nTesting workflow visualization...")
    
    for workflow in collection.workflows:
        print(f"\nVisualizing workflow: {workflow.name}")
        
        # Create HTML visualization
        html_path = viz_manager.visualize_workflow(
            workflow, output_format="html", include_details=True
        )
        print(f"  HTML visualization saved to: {html_path}")
        
        # Create PNG visualization
        png_path = viz_manager.visualize_workflow(
            workflow, output_format="png", include_details=True
        )
        print(f"  PNG visualization saved to: {png_path}")
    
    return True


def test_collection_visualization(collection, viz_manager):
    """Test collection visualization."""
    print("\nTesting collection visualization...")
    
    # Create HTML visualization
    html_path = viz_manager.visualize_workflow_collection(
        collection, output_format="html"
    )
    print(f"HTML collection overview saved to: {html_path}")
    
    # Create PNG visualization
    png_path = viz_manager.visualize_workflow_collection(
        collection, output_format="png"
    )
    print(f"PNG collection overview saved to: {png_path}")
    
    return True


def test_pattern_visualization(collection, viz_manager):
    """Test pattern visualization."""
    print("\nTesting pattern visualization...")
    
    # Run pattern mining
    miner = PatternMiner()
    patterns = miner.mine_frequent_patterns(collection)
    rules = miner.generate_association_rules(patterns)
    
    print(f"Found {len(patterns)} patterns and {len(rules)} rules")
    
    # Create HTML visualization
    html_path = viz_manager.visualize_pattern_analysis(
        collection, patterns, rules, output_format="html"
    )
    print(f"HTML pattern analysis saved to: {html_path}")
    
    # Create PNG visualization
    png_path = viz_manager.visualize_pattern_analysis(
        collection, patterns, rules, output_format="png"
    )
    print(f"PNG pattern analysis saved to: {png_path}")
    
    return True


def test_network_visualization(collection, viz_manager):
    """Test network visualization."""
    print("\nTesting network visualization...")
    
    # Create HTML visualization for each workflow
    for workflow in collection.workflows:
        print(f"\nVisualizing network for workflow: {workflow.name}")
        
        html_path = viz_manager.visualize_network_analysis(
            collection, workflow_id=workflow.id or workflow.name, output_format="html"
        )
        print(f"  HTML network visualization saved to: {html_path}")
    
    # Create collection network visualization
    html_path = viz_manager.visualize_network_analysis(
        collection, output_format="html"
    )
    print(f"HTML collection network saved to: {html_path}")
    
    png_path = viz_manager.visualize_network_analysis(
        collection, output_format="png"
    )
    print(f"PNG collection network saved to: {png_path}")
    
    return True


def test_statistical_dashboard(collection, viz_manager):
    """Test statistical dashboard creation."""
    print("\nTesting statistical dashboard...")
    
    # Create HTML dashboard
    html_path = viz_manager.create_statistical_dashboard(
        collection, output_format="html"
    )
    print(f"HTML statistical dashboard saved to: {html_path}")
    
    # Create PNG dashboard
    png_path = viz_manager.create_statistical_dashboard(
        collection, output_format="png"
    )
    print(f"PNG statistical dashboard saved to: {png_path}")
    
    return True


def test_comprehensive_report(collection, viz_manager):
    """Test comprehensive report creation."""
    print("\nTesting comprehensive report...")
    
    # Create comprehensive report
    report_path = viz_manager.create_comprehensive_report(collection)
    print(f"Comprehensive report saved to: {report_path}")
    
    return True


def test_export_visualizations(collection, viz_manager):
    """Test exporting visualizations in multiple formats."""
    print("\nTesting visualization export...")
    
    # Export visualizations
    exports = viz_manager.export_visualizations(
        collection, formats=["html", "png"]
    )
    
    for category, paths in exports.items():
        print(f"  {category}: {len(paths)} files")
    
    return exports


def main():
    """Main test function."""
    print("n8n Workflow Visualization Test Suite")
    print("=" * 45)
    
    try:
        # Initialize configuration
        config_manager = get_config_manager()
        print(f"Configuration loaded successfully")
        
        # Parse workflows
        parser = N8nWorkflowParser()
        sample_dir = Path("data/sample_workflows")
        collection = parser.parse_workflow_collection(sample_dir)
        
        if collection.total_workflows == 0:
            print("No workflows found to visualize!")
            return False
        
        print(f"Loaded {collection.total_workflows} workflows for visualization")
        
        # Create visualization manager
        viz_manager = VisualizationManager()
        
        # Test visualization components
        test_workflow_visualization(collection, viz_manager)
        test_collection_visualization(collection, viz_manager)
        test_pattern_visualization(collection, viz_manager)
        test_network_visualization(collection, viz_manager)
        test_statistical_dashboard(collection, viz_manager)
        test_comprehensive_report(collection, viz_manager)
        test_export_visualizations(collection, viz_manager)
        
        print("\nAll visualization tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

