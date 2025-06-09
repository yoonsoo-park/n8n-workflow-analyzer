#!/usr/bin/env python3
"""
Comprehensive test suite for the n8n workflow analyzer.

This script tests all components of the n8n workflow analyzer system:
1. Data models and parsing
2. Workflow analysis
3. Pattern mining
4. Network analysis
5. Visualization generation

Usage:
    python3 test_suite.py
"""

import os
import sys
import json
import logging
import unittest
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import (
    N8nWorkflow, WorkflowNode, WorkflowConnection, 
    WorkflowSettings, WorkflowCollection
)
from src.parser import N8nWorkflowParser
from src.analysis.workflow_analyzer import WorkflowAnalyzer
from src.mining.pattern_miner import PatternMiner
from src.network.network_analyzer import NetworkAnalyzer
from src.visualization.visualization_manager import VisualizationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_suite')

class TestN8nWorkflowAnalyzer(unittest.TestCase):
    """Test suite for the n8n workflow analyzer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.data_dir = Path('data/sample_workflows')
        cls.output_dir = Path('data/test_output')
        cls.reports_dir = Path('data/test_reports')
        
        # Create directories if they don't exist
        cls.output_dir.mkdir(exist_ok=True, parents=True)
        cls.reports_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        cls.parser = N8nWorkflowParser()
        cls.analyzer = WorkflowAnalyzer()
        cls.miner = PatternMiner()
        cls.network_analyzer = NetworkAnalyzer()
        cls.viz_manager = VisualizationManager(output_dir=str(cls.output_dir))
        
        # Parse workflows
        cls.collection = cls.parser.parse_workflow_collection(str(cls.data_dir))
        
        logger.info(f"Loaded {cls.collection.total_workflows} workflows for testing")
    
    def test_01_data_models(self):
        """Test data models and parsing."""
        logger.info("Testing data models and parsing...")
        
        # Test collection
        self.assertIsNotNone(self.collection)
        self.assertGreater(self.collection.total_workflows, 0)
        
        # Test workflows
        for workflow in self.collection.workflows:
            self.assertIsInstance(workflow, N8nWorkflow)
            self.assertIsNotNone(workflow.id)
            self.assertIsNotNone(workflow.name)
            
            # Test nodes
            self.assertGreater(len(workflow.nodes), 0)
            for node in workflow.nodes:
                self.assertIsInstance(node, WorkflowNode)
                self.assertIsNotNone(node.id)
                self.assertIsNotNone(node.type)
                
            # Test connections
            for connection in workflow.connections:
                self.assertIsInstance(connection, WorkflowConnection)
                self.assertIsNotNone(connection.source_node)
                self.assertIsNotNone(connection.target_node)
            
            # Test settings
            self.assertIsInstance(workflow.settings, WorkflowSettings)
        
        logger.info("Data models and parsing tests passed")
    
    def test_02_workflow_analysis(self):
        """Test workflow analysis functionality."""
        logger.info("Testing workflow analysis...")
        
        for workflow in self.collection.workflows:
            # Analyze workflow
            result = self.analyzer.analyze_workflow(workflow)
            
            # Test analysis result
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.metrics)
            self.assertIsNotNone(result.metrics.complexity)
            self.assertIsNotNone(result.metrics.structure)
            
            # Test complexity metrics
            self.assertGreaterEqual(result.metrics.complexity.structural_complexity, 0)
            self.assertGreaterEqual(result.metrics.complexity.cognitive_complexity, 0)
            self.assertGreaterEqual(result.metrics.complexity.cyclomatic_complexity, 0)
            
            # Test structure metrics
            self.assertGreaterEqual(result.metrics.structure.error_handling_coverage, 0)
            self.assertLessEqual(result.metrics.structure.error_handling_coverage, 1)
            self.assertGreaterEqual(result.metrics.structure.branching_factor, 0)
            
            # Save analysis result
            output_path = self.reports_dir / f"{workflow.id}_analysis.json"
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        logger.info("Workflow analysis tests passed")
    
    def test_03_pattern_mining(self):
        """Test pattern mining functionality."""
        logger.info("Testing pattern mining...")
        
        # Mine patterns
        patterns = self.miner.mine_frequent_patterns(self.collection)
        
        # Test patterns
        self.assertIsNotNone(patterns)
        self.assertGreater(len(patterns), 0)
        
        # Generate rules
        rules = self.miner.generate_association_rules(patterns)
        
        # Test rules
        self.assertIsNotNone(rules)
        
        # Save pattern mining results
        output_path = self.reports_dir / "pattern_mining_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                'pattern_count': len(patterns),
                'rule_count': len(rules),
                'top_patterns': [
                    {
                        'items': list(p.items),
                        'support': p.support
                    } for p in sorted(patterns, key=lambda x: x.support, reverse=True)[:10]
                ],
                'top_rules': [
                    {
                        'antecedent': list(r.antecedent),
                        'consequent': list(r.consequent),
                        'confidence': r.confidence,
                        'lift': r.lift
                    } for r in sorted(rules, key=lambda x: x.lift, reverse=True)[:10]
                ]
            }, f, indent=2)
        
        logger.info(f"Pattern mining tests passed - Found {len(patterns)} patterns and {len(rules)} rules")
    
    def test_04_network_analysis(self):
        """Test network analysis functionality."""
        logger.info("Testing network analysis...")
        
        # Create collection graph
        graph = self.network_analyzer.create_collection_graph(self.collection)
        
        # Test graph
        self.assertIsNotNone(graph)
        
        # Calculate network metrics
        metrics = self.network_analyzer.calculate_network_metrics(graph)
        
        # Test metrics
        self.assertIsNotNone(metrics)
        self.assertIn('node_count', metrics)
        self.assertIn('edge_count', metrics)
        self.assertIn('density', metrics)
        self.assertIn('average_degree', metrics)
        
        # Save network analysis results
        output_path = self.reports_dir / "network_analysis_results.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Network analysis tests passed")
    
    def test_05_visualization(self):
        """Test visualization functionality."""
        logger.info("Testing visualization generation...")
        
        # Test workflow visualization
        for workflow in self.collection.workflows:
            html_path = self.viz_manager.visualize_workflow(workflow, output_format="html")
            self.assertTrue(os.path.exists(html_path))
            
            png_path = self.viz_manager.visualize_workflow(workflow, output_format="png")
            self.assertTrue(os.path.exists(png_path))
        
        # Test collection visualization
        collection_viz = self.viz_manager.visualize_workflow_collection(self.collection)
        self.assertTrue(os.path.exists(collection_viz))
        
        # Test pattern visualization
        patterns = self.miner.mine_frequent_patterns(self.collection, min_support=0.3, max_itemsets=100)
        rules = self.miner.generate_association_rules(patterns, min_confidence=0.5, max_rules=100)
        
        pattern_viz = self.viz_manager.visualize_pattern_analysis(self.collection, patterns, rules)
        self.assertTrue(os.path.exists(pattern_viz))
        
        # Test network visualization
        graph = self.network_analyzer.create_collection_graph(self.collection)
        network_viz = self.viz_manager.visualize_network_analysis(self.collection)
        self.assertTrue(os.path.exists(network_viz))
        
        # Test dashboard
        dashboard = self.viz_manager.create_statistical_dashboard(self.collection)
        self.assertTrue(os.path.exists(dashboard))
        
        # Test report
        report = self.viz_manager.create_comprehensive_report(self.collection)
        self.assertTrue(os.path.exists(report))
        
        logger.info("Visualization tests passed")
    
    def test_06_end_to_end(self):
        """Test end-to-end functionality."""
        logger.info("Testing end-to-end functionality...")
        
        # Create a comprehensive analysis report
        report_data = {
            'collection': {
                'total_workflows': self.collection.total_workflows,
                'workflows': [
                    {
                        'id': workflow.id,
                        'name': workflow.name,
                        'node_count': len(workflow.nodes),
                        'connection_count': len(workflow.connections)
                    } for workflow in self.collection.workflows
                ]
            },
            'analysis': {},
            'patterns': {},
            'network': {},
            'visualizations': {}
        }
        
        # Add analysis results
        for workflow in self.collection.workflows:
            result = self.analyzer.analyze_workflow(workflow)
            report_data['analysis'][workflow.id] = result.to_dict()
        
        # Add pattern mining results
        patterns = self.miner.mine_frequent_patterns(self.collection, min_support=0.3, max_itemsets=100)
        rules = self.miner.generate_association_rules(patterns, min_confidence=0.5, max_rules=100)
        
        report_data['patterns'] = {
            'pattern_count': len(patterns),
            'rule_count': len(rules),
            'top_patterns': [
                {
                    'items': list(p.items),
                    'support': p.support
                } for p in sorted(patterns, key=lambda x: x.support, reverse=True)[:10]
            ],
            'top_rules': [
                {
                    'antecedent': list(r.antecedent),
                    'consequent': list(r.consequent),
                    'confidence': r.confidence,
                    'lift': r.lift
                } for r in sorted(rules, key=lambda x: x.lift, reverse=True)[:10]
            ]
        }
        
        # Add network analysis results
        graph = self.network_analyzer.create_collection_graph(self.collection)
        metrics = self.network_analyzer.calculate_network_metrics(graph)
        report_data['network'] = metrics
        
        # Add visualization paths
        report_data['visualizations'] = {
            'workflow_diagrams': {
                workflow.id: {
                    'html': str(self.output_dir / "workflows" / f"{workflow.id}_diagram.html"),
                    'png': str(self.output_dir / "workflows" / f"{workflow.id}_diagram.png")
                } for workflow in self.collection.workflows
            },
            'collection_overview': str(self.output_dir / "collection" / "collection_overview.html"),
            'pattern_analysis': str(self.output_dir / "patterns" / "pattern_analysis.html"),
            'network_analysis': str(self.output_dir / "networks" / "network_analysis.html"),
            'dashboard': str(self.output_dir / "dashboard" / "statistical_dashboard.html"),
            'report': str(self.output_dir / "comprehensive_report.html")
        }
        
        # Save comprehensive report
        output_path = self.reports_dir / "comprehensive_analysis_report.json"
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info("End-to-end tests passed")
        logger.info(f"Comprehensive report saved to {output_path}")

if __name__ == '__main__':
    unittest.main()

