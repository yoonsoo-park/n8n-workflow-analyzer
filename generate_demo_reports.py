#!/usr/bin/env python3
"""
Generate sample analysis reports for demonstration purposes.

This script creates a set of sample n8n workflows and generates comprehensive
analysis reports to demonstrate the capabilities of the n8n workflow analyzer.

Usage:
    python3 generate_demo_reports.py
"""

import os
import sys
import json
import logging
import random
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
logger = logging.getLogger('demo_generator')

# Node types for sample workflows
NODE_TYPES = [
    "n8n-nodes-base.httpRequest",
    "n8n-nodes-base.set",
    "n8n-nodes-base.if",
    "n8n-nodes-base.switch",
    "n8n-nodes-base.function",
    "n8n-nodes-base.merge",
    "n8n-nodes-base.spreadsheetFile",
    "n8n-nodes-base.emailSend",
    "n8n-nodes-base.slack",
    "n8n-nodes-base.googleSheets",
    "n8n-nodes-base.webhook",
    "n8n-nodes-base.cron",
    "n8n-nodes-base.noOp",
    "n8n-nodes-base.wait",
    "n8n-nodes-base.filter",
    "n8n-nodes-base.code",
    "n8n-nodes-base.itemLists",
    "n8n-nodes-base.readPDF",
    "n8n-nodes-base.writePDF",
    "n8n-nodes-base.executeCommand",
    "n8n-nodes-base.httpRequest",
    "n8n-nodes-base.redis",
    "n8n-nodes-base.mongodb",
    "n8n-nodes-base.mysql",
    "n8n-nodes-base.postgres",
    "n8n-nodes-base.aws",
    "n8n-nodes-base.googleDrive",
    "n8n-nodes-base.twitter",
    "n8n-nodes-base.telegram"
]

# Workflow templates
WORKFLOW_TEMPLATES = [
    {
        "id": "demo_workflow_001",
        "name": "Customer Data Processing",
        "description": "Process customer data from multiple sources",
        "node_count": 12,
        "complexity": "medium"
    },
    {
        "id": "demo_workflow_002",
        "name": "Social Media Monitoring",
        "description": "Monitor social media platforms for brand mentions",
        "node_count": 8,
        "complexity": "low"
    },
    {
        "id": "demo_workflow_003",
        "name": "E-commerce Order Processing",
        "description": "Process and fulfill e-commerce orders",
        "node_count": 15,
        "complexity": "high"
    },
    {
        "id": "demo_workflow_004",
        "name": "Data Backup Automation",
        "description": "Automated backup of critical data",
        "node_count": 6,
        "complexity": "low"
    },
    {
        "id": "demo_workflow_005",
        "name": "Marketing Campaign Automation",
        "description": "Automate multi-channel marketing campaigns",
        "node_count": 18,
        "complexity": "high"
    },
    {
        "id": "demo_workflow_006",
        "name": "Invoice Processing",
        "description": "Process and archive invoices",
        "node_count": 10,
        "complexity": "medium"
    },
    {
        "id": "demo_workflow_007",
        "name": "HR Onboarding",
        "description": "Automate employee onboarding process",
        "node_count": 14,
        "complexity": "medium"
    },
    {
        "id": "demo_workflow_008",
        "name": "Inventory Management",
        "description": "Track and manage inventory levels",
        "node_count": 9,
        "complexity": "medium"
    },
    {
        "id": "demo_workflow_009",
        "name": "Customer Support Ticket Routing",
        "description": "Route support tickets to appropriate teams",
        "node_count": 11,
        "complexity": "medium"
    },
    {
        "id": "demo_workflow_010",
        "name": "Website Monitoring",
        "description": "Monitor website uptime and performance",
        "node_count": 7,
        "complexity": "low"
    }
]

def generate_node_position(index, total_nodes, complexity):
    """Generate a node position based on index and complexity."""
    if complexity == "low":
        # More linear layout
        return {
            "x": 100 + (index * 200),
            "y": 100 + random.randint(-50, 50)
        }
    elif complexity == "medium":
        # Moderate branching
        if index % 3 == 0:
            return {
                "x": 100 + ((index // 3) * 300),
                "y": 100
            }
        elif index % 3 == 1:
            return {
                "x": 100 + ((index // 3) * 300),
                "y": 250
            }
        else:
            return {
                "x": 100 + ((index // 3) * 300),
                "y": 400
            }
    else:  # high
        # Complex layout with multiple branches
        row = index % 4
        col = index // 4
        return {
            "x": 100 + (col * 200),
            "y": 100 + (row * 150) + random.randint(-30, 30)
        }

def generate_sample_workflow(template):
    """Generate a sample workflow based on a template."""
    workflow_id = template["id"]
    workflow_name = template["name"]
    node_count = template["node_count"]
    complexity = template["complexity"]
    
    # Create nodes
    nodes = []
    for i in range(node_count):
        node_type = random.choice(NODE_TYPES)
        node_name = f"{node_type.split('.')[-1]}_{i+1}"
        
        node = WorkflowNode(
            id=f"node_{i+1}",
            name=node_name,
            type=node_type,
            position=generate_node_position(i, node_count, complexity),
            parameters={
                "operation": random.choice(["create", "read", "update", "delete"]),
                "resource": random.choice(["user", "product", "order", "invoice"]),
                "options": {
                    "limit": random.randint(10, 100)
                }
            }
        )
        nodes.append(node)
    
    # Create connections based on complexity
    connections = []
    if complexity == "low":
        # Simple linear connections
        for i in range(node_count - 1):
            connection = WorkflowConnection(
                source_node=f"node_{i+1}",
                target_node=f"node_{i+2}"
            )
            connections.append(connection)
    elif complexity == "medium":
        # Moderate branching
        for i in range(node_count - 1):
            if i % 3 == 0 and i + 2 < node_count:
                # Branch out
                connection1 = WorkflowConnection(
                    source_node=f"node_{i+1}",
                    target_node=f"node_{i+2}"
                )
                connection2 = WorkflowConnection(
                    source_node=f"node_{i+1}",
                    target_node=f"node_{i+3}"
                )
                connections.append(connection1)
                connections.append(connection2)
            elif i % 3 != 0:
                # Continue linear path
                if i + 1 < node_count:
                    connection = WorkflowConnection(
                        source_node=f"node_{i+1}",
                        target_node=f"node_{i+2}"
                    )
                    connections.append(connection)
    else:  # high
        # Complex connections with multiple branches and merges
        for i in range(node_count):
            # Each node connects to 1-3 other nodes
            targets = random.sample(
                range(1, node_count + 1),
                min(random.randint(1, 3), node_count - i - 1)
            )
            for target in targets:
                if target > i + 1:  # Avoid circular references
                    connection = WorkflowConnection(
                        source_node=f"node_{i+1}",
                        target_node=f"node_{target}"
                    )
                    connections.append(connection)
    
    # Create workflow settings
    settings = WorkflowSettings(
        execution_order="sequential",
        timezone="UTC",
        save_execution_progress=True,
        save_manual_executions=True
    )
    
    # Create workflow
    workflow = N8nWorkflow(
        id=workflow_id,
        name=workflow_name,
        nodes=nodes,
        connections=connections,
        settings=settings,
        version=1,
        active=True
    )
    
    return workflow

def generate_sample_workflows():
    """Generate sample workflows based on templates."""
    workflows = []
    for template in WORKFLOW_TEMPLATES:
        workflow = generate_sample_workflow(template)
        workflows.append(workflow)
    
    return workflows

def save_workflow_json(workflow, output_dir):
    """Save workflow as JSON file."""
    output_path = output_dir / f"{workflow.id}.json"
    
    # Convert workflow to dictionary
    workflow_dict = {
        "id": workflow.id,
        "name": workflow.name,
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "position": node.position,
                "parameters": node.parameters
            } for node in workflow.nodes
        ],
        "connections": [
            {
                "source": connection.source_node,
                "target": connection.target_node
            } for connection in workflow.connections
        ],
        "settings": {
            "executionOrder": workflow.settings.execution_order,
            "timezone": workflow.settings.timezone,
            "saveExecutionProgress": workflow.settings.save_execution_progress,
            "saveManualExecutions": workflow.settings.save_manual_executions
        },
        "version": workflow.version,
        "active": workflow.active
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(workflow_dict, f, indent=2)
    
    return output_path

def main():
    """Main function to generate sample workflows and analysis reports."""
    # Create directories
    demo_dir = Path('data/demo')
    workflows_dir = demo_dir / 'workflows'
    reports_dir = demo_dir / 'reports'
    visualizations_dir = demo_dir / 'visualizations'
    
    workflows_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    visualizations_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("Generating sample workflows...")
    
    # Generate sample workflows
    workflows = generate_sample_workflows()
    
    # Save workflows as JSON
    for workflow in workflows:
        output_path = save_workflow_json(workflow, workflows_dir)
        logger.info(f"Saved workflow {workflow.name} to {output_path}")
    
    # Create workflow collection
    collection = WorkflowCollection(workflows=workflows)
    
    logger.info(f"Generated {len(workflows)} sample workflows")
    
    # Initialize analysis components
    analyzer = WorkflowAnalyzer()
    miner = PatternMiner()
    network_analyzer = NetworkAnalyzer()
    viz_manager = VisualizationManager(output_dir=str(visualizations_dir))
    
    logger.info("Analyzing workflows...")
    
    # Analyze workflows
    analysis_results = {}
    for workflow in workflows:
        result = analyzer.analyze_workflow(workflow)
        analysis_results[workflow.id] = result
        
        # Save analysis result
        output_path = reports_dir / f"{workflow.id}_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Analyzed workflow {workflow.name}")
    
    logger.info("Mining patterns...")
    
    # Mine patterns
    patterns = miner.mine_frequent_patterns(collection)
    rules = miner.generate_association_rules(patterns)
    
    # Save pattern mining results
    output_path = reports_dir / "pattern_mining_results.json"
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
    
    logger.info(f"Found {len(patterns)} patterns and {len(rules)} rules")
    
    logger.info("Analyzing network properties...")
    
    # Analyze network
    graph = network_analyzer.create_collection_graph(collection)
    metrics = network_analyzer.calculate_network_metrics(graph)
    
    # Save network analysis results
    output_path = reports_dir / "network_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Generating visualizations...")
    
    # Generate visualizations
    for workflow in workflows:
        html_path = viz_manager.visualize_workflow(workflow, output_format="html")
        png_path = viz_manager.visualize_workflow(workflow, output_format="png")
        logger.info(f"Generated visualizations for workflow {workflow.name}")
    
    collection_viz = viz_manager.visualize_workflow_collection(collection)
    pattern_viz = viz_manager.visualize_pattern_analysis(collection, patterns, rules)
    network_viz = viz_manager.visualize_network_analysis(collection)
    dashboard = viz_manager.create_statistical_dashboard(collection)
    report = viz_manager.create_comprehensive_report(collection)
    
    logger.info("Creating comprehensive report...")
    
    # Create comprehensive report
    report_data = {
        'collection': {
            'total_workflows': collection.total_workflows,
            'workflows': [
                {
                    'id': workflow.id,
                    'name': workflow.name,
                    'node_count': len(workflow.nodes),
                    'connection_count': len(workflow.connections)
                } for workflow in collection.workflows
            ]
        },
        'analysis': {
            workflow.id: analysis_results[workflow.id].to_dict()
            for workflow in workflows
        },
        'patterns': {
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
        },
        'network': metrics,
        'visualizations': {
            'workflow_diagrams': {
                workflow.id: {
                    'html': str(visualizations_dir / "workflows" / f"{workflow.id}_diagram.html"),
                    'png': str(visualizations_dir / "workflows" / f"{workflow.id}_diagram.png")
                } for workflow in workflows
            },
            'collection_overview': str(visualizations_dir / "collection" / "collection_overview.html"),
            'pattern_analysis': str(visualizations_dir / "patterns" / "pattern_analysis.html"),
            'network_analysis': str(visualizations_dir / "networks" / "network_analysis.html"),
            'dashboard': str(visualizations_dir / "dashboard" / "statistical_dashboard.html"),
            'report': str(visualizations_dir / "comprehensive_report.html")
        }
    }
    
    # Save comprehensive report
    output_path = reports_dir / "comprehensive_analysis_report.json"
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Comprehensive report saved to {output_path}")
    logger.info("Demo generation complete!")

if __name__ == '__main__':
    main()

