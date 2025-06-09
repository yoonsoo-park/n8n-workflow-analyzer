from flask import Blueprint, jsonify, request, current_app
import os
import json
import sys
import tempfile
from pathlib import Path
import uuid

# Add the parent directory to the path to import our analyzer modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.parser import N8nWorkflowParser
from src.models import WorkflowCollection
from src.analysis.workflow_analyzer import WorkflowAnalyzer
from src.mining.pattern_miner import PatternMiner
from src.network.network_analyzer import NetworkAnalyzer
from src.visualization.visualization_manager import VisualizationManager

workflow_bp = Blueprint('workflow', __name__)

# Directory to store uploaded workflows
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
VISUALIZATION_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Store analysis results in memory for quick retrieval
analysis_results = {}

@workflow_bp.route('/upload', methods=['POST'])
def upload_workflow():
    """
    Upload n8n workflow files.
    
    Accepts multiple workflow files in JSON format.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Create a unique session ID for this upload
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    saved_files = []
    
    for file in files:
        if file and file.filename.endswith('.json'):
            filename = os.path.join(session_folder, file.filename)
            file.save(filename)
            saved_files.append(filename)
    
    if not saved_files:
        return jsonify({'error': 'No valid JSON files uploaded'}), 400
    
    return jsonify({
        'message': f'Successfully uploaded {len(saved_files)} workflow files',
        'session_id': session_id,
        'file_count': len(saved_files)
    }), 201

@workflow_bp.route('/analyze/<session_id>', methods=['GET'])
def analyze_workflows(session_id):
    """
    Analyze uploaded workflows for a given session.
    
    Performs comprehensive analysis including structure, patterns, and network analysis.
    """
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    
    if not os.path.exists(session_folder):
        return jsonify({'error': 'Session not found'}), 404
    
    # Parse workflows
    parser = N8nWorkflowParser()
    collection = parser.parse_workflow_collection(session_folder)
    
    if collection.total_workflows == 0:
        return jsonify({'error': 'No valid workflows found in session'}), 400
    
    # Perform analysis
    analyzer = WorkflowAnalyzer()
    miner = PatternMiner()
    network_analyzer = NetworkAnalyzer()
    
    # Structure analysis
    workflow_analyses = {}
    for workflow in collection.workflows:
        analysis_result = analyzer.analyze_workflow(workflow)
        workflow_analyses[workflow.id or workflow.name] = {
            'id': workflow.id,
            'name': workflow.name,
            'node_count': len(workflow.nodes),
            'connection_count': len(workflow.connections),
            'metrics': {
                'complexity': {
                    'structural_complexity': getattr(analysis_result.metrics.complexity, 'structural_complexity', 0),
                    'cognitive_complexity': getattr(analysis_result.metrics.complexity, 'cognitive_complexity', 0),
                    'cyclomatic_complexity': getattr(analysis_result.metrics.complexity, 'cyclomatic_complexity', 0)
                },
                'structure': {
                    'error_handling_coverage': getattr(analysis_result.metrics.structure, 'error_handling_coverage', 0),
                    'branching_factor': getattr(analysis_result.metrics.structure, 'branching_factor', 0)
                }
            }
        }
    
    # Pattern mining (limit to avoid performance issues)
    patterns = miner.mine_frequent_patterns(collection, min_support=0.3, max_itemsets=100)
    rules = miner.generate_association_rules(patterns, min_confidence=0.5, max_rules=100)
    
    pattern_results = {
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
    
    # Network analysis
    collection_graph = network_analyzer.create_collection_graph(collection)
    network_metrics = network_analyzer.calculate_network_metrics(collection_graph)
    
    network_results = {
        'node_count': network_metrics.get('node_count', 0),
        'edge_count': network_metrics.get('edge_count', 0),
        'density': network_metrics.get('density', 0),
        'average_degree': network_metrics.get('average_degree', 0),
        'diameter': network_metrics.get('diameter', 0),
        'average_path_length': network_metrics.get('average_path_length', 0),
        'clustering_coefficient': network_metrics.get('clustering_coefficient', 0)
    }
    
    # Generate visualizations
    viz_manager = VisualizationManager(output_dir=os.path.join(VISUALIZATION_FOLDER, session_id))
    
    visualization_paths = {}
    for workflow in collection.workflows:
        workflow_id = workflow.id or workflow.name
        html_path = viz_manager.visualize_workflow(workflow, output_format="html")
        visualization_paths[workflow_id] = {
            'workflow_diagram': os.path.basename(html_path)
        }
    
    collection_viz = viz_manager.visualize_workflow_collection(collection, output_format="html")
    pattern_viz = viz_manager.visualize_pattern_analysis(collection, patterns, rules, output_format="html")
    network_viz = viz_manager.visualize_network_analysis(collection, output_format="html")
    dashboard = viz_manager.create_statistical_dashboard(collection, output_format="html")
    report = viz_manager.create_comprehensive_report(collection)
    
    visualization_results = {
        'workflow_diagrams': visualization_paths,
        'collection_overview': os.path.basename(collection_viz),
        'pattern_analysis': os.path.basename(pattern_viz),
        'network_analysis': os.path.basename(network_viz),
        'statistical_dashboard': os.path.basename(dashboard),
        'comprehensive_report': os.path.basename(report),
        'base_path': f'/api/visualizations/{session_id}'
    }
    
    # Combine all results
    result = {
        'session_id': session_id,
        'workflow_count': collection.total_workflows,
        'workflow_analyses': workflow_analyses,
        'pattern_analysis': pattern_results,
        'network_analysis': network_results,
        'visualizations': visualization_results
    }
    
    # Store results for later retrieval
    analysis_results[session_id] = result
    
    return jsonify(result), 200

@workflow_bp.route('/results/<session_id>', methods=['GET'])
def get_analysis_results(session_id):
    """
    Retrieve previously generated analysis results.
    """
    if session_id not in analysis_results:
        return jsonify({'error': 'Analysis results not found for this session'}), 404
    
    return jsonify(analysis_results[session_id]), 200

@workflow_bp.route('/visualizations/<session_id>/<path:filename>', methods=['GET'])
def get_visualization(session_id, filename):
    """
    Serve visualization files.
    """
    session_folder = os.path.join(VISUALIZATION_FOLDER, session_id)
    
    # Find the file in any subdirectory
    for root, dirs, files in os.walk(session_folder):
        if filename in files:
            return send_from_directory(root, filename)
    
    return jsonify({'error': 'Visualization file not found'}), 404

@workflow_bp.route('/sessions', methods=['GET'])
def list_sessions():
    """
    List all available analysis sessions.
    """
    sessions = []
    
    for session_id in os.listdir(UPLOAD_FOLDER):
        session_path = os.path.join(UPLOAD_FOLDER, session_id)
        if os.path.isdir(session_path):
            file_count = len([f for f in os.listdir(session_path) if f.endswith('.json')])
            has_analysis = session_id in analysis_results
            
            sessions.append({
                'session_id': session_id,
                'file_count': file_count,
                'has_analysis': has_analysis,
                'created_at': os.path.getctime(session_path)
            })
    
    return jsonify(sessions), 200

@workflow_bp.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """
    Delete a session and all its files.
    """
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    viz_folder = os.path.join(VISUALIZATION_FOLDER, session_id)
    
    if not os.path.exists(session_folder):
        return jsonify({'error': 'Session not found'}), 404
    
    # Delete uploaded files
    for file in os.listdir(session_folder):
        os.remove(os.path.join(session_folder, file))
    os.rmdir(session_folder)
    
    # Delete visualizations if they exist
    if os.path.exists(viz_folder):
        for root, dirs, files in os.walk(viz_folder, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(viz_folder)
    
    # Remove from in-memory results
    if session_id in analysis_results:
        del analysis_results[session_id]
    
    return jsonify({'message': 'Session deleted successfully'}), 200

