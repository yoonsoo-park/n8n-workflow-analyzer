"""
Workflow analysis engine for n8n workflows.

This module provides comprehensive analysis capabilities for n8n workflows,
including structure analysis, complexity metrics, and pattern detection.
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass, field

try:
    from ..models import N8nWorkflow, WorkflowNode, WorkflowConnection, WorkflowCollection, WorkflowAnalysisResult
    from ..config import get_analysis_config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import N8nWorkflow, WorkflowNode, WorkflowConnection, WorkflowCollection, WorkflowAnalysisResult
    from config import get_analysis_config


logger = logging.getLogger(__name__)


@dataclass
class WorkflowStructureMetrics:
    """Container for workflow structure metrics."""
    
    node_count: int = 0
    connection_count: int = 0
    max_depth: int = 0
    max_width: int = 0
    branching_factor: float = 0.0
    convergence_factor: float = 0.0
    cyclic: bool = False
    connected_components: int = 1
    
    # Node type statistics
    node_type_distribution: Dict[str, int] = field(default_factory=dict)
    unique_node_types: int = 0
    
    # Connection patterns
    avg_connections_per_node: float = 0.0
    max_outgoing_connections: int = 0
    max_incoming_connections: int = 0
    
    # Error handling
    has_error_handling: bool = False
    error_handling_coverage: float = 0.0


@dataclass
class ComplexityMetrics:
    """Container for workflow complexity metrics."""
    
    # Basic complexity
    structural_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    cyclomatic_complexity: int = 0
    
    # Advanced metrics
    halstead_volume: float = 0.0
    maintainability_index: float = 0.0
    technical_debt_ratio: float = 0.0
    
    # Readability metrics
    naming_consistency_score: float = 0.0
    documentation_coverage: float = 0.0


class WorkflowAnalyzer:
    """Main workflow analysis engine."""
    
    def __init__(self):
        """Initialize the workflow analyzer."""
        self.config = get_analysis_config()
        self.logger = logging.getLogger(__name__)
    
    def analyze_workflow(self, workflow: N8nWorkflow) -> WorkflowAnalysisResult:
        """
        Perform comprehensive analysis of a single workflow.
        
        Args:
            workflow: The workflow to analyze
            
        Returns:
            Analysis results containing metrics and insights
        """
        self.logger.info(f"Analyzing workflow: {workflow.name}")
        
        # Create analysis result container
        result = WorkflowAnalysisResult(
            workflow_id=workflow.id or "unknown",
            workflow_name=workflow.name,
            analysis_timestamp=datetime.now()
        )
        
        try:
            # Perform structure analysis
            structure_metrics = self.analyze_structure(workflow)
            result.add_metric("structure", structure_metrics)
            
            # Calculate complexity metrics
            complexity_metrics = self.calculate_complexity(workflow, structure_metrics)
            result.add_metric("complexity", complexity_metrics)
            
            # Analyze node types and distribution
            node_analysis = self.analyze_node_types(workflow)
            result.add_metric("node_analysis", node_analysis)
            
            # Analyze connection patterns
            connection_analysis = self.analyze_connections(workflow)
            result.add_metric("connection_analysis", connection_analysis)
            
            # Check error handling
            error_handling = self.analyze_error_handling(workflow)
            result.add_metric("error_handling", error_handling)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(workflow, result)
            for rec in recommendations:
                result.add_recommendation(rec)
            
            self.logger.info(f"Analysis completed for workflow: {workflow.name}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing workflow {workflow.name}: {e}")
            result.add_metric("error", str(e))
        
        return result
    
    def analyze_structure(self, workflow: N8nWorkflow) -> WorkflowStructureMetrics:
        """Analyze the structural properties of a workflow."""
        metrics = WorkflowStructureMetrics()
        
        # Basic counts
        metrics.node_count = len(workflow.nodes)
        metrics.connection_count = len(workflow.connections)
        
        if metrics.node_count == 0:
            return metrics
        
        # Build adjacency lists for analysis
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        
        for conn in workflow.connections:
            outgoing[conn.source_node].append(conn.target_node)
            incoming[conn.target_node].append(conn.source_node)
        
        # Calculate depth and width using BFS
        depth_width = self._calculate_depth_width(workflow.nodes, outgoing)
        metrics.max_depth = depth_width["max_depth"]
        metrics.max_width = depth_width["max_width"]
        
        # Calculate branching and convergence factors
        metrics.branching_factor = self._calculate_branching_factor(outgoing)
        metrics.convergence_factor = self._calculate_convergence_factor(incoming)
        
        # Check for cycles
        metrics.cyclic = self._has_cycles(workflow.nodes, outgoing)
        
        # Count connected components
        metrics.connected_components = self._count_connected_components(workflow.nodes, outgoing, incoming)
        
        # Node type distribution
        node_types = [node.type for node in workflow.nodes]
        metrics.node_type_distribution = dict(Counter(node_types))
        metrics.unique_node_types = len(set(node_types))
        
        # Connection statistics
        if metrics.node_count > 0:
            metrics.avg_connections_per_node = metrics.connection_count / metrics.node_count
        
        if outgoing:
            metrics.max_outgoing_connections = max(len(conns) for conns in outgoing.values())
        if incoming:
            metrics.max_incoming_connections = max(len(conns) for conns in incoming.values())
        
        return metrics
    
    def calculate_complexity(self, workflow: N8nWorkflow, structure_metrics: WorkflowStructureMetrics) -> ComplexityMetrics:
        """Calculate various complexity metrics for the workflow."""
        metrics = ComplexityMetrics()
        
        # Structural complexity (based on nodes, connections, and branching)
        base_complexity = structure_metrics.node_count + structure_metrics.connection_count
        branching_penalty = structure_metrics.branching_factor * 2
        depth_penalty = structure_metrics.max_depth * 0.5
        
        metrics.structural_complexity = base_complexity + branching_penalty + depth_penalty
        
        # Cognitive complexity (how hard it is to understand)
        cognitive_factors = 0
        
        # Add complexity for each decision point
        for node in workflow.nodes:
            if self._is_decision_node(node):
                cognitive_factors += 2
            elif self._is_loop_node(node):
                cognitive_factors += 3
            elif self._is_complex_node(node):
                cognitive_factors += 1
        
        metrics.cognitive_complexity = cognitive_factors + structure_metrics.max_depth
        
        # Cyclomatic complexity (number of linearly independent paths)
        # V(G) = E - N + 2P (where E=edges, N=nodes, P=connected components)
        metrics.cyclomatic_complexity = (
            structure_metrics.connection_count - 
            structure_metrics.node_count + 
            2 * structure_metrics.connected_components
        )
        
        # Halstead volume (based on operators and operands)
        halstead_metrics = self._calculate_halstead_metrics(workflow)
        metrics.halstead_volume = halstead_metrics["volume"]
        
        # Maintainability index
        metrics.maintainability_index = self._calculate_maintainability_index(
            metrics.halstead_volume,
            metrics.cyclomatic_complexity,
            structure_metrics.node_count
        )
        
        # Technical debt ratio (estimated)
        metrics.technical_debt_ratio = self._estimate_technical_debt(workflow, structure_metrics)
        
        # Naming consistency score
        metrics.naming_consistency_score = self._calculate_naming_consistency(workflow)
        
        # Documentation coverage
        metrics.documentation_coverage = self._calculate_documentation_coverage(workflow)
        
        return metrics
    
    def analyze_node_types(self, workflow: N8nWorkflow) -> Dict[str, Any]:
        """Analyze node type usage and patterns."""
        analysis = {
            "total_nodes": len(workflow.nodes),
            "unique_types": len(workflow.node_types),
            "type_distribution": {},
            "type_percentages": {},
            "common_types": [],
            "rare_types": [],
            "node_categories": {}
        }
        
        if not workflow.nodes:
            return analysis
        
        # Count node types
        type_counts = Counter(node.type for node in workflow.nodes)
        analysis["type_distribution"] = dict(type_counts)
        
        # Calculate percentages
        total_nodes = len(workflow.nodes)
        analysis["type_percentages"] = {
            node_type: (count / total_nodes) * 100
            for node_type, count in type_counts.items()
        }
        
        # Identify common and rare types
        sorted_types = type_counts.most_common()
        analysis["common_types"] = [t[0] for t in sorted_types[:5]]
        analysis["rare_types"] = [t[0] for t in sorted_types if t[1] == 1]
        
        # Categorize nodes
        analysis["node_categories"] = self._categorize_nodes(workflow.nodes)
        
        return analysis
    
    def analyze_connections(self, workflow: N8nWorkflow) -> Dict[str, Any]:
        """Analyze connection patterns and flow characteristics."""
        analysis = {
            "total_connections": len(workflow.connections),
            "connection_types": {},
            "flow_patterns": {},
            "bottlenecks": [],
            "dead_ends": [],
            "entry_points": [],
            "exit_points": []
        }
        
        if not workflow.connections:
            return analysis
        
        # Build connection maps
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        
        for conn in workflow.connections:
            outgoing[conn.source_node].append(conn.target_node)
            incoming[conn.target_node].append(conn.source_node)
        
        # Analyze connection types
        connection_types = Counter(conn.source_output for conn in workflow.connections)
        analysis["connection_types"] = dict(connection_types)
        
        # Identify flow patterns
        analysis["flow_patterns"] = {
            "linear_sequences": self._find_linear_sequences(workflow, outgoing, incoming),
            "parallel_branches": self._find_parallel_branches(workflow, outgoing),
            "merge_points": self._find_merge_points(workflow, incoming),
            "loops": self._find_loops(workflow, outgoing)
        }
        
        # Identify bottlenecks (nodes with many incoming connections)
        bottlenecks = [(node_id, len(conns)) for node_id, conns in incoming.items() if len(conns) > 2]
        analysis["bottlenecks"] = sorted(bottlenecks, key=lambda x: x[1], reverse=True)
        
        # Find dead ends (nodes with no outgoing connections)
        all_nodes = {node.id for node in workflow.nodes}
        nodes_with_outgoing = set(outgoing.keys())
        analysis["dead_ends"] = list(all_nodes - nodes_with_outgoing)
        
        # Find entry points (nodes with no incoming connections)
        nodes_with_incoming = set(incoming.keys())
        analysis["entry_points"] = list(all_nodes - nodes_with_incoming)
        
        # Exit points are the same as dead ends in this context
        analysis["exit_points"] = analysis["dead_ends"]
        
        return analysis
    
    def analyze_error_handling(self, workflow: N8nWorkflow) -> Dict[str, Any]:
        """Analyze error handling mechanisms in the workflow."""
        analysis = {
            "has_error_handling": False,
            "error_handling_types": [],
            "coverage_percentage": 0.0,
            "nodes_with_error_handling": [],
            "recommendations": []
        }
        
        error_handling_nodes = []
        
        for node in workflow.nodes:
            has_error_handling = False
            error_types = []
            
            # Check for continue on fail setting
            if node.parameters and node.parameters.get("continueOnFail", False):
                has_error_handling = True
                error_types.append("continue_on_fail")
            
            # Check for error workflow
            if node.parameters and node.parameters.get("onError"):
                has_error_handling = True
                error_types.append("error_workflow")
            
            # Check for try-catch patterns in function nodes
            if node.type == "n8n-nodes-base.function":
                code = node.parameters.get("functionCode", "")
                if "try" in code and "catch" in code:
                    has_error_handling = True
                    error_types.append("try_catch")
            
            # Check for error handling node types
            if any(keyword in node.type.lower() for keyword in ["error", "catch", "exception"]):
                has_error_handling = True
                error_types.append("error_node")
            
            if has_error_handling:
                error_handling_nodes.append({
                    "node_id": node.id,
                    "node_name": node.name,
                    "node_type": node.type,
                    "error_types": error_types
                })
        
        analysis["has_error_handling"] = len(error_handling_nodes) > 0
        analysis["nodes_with_error_handling"] = error_handling_nodes
        
        if workflow.nodes:
            analysis["coverage_percentage"] = (len(error_handling_nodes) / len(workflow.nodes)) * 100
        
        # Extract unique error handling types
        all_error_types = []
        for node_info in error_handling_nodes:
            all_error_types.extend(node_info["error_types"])
        analysis["error_handling_types"] = list(set(all_error_types))
        
        # Generate recommendations
        if analysis["coverage_percentage"] < 50:
            analysis["recommendations"].append("Consider adding error handling to more nodes")
        
        if not any("try_catch" in node_info["error_types"] for node_info in error_handling_nodes):
            analysis["recommendations"].append("Consider using try-catch blocks in function nodes")
        
        return analysis
    
    def generate_recommendations(self, workflow: N8nWorkflow, analysis_result: WorkflowAnalysisResult) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        structure_metrics = analysis_result.metrics.get("structure")
        complexity_metrics = analysis_result.metrics.get("complexity")
        
        if structure_metrics:
            # Complexity recommendations
            if structure_metrics.node_count > 50:
                recommendations.append("Consider breaking this large workflow into smaller, more manageable workflows")
            
            if structure_metrics.max_depth > 10:
                recommendations.append("Workflow has deep nesting - consider flattening the structure")
            
            if structure_metrics.branching_factor > 3:
                recommendations.append("High branching factor detected - consider simplifying decision logic")
            
            if not structure_metrics.has_error_handling:
                recommendations.append("Add error handling mechanisms to improve workflow reliability")
        
        if complexity_metrics:
            if complexity_metrics.cyclomatic_complexity > 15:
                recommendations.append("High cyclomatic complexity - consider refactoring to reduce complexity")
            
            if complexity_metrics.maintainability_index < 50:
                recommendations.append("Low maintainability index - improve code quality and documentation")
            
            if complexity_metrics.naming_consistency_score < 0.7:
                recommendations.append("Improve naming consistency across workflow nodes")
            
            if complexity_metrics.documentation_coverage < 0.5:
                recommendations.append("Add more documentation and notes to workflow nodes")
        
        return recommendations
    
    # Helper methods
    
    def _calculate_depth_width(self, nodes: List[WorkflowNode], outgoing: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate maximum depth and width of the workflow graph."""
        from collections import deque
        
        # Find entry points (nodes with no incoming connections)
        all_nodes = {node.id for node in nodes}
        nodes_with_incoming = set()
        for source_nodes in outgoing.values():
            nodes_with_incoming.update(source_nodes)
        
        entry_points = all_nodes - nodes_with_incoming
        
        if not entry_points:
            # If no clear entry points, start from first node
            entry_points = {nodes[0].id} if nodes else set()
        
        max_depth = 0
        max_width = 0
        
        for entry_point in entry_points:
            # BFS to calculate depth and width
            queue = deque([(entry_point, 0)])
            visited = set()
            level_counts = defaultdict(int)
            
            while queue:
                node_id, depth = queue.popleft()
                
                if node_id in visited:
                    continue
                
                visited.add(node_id)
                level_counts[depth] += 1
                max_depth = max(max_depth, depth)
                
                # Add children to queue
                for child in outgoing.get(node_id, []):
                    if child not in visited:
                        queue.append((child, depth + 1))
            
            # Update max width
            if level_counts:
                max_width = max(max_width, max(level_counts.values()))
        
        return {"max_depth": max_depth, "max_width": max_width}
    
    def _calculate_branching_factor(self, outgoing: Dict[str, List[str]]) -> float:
        """Calculate average branching factor."""
        if not outgoing:
            return 0.0
        
        branch_counts = [len(children) for children in outgoing.values() if children]
        return sum(branch_counts) / len(branch_counts) if branch_counts else 0.0
    
    def _calculate_convergence_factor(self, incoming: Dict[str, List[str]]) -> float:
        """Calculate average convergence factor."""
        if not incoming:
            return 0.0
        
        convergence_counts = [len(parents) for parents in incoming.values() if len(parents) > 1]
        return sum(convergence_counts) / len(convergence_counts) if convergence_counts else 0.0
    
    def _has_cycles(self, nodes: List[WorkflowNode], outgoing: Dict[str, List[str]]) -> bool:
        """Check if the workflow graph has cycles using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in outgoing.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in nodes:
            if node.id not in visited:
                if dfs(node.id):
                    return True
        
        return False
    
    def _count_connected_components(self, nodes: List[WorkflowNode], outgoing: Dict[str, List[str]], incoming: Dict[str, List[str]]) -> int:
        """Count the number of connected components in the graph."""
        visited = set()
        components = 0
        
        # Build undirected adjacency list
        adjacency = defaultdict(set)
        for node_id, children in outgoing.items():
            for child in children:
                adjacency[node_id].add(child)
                adjacency[child].add(node_id)
        
        def dfs(node_id):
            visited.add(node_id)
            for neighbor in adjacency[node_id]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        for node in nodes:
            if node.id not in visited:
                dfs(node.id)
                components += 1
        
        return components
    
    def _is_decision_node(self, node: WorkflowNode) -> bool:
        """Check if a node is a decision/conditional node."""
        decision_types = ["if", "switch", "router", "condition"]
        return any(dt in node.type.lower() for dt in decision_types)
    
    def _is_loop_node(self, node: WorkflowNode) -> bool:
        """Check if a node represents a loop or iteration."""
        loop_types = ["loop", "iterate", "repeat", "while", "for"]
        return any(lt in node.type.lower() for lt in loop_types)
    
    def _is_complex_node(self, node: WorkflowNode) -> bool:
        """Check if a node is inherently complex."""
        complex_types = ["function", "code", "script", "transform"]
        return any(ct in node.type.lower() for ct in complex_types)
    
    def _calculate_halstead_metrics(self, workflow: N8nWorkflow) -> Dict[str, float]:
        """Calculate Halstead complexity metrics."""
        # Simplified Halstead metrics for workflows
        operators = set()  # Node types
        operands = set()   # Node names and parameters
        
        operator_count = 0
        operand_count = 0
        
        for node in workflow.nodes:
            # Node type is an operator
            operators.add(node.type)
            operator_count += 1
            
            # Node name is an operand
            if node.name:
                operands.add(node.name)
                operand_count += 1
            
            # Parameters are operands
            if node.parameters:
                for key, value in node.parameters.items():
                    if isinstance(value, str):
                        operands.add(value)
                        operand_count += 1
        
        n1 = len(operators)  # Number of distinct operators
        n2 = len(operands)   # Number of distinct operands
        N1 = operator_count  # Total number of operators
        N2 = operand_count   # Total number of operands
        
        if n1 == 0 or n2 == 0:
            return {"volume": 0.0}
        
        vocabulary = n1 + n2
        length = N1 + N2
        
        if vocabulary <= 0:
            return {"volume": 0.0}
        
        volume = length * np.log2(vocabulary)
        
        return {"volume": volume}
    
    def _calculate_maintainability_index(self, halstead_volume: float, cyclomatic_complexity: int, lines_of_code: int) -> float:
        """Calculate maintainability index."""
        # Simplified maintainability index for workflows
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        
        if halstead_volume <= 0 or lines_of_code <= 0:
            return 100.0  # Default high maintainability for simple workflows
        
        mi = 171 - 5.2 * np.log(halstead_volume) - 0.23 * cyclomatic_complexity - 16.2 * np.log(lines_of_code)
        return max(0.0, min(100.0, mi))  # Clamp between 0 and 100
    
    def _estimate_technical_debt(self, workflow: N8nWorkflow, structure_metrics: WorkflowStructureMetrics) -> float:
        """Estimate technical debt ratio."""
        debt_factors = 0
        
        # Lack of error handling
        if not structure_metrics.has_error_handling:
            debt_factors += 0.2
        
        # High complexity
        if structure_metrics.node_count > 30:
            debt_factors += 0.1
        
        # Poor naming (nodes with default names)
        default_names = sum(1 for node in workflow.nodes if not node.name or node.name == node.type)
        if workflow.nodes and default_names / len(workflow.nodes) > 0.3:
            debt_factors += 0.15
        
        # Lack of documentation
        documented_nodes = sum(1 for node in workflow.nodes if node.notes)
        if workflow.nodes and documented_nodes / len(workflow.nodes) < 0.2:
            debt_factors += 0.1
        
        return min(1.0, debt_factors)
    
    def _calculate_naming_consistency(self, workflow: N8nWorkflow) -> float:
        """Calculate naming consistency score."""
        if not workflow.nodes:
            return 1.0
        
        # Check for consistent naming patterns
        names = [node.name for node in workflow.nodes if node.name]
        
        if not names:
            return 0.0
        
        # Simple heuristics for naming consistency
        score = 1.0
        
        # Check for default names
        default_names = sum(1 for name in names if not name or name.startswith("Node"))
        if names:
            score -= (default_names / len(names)) * 0.5
        
        # Check for consistent casing
        title_case = sum(1 for name in names if name.istitle())
        if names and title_case / len(names) < 0.8:
            score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_documentation_coverage(self, workflow: N8nWorkflow) -> float:
        """Calculate documentation coverage."""
        if not workflow.nodes:
            return 1.0
        
        documented_nodes = sum(1 for node in workflow.nodes if node.notes and node.notes.strip())
        return documented_nodes / len(workflow.nodes)
    
    def _categorize_nodes(self, nodes: List[WorkflowNode]) -> Dict[str, List[str]]:
        """Categorize nodes by their function."""
        categories = {
            "triggers": [],
            "actions": [],
            "transformations": [],
            "conditions": [],
            "integrations": [],
            "utilities": []
        }
        
        for node in nodes:
            node_type = node.type.lower()
            
            if any(trigger in node_type for trigger in ["trigger", "webhook", "cron", "start"]):
                categories["triggers"].append(node.id)
            elif any(action in node_type for action in ["send", "post", "create", "update", "delete"]):
                categories["actions"].append(node.id)
            elif any(transform in node_type for transform in ["function", "set", "json", "xml", "transform"]):
                categories["transformations"].append(node.id)
            elif any(condition in node_type for condition in ["if", "switch", "filter", "condition"]):
                categories["conditions"].append(node.id)
            elif any(integration in node_type for integration in ["http", "api", "database", "slack", "email"]):
                categories["integrations"].append(node.id)
            else:
                categories["utilities"].append(node.id)
        
        return categories
    
    def _find_linear_sequences(self, workflow: N8nWorkflow, outgoing: Dict[str, List[str]], incoming: Dict[str, List[str]]) -> List[List[str]]:
        """Find linear sequences of nodes."""
        sequences = []
        visited = set()
        
        for node in workflow.nodes:
            if node.id in visited:
                continue
            
            # Start a sequence if this node has at most one incoming and one outgoing connection
            if (len(incoming.get(node.id, [])) <= 1 and 
                len(outgoing.get(node.id, [])) <= 1):
                
                sequence = [node.id]
                visited.add(node.id)
                
                # Follow the chain forward
                current = node.id
                while (len(outgoing.get(current, [])) == 1 and 
                       len(incoming.get(outgoing[current][0], [])) == 1):
                    next_node = outgoing[current][0]
                    if next_node in visited:
                        break
                    sequence.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if len(sequence) > 1:
                    sequences.append(sequence)
        
        return sequences
    
    def _find_parallel_branches(self, workflow: N8nWorkflow, outgoing: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Find parallel branch patterns."""
        branches = []
        
        for node_id, children in outgoing.items():
            if len(children) > 1:
                branches.append({
                    "source": node_id,
                    "branches": children,
                    "branch_count": len(children)
                })
        
        return branches
    
    def _find_merge_points(self, workflow: N8nWorkflow, incoming: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Find merge points where multiple branches converge."""
        merge_points = []
        
        for node_id, parents in incoming.items():
            if len(parents) > 1:
                merge_points.append({
                    "target": node_id,
                    "sources": parents,
                    "merge_count": len(parents)
                })
        
        return merge_points
    
    def _find_loops(self, workflow: N8nWorkflow, outgoing: Dict[str, List[str]]) -> List[List[str]]:
        """Find loop patterns in the workflow."""
        loops = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node_id):
            if node_id in rec_stack:
                # Found a cycle, extract the loop
                loop_start = path.index(node_id)
                loop = path[loop_start:] + [node_id]
                loops.append(loop)
                return
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for neighbor in outgoing.get(node_id, []):
                dfs(neighbor)
            
            rec_stack.remove(node_id)
            path.pop()
        
        for node in workflow.nodes:
            if node.id not in visited:
                dfs(node.id)
        
        return loops


# Import datetime at the top of the file
from datetime import datetime

