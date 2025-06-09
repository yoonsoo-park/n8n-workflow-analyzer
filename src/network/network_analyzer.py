"""
Network analysis and graph processing for n8n workflows.

This module provides graph-based analysis capabilities for workflows,
including centrality measures, community detection, and path analysis.
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from dataclasses import dataclass, field

# Community detection
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    logging.warning("igraph not available, some community detection features will be limited")

try:
    from ..models import N8nWorkflow, WorkflowCollection, WorkflowNode, WorkflowConnection
    from ..config import get_analysis_config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import N8nWorkflow, WorkflowCollection, WorkflowNode, WorkflowConnection
    from config import get_analysis_config


logger = logging.getLogger(__name__)


@dataclass
class NodeCentrality:
    """Container for node centrality measures."""
    
    node_id: str
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'degree': self.degree_centrality,
            'betweenness': self.betweenness_centrality,
            'closeness': self.closeness_centrality,
            'eigenvector': self.eigenvector_centrality,
            'pagerank': self.pagerank
        }


@dataclass
class Community:
    """Represents a community of nodes in a workflow graph."""
    
    community_id: int
    nodes: List[str]
    size: int = 0
    modularity: float = 0.0
    internal_edges: int = 0
    external_edges: int = 0
    
    def __post_init__(self):
        self.size = len(self.nodes)


@dataclass
class PathAnalysis:
    """Container for path analysis results."""
    
    shortest_paths: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    path_lengths: Dict[Tuple[str, str], int] = field(default_factory=dict)
    critical_paths: List[List[str]] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    diameter: int = 0
    average_path_length: float = 0.0


class NetworkAnalyzer:
    """Main class for network analysis of workflow graphs."""
    
    def __init__(self):
        """Initialize the network analyzer."""
        self.config = get_analysis_config()
        self.logger = logging.getLogger(__name__)
    
    def create_workflow_graph(self, workflow: N8nWorkflow) -> nx.DiGraph:
        """
        Create a NetworkX directed graph from a workflow.
        
        Args:
            workflow: The workflow to convert to a graph
            
        Returns:
            NetworkX directed graph representation
        """
        graph = nx.DiGraph()
        
        # Add nodes with attributes
        for node in workflow.nodes:
            graph.add_node(
                node.id,
                name=node.name,
                type=node.type,
                position=node.position,
                disabled=node.disabled,
                has_notes=bool(node.notes)
            )
        
        # Add edges
        for connection in workflow.connections:
            graph.add_edge(
                connection.source_node,
                connection.target_node,
                output_type=connection.source_output,
                input_type=connection.target_input,
                output_index=connection.source_output_index,
                input_index=connection.target_input_index
            )
        
        return graph
    
    def create_collection_graph(self, collection: WorkflowCollection) -> nx.Graph:
        """
        Create a graph representing relationships between workflows in a collection.
        
        Args:
            collection: Collection of workflows
            
        Returns:
            NetworkX graph where nodes are workflows and edges represent similarities
        """
        graph = nx.Graph()
        
        # Add workflow nodes
        for workflow in collection.workflows:
            graph.add_node(
                workflow.id or workflow.name,
                name=workflow.name,
                node_count=len(workflow.nodes),
                connection_count=len(workflow.connections),
                complexity=workflow.complexity_score,
                active=workflow.active,
                tags=workflow.tags
            )
        
        # Add edges based on similarity
        workflows = collection.workflows
        for i in range(len(workflows)):
            for j in range(i + 1, len(workflows)):
                similarity = self._calculate_workflow_similarity(workflows[i], workflows[j])
                if similarity > 0.3:  # Threshold for creating an edge
                    graph.add_edge(
                        workflows[i].id or workflows[i].name,
                        workflows[j].id or workflows[j].name,
                        weight=similarity,
                        similarity=similarity
                    )
        
        return graph
    
    def calculate_centrality_measures(self, graph: nx.DiGraph) -> Dict[str, NodeCentrality]:
        """
        Calculate various centrality measures for all nodes in the graph.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary mapping node IDs to centrality measures
        """
        self.logger.info(f"Calculating centrality measures for {len(graph.nodes)} nodes")
        
        centralities = {}
        
        # Initialize centrality objects
        for node_id in graph.nodes():
            centralities[node_id] = NodeCentrality(node_id=node_id)
        
        try:
            # Degree centrality
            degree_cent = nx.degree_centrality(graph)
            for node_id, value in degree_cent.items():
                centralities[node_id].degree_centrality = value
            
            # Betweenness centrality
            betweenness_cent = nx.betweenness_centrality(graph)
            for node_id, value in betweenness_cent.items():
                centralities[node_id].betweenness_centrality = value
            
            # Closeness centrality (only for strongly connected components)
            if nx.is_strongly_connected(graph):
                closeness_cent = nx.closeness_centrality(graph)
                for node_id, value in closeness_cent.items():
                    centralities[node_id].closeness_centrality = value
            else:
                # Calculate for largest strongly connected component
                largest_scc = max(nx.strongly_connected_components(graph), key=len)
                if len(largest_scc) > 1:
                    scc_graph = graph.subgraph(largest_scc)
                    closeness_cent = nx.closeness_centrality(scc_graph)
                    for node_id, value in closeness_cent.items():
                        centralities[node_id].closeness_centrality = value
            
            # Eigenvector centrality (for strongly connected graphs)
            try:
                eigenvector_cent = nx.eigenvector_centrality(graph, max_iter=1000)
                for node_id, value in eigenvector_cent.items():
                    centralities[node_id].eigenvector_centrality = value
            except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
                self.logger.warning("Could not calculate eigenvector centrality")
            
            # PageRank
            pagerank = nx.pagerank(graph)
            for node_id, value in pagerank.items():
                centralities[node_id].pagerank = value
                
        except Exception as e:
            self.logger.error(f"Error calculating centrality measures: {e}")
        
        return centralities
    
    def detect_communities(self, graph: Union[nx.Graph, nx.DiGraph]) -> List[Community]:
        """
        Detect communities in the graph using various algorithms.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            List of detected communities
        """
        self.logger.info(f"Detecting communities in graph with {len(graph.nodes)} nodes")
        
        communities = []
        
        # Convert to undirected for community detection
        if isinstance(graph, nx.DiGraph):
            undirected_graph = graph.to_undirected()
        else:
            undirected_graph = graph
        
        if len(undirected_graph.nodes) < 2:
            return communities
        
        try:
            # Use different community detection algorithms
            algorithm = self.config.community_detection_algorithm.lower()
            
            if algorithm == "louvain":
                communities = self._detect_communities_louvain(undirected_graph)
            elif algorithm == "greedy_modularity" or not IGRAPH_AVAILABLE:
                communities = self._detect_communities_greedy(undirected_graph)
            elif algorithm == "leiden" and IGRAPH_AVAILABLE:
                communities = self._detect_communities_leiden(undirected_graph)
            else:
                # Default to greedy modularity
                communities = self._detect_communities_greedy(undirected_graph)
                
        except Exception as e:
            self.logger.error(f"Error detecting communities: {e}")
        
        return communities
    
    def analyze_paths(self, graph: nx.DiGraph) -> PathAnalysis:
        """
        Analyze paths in the workflow graph.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Path analysis results
        """
        self.logger.info(f"Analyzing paths in graph with {len(graph.nodes)} nodes")
        
        analysis = PathAnalysis()
        
        if len(graph.nodes) < 2:
            return analysis
        
        try:
            # Calculate shortest paths
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        try:
                            path = nx.shortest_path(graph, source, target)
                            analysis.shortest_paths[(source, target)] = path
                            analysis.path_lengths[(source, target)] = len(path) - 1
                        except nx.NetworkXNoPath:
                            # No path exists
                            analysis.path_lengths[(source, target)] = float('inf')
            
            # Calculate diameter (longest shortest path)
            finite_lengths = [length for length in analysis.path_lengths.values() 
                            if length != float('inf')]
            if finite_lengths:
                analysis.diameter = max(finite_lengths)
                analysis.average_path_length = np.mean(finite_lengths)
            
            # Find critical paths (longest paths from entry to exit points)
            analysis.critical_paths = self._find_critical_paths(graph)
            
            # Identify bottlenecks
            analysis.bottlenecks = self._identify_bottlenecks(graph)
            
        except Exception as e:
            self.logger.error(f"Error analyzing paths: {e}")
        
        return analysis
    
    def calculate_graph_metrics(self, graph: Union[nx.Graph, nx.DiGraph]) -> Dict[str, Any]:
        """
        Calculate various graph-level metrics.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of graph metrics
        """
        metrics = {
            'node_count': len(graph.nodes),
            'edge_count': len(graph.edges),
            'density': 0.0,
            'clustering_coefficient': 0.0,
            'connected_components': 0,
            'strongly_connected_components': 0,
            'diameter': 0,
            'radius': 0,
            'average_degree': 0.0,
            'degree_distribution': {},
            'assortativity': 0.0
        }
        
        if len(graph.nodes) == 0:
            return metrics
        
        try:
            # Basic metrics
            metrics['density'] = nx.density(graph)
            
            if isinstance(graph, nx.DiGraph):
                metrics['strongly_connected_components'] = nx.number_strongly_connected_components(graph)
                # Convert to undirected for some metrics
                undirected = graph.to_undirected()
                metrics['connected_components'] = nx.number_connected_components(undirected)
                metrics['clustering_coefficient'] = nx.average_clustering(undirected)
            else:
                metrics['connected_components'] = nx.number_connected_components(graph)
                metrics['clustering_coefficient'] = nx.average_clustering(graph)
            
            # Degree statistics
            degrees = [d for n, d in graph.degree()]
            if degrees:
                metrics['average_degree'] = np.mean(degrees)
                degree_counts = {}
                for degree in degrees:
                    degree_counts[degree] = degree_counts.get(degree, 0) + 1
                metrics['degree_distribution'] = degree_counts
            
            # Diameter and radius (for connected graphs)
            if isinstance(graph, nx.DiGraph):
                if nx.is_strongly_connected(graph):
                    metrics['diameter'] = nx.diameter(graph)
                    metrics['radius'] = nx.radius(graph)
            else:
                if nx.is_connected(graph):
                    metrics['diameter'] = nx.diameter(graph)
                    metrics['radius'] = nx.radius(graph)
            
            # Assortativity
            try:
                metrics['assortativity'] = nx.degree_assortativity_coefficient(graph)
            except:
                pass  # May fail for some graph types
                
        except Exception as e:
            self.logger.error(f"Error calculating graph metrics: {e}")
        
        return metrics
    
    def find_influential_nodes(self, graph: nx.DiGraph, top_k: int = 5) -> Dict[str, List[str]]:
        """
        Find the most influential nodes based on various centrality measures.
        
        Args:
            graph: NetworkX directed graph
            top_k: Number of top nodes to return for each measure
            
        Returns:
            Dictionary mapping centrality measure names to lists of top nodes
        """
        centralities = self.calculate_centrality_measures(graph)
        
        influential_nodes = {}
        
        # Sort by each centrality measure
        measures = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 
                   'eigenvector_centrality', 'pagerank']
        
        for measure in measures:
            sorted_nodes = sorted(
                centralities.items(),
                key=lambda x: getattr(x[1], measure),
                reverse=True
            )
            influential_nodes[measure] = [node_id for node_id, _ in sorted_nodes[:top_k]]
        
        return influential_nodes
    
    def analyze_workflow_dependencies(self, collection: WorkflowCollection) -> Dict[str, Any]:
        """
        Analyze dependencies between workflows in a collection.
        
        Args:
            collection: Collection of workflows
            
        Returns:
            Dependency analysis results
        """
        self.logger.info("Analyzing workflow dependencies")
        
        analysis = {
            'dependency_graph': None,
            'dependency_chains': [],
            'isolated_workflows': [],
            'hub_workflows': [],
            'dependency_metrics': {}
        }
        
        # Create dependency graph
        dep_graph = self._create_dependency_graph(collection)
        analysis['dependency_graph'] = dep_graph
        
        # Find dependency chains
        analysis['dependency_chains'] = self._find_dependency_chains(dep_graph)
        
        # Find isolated workflows
        analysis['isolated_workflows'] = [
            node for node in dep_graph.nodes() 
            if dep_graph.degree(node) == 0
        ]
        
        # Find hub workflows (high degree)
        degrees = dict(dep_graph.degree())
        avg_degree = np.mean(list(degrees.values())) if degrees else 0
        analysis['hub_workflows'] = [
            node for node, degree in degrees.items() 
            if degree > avg_degree * 2
        ]
        
        # Calculate dependency metrics
        analysis['dependency_metrics'] = self.calculate_graph_metrics(dep_graph)
        
        return analysis
    
    # Helper methods
    
    def _calculate_workflow_similarity(self, workflow1: N8nWorkflow, workflow2: N8nWorkflow) -> float:
        """Calculate similarity between two workflows."""
        # Node type similarity
        types1 = set(node.type for node in workflow1.nodes)
        types2 = set(node.type for node in workflow2.nodes)
        
        if not types1 and not types2:
            type_similarity = 1.0
        elif not types1 or not types2:
            type_similarity = 0.0
        else:
            intersection = len(types1 & types2)
            union = len(types1 | types2)
            type_similarity = intersection / union if union > 0 else 0.0
        
        # Tag similarity
        tags1 = set(workflow1.tags)
        tags2 = set(workflow2.tags)
        
        if not tags1 and not tags2:
            tag_similarity = 1.0
        elif not tags1 or not tags2:
            tag_similarity = 0.0
        else:
            intersection = len(tags1 & tags2)
            union = len(tags1 | tags2)
            tag_similarity = intersection / union if union > 0 else 0.0
        
        # Size similarity
        size1 = len(workflow1.nodes)
        size2 = len(workflow2.nodes)
        max_size = max(size1, size2)
        size_similarity = 1.0 - abs(size1 - size2) / max_size if max_size > 0 else 1.0
        
        # Weighted average
        similarity = (type_similarity * 0.5 + tag_similarity * 0.3 + size_similarity * 0.2)
        return similarity
    
    def _detect_communities_louvain(self, graph: nx.Graph) -> List[Community]:
        """Detect communities using Louvain algorithm (requires python-louvain)."""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
            
            result = []
            for comm_id, nodes in communities.items():
                community = Community(
                    community_id=comm_id,
                    nodes=nodes
                )
                result.append(community)
            
            return result
            
        except ImportError:
            self.logger.warning("python-louvain not available, falling back to greedy modularity")
            return self._detect_communities_greedy(graph)
    
    def _detect_communities_greedy(self, graph: nx.Graph) -> List[Community]:
        """Detect communities using greedy modularity optimization."""
        try:
            communities_generator = nx.community.greedy_modularity_communities(graph)
            communities = list(communities_generator)
            
            result = []
            for i, community_nodes in enumerate(communities):
                community = Community(
                    community_id=i,
                    nodes=list(community_nodes)
                )
                result.append(community)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in greedy community detection: {e}")
            return []
    
    def _detect_communities_leiden(self, graph: nx.Graph) -> List[Community]:
        """Detect communities using Leiden algorithm (requires igraph)."""
        if not IGRAPH_AVAILABLE:
            return self._detect_communities_greedy(graph)
        
        try:
            # Convert NetworkX graph to igraph
            ig_graph = ig.Graph.from_networkx(graph)
            
            # Apply Leiden algorithm
            partition = ig_graph.community_leiden()
            
            result = []
            for i, community_nodes in enumerate(partition):
                node_names = [ig_graph.vs[node]['_nx_name'] for node in community_nodes]
                community = Community(
                    community_id=i,
                    nodes=node_names,
                    modularity=partition.modularity
                )
                result.append(community)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Leiden community detection: {e}")
            return self._detect_communities_greedy(graph)
    
    def _find_critical_paths(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find critical paths in the workflow graph."""
        critical_paths = []
        
        # Find entry points (nodes with no incoming edges)
        entry_points = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        
        # Find exit points (nodes with no outgoing edges)
        exit_points = [node for node in graph.nodes() if graph.out_degree(node) == 0]
        
        # Find longest paths from each entry to each exit point
        for entry in entry_points:
            for exit_point in exit_points:
                try:
                    # Use all simple paths and find the longest one
                    all_paths = list(nx.all_simple_paths(graph, entry, exit_point))
                    if all_paths:
                        longest_path = max(all_paths, key=len)
                        critical_paths.append(longest_path)
                except nx.NetworkXNoPath:
                    continue
        
        # Remove duplicates and sort by length
        unique_paths = []
        for path in critical_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        
        unique_paths.sort(key=len, reverse=True)
        return unique_paths[:5]  # Return top 5 critical paths
    
    def _identify_bottlenecks(self, graph: nx.DiGraph) -> List[str]:
        """Identify bottleneck nodes in the graph."""
        bottlenecks = []
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(graph)
        
        # Nodes with high betweenness centrality are potential bottlenecks
        threshold = np.mean(list(betweenness.values())) + np.std(list(betweenness.values()))
        
        for node, centrality in betweenness.items():
            if centrality > threshold:
                bottlenecks.append(node)
        
        # Also consider nodes with high in-degree (many dependencies)
        in_degrees = dict(graph.in_degree())
        avg_in_degree = np.mean(list(in_degrees.values()))
        
        for node, in_degree in in_degrees.items():
            if in_degree > avg_in_degree * 2 and node not in bottlenecks:
                bottlenecks.append(node)
        
        return bottlenecks
    
    def _create_dependency_graph(self, collection: WorkflowCollection) -> nx.DiGraph:
        """Create a dependency graph between workflows."""
        # For now, create a simple graph based on workflow similarities
        # In a real implementation, this would analyze actual dependencies
        # like shared data sources, webhook chains, etc.
        
        graph = nx.DiGraph()
        
        # Add workflow nodes
        for workflow in collection.workflows:
            graph.add_node(workflow.id or workflow.name, workflow=workflow)
        
        # Add edges based on potential dependencies
        # This is a simplified implementation
        workflows = collection.workflows
        for i, workflow1 in enumerate(workflows):
            for j, workflow2 in enumerate(workflows):
                if i != j:
                    # Check for potential dependencies
                    if self._has_potential_dependency(workflow1, workflow2):
                        graph.add_edge(
                            workflow1.id or workflow1.name,
                            workflow2.id or workflow2.name,
                            dependency_type="potential"
                        )
        
        return graph
    
    def _has_potential_dependency(self, workflow1: N8nWorkflow, workflow2: N8nWorkflow) -> bool:
        """Check if workflow1 might depend on workflow2."""
        # Simple heuristic: if workflow1 has webhook and workflow2 has HTTP request
        has_webhook_1 = any("webhook" in node.type.lower() for node in workflow1.nodes)
        has_http_2 = any("http" in node.type.lower() for node in workflow2.nodes)
        
        return has_webhook_1 and has_http_2
    
    def _find_dependency_chains(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find chains of dependencies in the graph."""
        chains = []
        
        # Find all simple paths of length > 2
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
                        for path in paths:
                            if len(path) > 2:  # Chain of at least 3 workflows
                                chains.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        # Remove duplicates and sort by length
        unique_chains = []
        for chain in chains:
            if chain not in unique_chains:
                unique_chains.append(chain)
        
        unique_chains.sort(key=len, reverse=True)
        return unique_chains[:10]  # Return top 10 chains

