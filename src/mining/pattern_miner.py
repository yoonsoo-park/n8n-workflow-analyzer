"""
Pattern mining and recognition for n8n workflows.

This module implements various data mining techniques to discover patterns
in workflow collections, including frequent pattern mining, association rules,
and clustering algorithms.
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# Machine learning imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Pattern mining imports
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

try:
    from ..models import N8nWorkflow, WorkflowCollection, WorkflowAnalysisResult
    from ..config import get_analysis_config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import N8nWorkflow, WorkflowCollection, WorkflowAnalysisResult
    from config import get_analysis_config


logger = logging.getLogger(__name__)


@dataclass
class FrequentPattern:
    """Represents a frequent pattern found in workflows."""
    
    items: frozenset
    support: float
    frequency: int
    workflows: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"Pattern({list(self.items)}, support={self.support:.3f})"


@dataclass
class AssociationRule:
    """Represents an association rule between workflow patterns."""
    
    antecedent: frozenset
    consequent: frozenset
    support: float
    confidence: float
    lift: float
    conviction: float
    
    def __str__(self) -> str:
        return f"Rule({list(self.antecedent)} -> {list(self.consequent)}, conf={self.confidence:.3f}, lift={self.lift:.3f})"


@dataclass
class WorkflowCluster:
    """Represents a cluster of similar workflows."""
    
    cluster_id: int
    workflows: List[str]
    centroid: Optional[np.ndarray] = None
    characteristics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        return len(self.workflows)


class PatternMiner:
    """Main class for pattern mining and recognition in workflow collections."""
    
    def __init__(self):
        """Initialize the pattern miner."""
        self.config = get_analysis_config()
        self.logger = logging.getLogger(__name__)
    
    def mine_frequent_patterns(self, collection: WorkflowCollection) -> List[FrequentPattern]:
        """
        Mine frequent patterns using FP-Growth algorithm.
        
        Args:
            collection: Collection of workflows to analyze
            
        Returns:
            List of frequent patterns found
        """
        self.logger.info(f"Mining frequent patterns from {collection.total_workflows} workflows")
        
        # Prepare transaction data
        transactions = self._prepare_transactions(collection)
        
        if not transactions:
            self.logger.warning("No transactions found for pattern mining")
            return []
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Apply FP-Growth algorithm
        frequent_itemsets = fpgrowth(
            df, 
            min_support=self.config.min_support, 
            use_colnames=True,
            max_len=self.config.max_itemsets
        )
        
        # Convert to FrequentPattern objects
        patterns = []
        for _, row in frequent_itemsets.iterrows():
            pattern = FrequentPattern(
                items=row['itemsets'],
                support=row['support'],
                frequency=int(row['support'] * len(transactions))
            )
            
            # Find workflows containing this pattern
            pattern.workflows = self._find_workflows_with_pattern(collection, pattern.items)
            patterns.append(pattern)
        
        self.logger.info(f"Found {len(patterns)} frequent patterns")
        return patterns
    
    def generate_association_rules(self, patterns: List[FrequentPattern]) -> List[AssociationRule]:
        """
        Generate association rules from frequent patterns.
        
        Args:
            patterns: List of frequent patterns
            
        Returns:
            List of association rules
        """
        self.logger.info(f"Generating association rules from {len(patterns)} patterns")
        
        if not patterns:
            return []
        
        # Convert patterns to DataFrame for mlxtend
        pattern_data = []
        for pattern in patterns:
            pattern_data.append({
                'itemsets': pattern.items,
                'support': pattern.support
            })
        
        df_patterns = pd.DataFrame(pattern_data)
        
        # Generate association rules
        rules_df = association_rules(
            df_patterns,
            metric="confidence",
            min_threshold=self.config.min_confidence
        )
        
        # Convert to AssociationRule objects
        rules = []
        for _, row in rules_df.iterrows():
            rule = AssociationRule(
                antecedent=row['antecedents'],
                consequent=row['consequents'],
                support=row['support'],
                confidence=row['confidence'],
                lift=row['lift'],
                conviction=row['conviction']
            )
            rules.append(rule)
        
        self.logger.info(f"Generated {len(rules)} association rules")
        return rules
    
    def mine_sequential_patterns(self, collection: WorkflowCollection) -> List[List[str]]:
        """
        Mine sequential patterns in workflow execution flows.
        
        Args:
            collection: Collection of workflows to analyze
            
        Returns:
            List of sequential patterns
        """
        self.logger.info(f"Mining sequential patterns from {collection.total_workflows} workflows")
        
        sequences = []
        
        for workflow in collection.workflows:
            # Extract execution sequence from workflow structure
            sequence = self._extract_execution_sequence(workflow)
            if sequence:
                sequences.append(sequence)
        
        # Find common subsequences
        common_sequences = self._find_common_subsequences(sequences)
        
        self.logger.info(f"Found {len(common_sequences)} sequential patterns")
        return common_sequences
    
    def cluster_workflows(self, collection: WorkflowCollection, n_clusters: Optional[int] = None) -> List[WorkflowCluster]:
        """
        Cluster workflows based on their characteristics.
        
        Args:
            collection: Collection of workflows to cluster
            n_clusters: Number of clusters (if None, will be determined automatically)
            
        Returns:
            List of workflow clusters
        """
        self.logger.info(f"Clustering {collection.total_workflows} workflows")
        
        if collection.total_workflows < 2:
            self.logger.warning("Need at least 2 workflows for clustering")
            return []
        
        # Extract features for clustering
        features, workflow_ids = self._extract_clustering_features(collection)
        
        if features.shape[0] < 2:
            self.logger.warning("Insufficient features for clustering")
            return []
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(features_scaled)
        
        # Perform clustering
        clusters = self._perform_clustering(features_scaled, workflow_ids, n_clusters)
        
        # Analyze cluster characteristics
        for cluster in clusters:
            cluster.characteristics = self._analyze_cluster_characteristics(
                collection, cluster.workflows
            )
        
        self.logger.info(f"Created {len(clusters)} workflow clusters")
        return clusters
    
    def find_workflow_templates(self, collection: WorkflowCollection) -> List[Dict[str, Any]]:
        """
        Identify common workflow templates and patterns.
        
        Args:
            collection: Collection of workflows to analyze
            
        Returns:
            List of identified templates
        """
        self.logger.info("Identifying workflow templates")
        
        templates = []
        
        # Group workflows by structural similarity
        structure_groups = self._group_by_structure(collection)
        
        for group_id, workflows in structure_groups.items():
            if len(workflows) >= 2:  # Template needs at least 2 instances
                template = self._create_template_from_group(workflows)
                templates.append(template)
        
        self.logger.info(f"Identified {len(templates)} workflow templates")
        return templates
    
    def analyze_node_usage_patterns(self, collection: WorkflowCollection) -> Dict[str, Any]:
        """
        Analyze patterns in node usage across workflows.
        
        Args:
            collection: Collection of workflows to analyze
            
        Returns:
            Dictionary containing usage pattern analysis
        """
        self.logger.info("Analyzing node usage patterns")
        
        analysis = {
            "node_frequency": {},
            "node_combinations": {},
            "node_sequences": {},
            "rare_nodes": [],
            "popular_nodes": [],
            "node_categories": {}
        }
        
        # Count node type frequencies
        all_node_types = []
        for workflow in collection.workflows:
            all_node_types.extend([node.type for node in workflow.nodes])
        
        node_counts = Counter(all_node_types)
        analysis["node_frequency"] = dict(node_counts)
        
        # Identify popular and rare nodes
        total_workflows = collection.total_workflows
        for node_type, count in node_counts.items():
            frequency = count / total_workflows
            if frequency > 0.7:  # Used in >70% of workflows
                analysis["popular_nodes"].append(node_type)
            elif frequency < 0.1:  # Used in <10% of workflows
                analysis["rare_nodes"].append(node_type)
        
        # Analyze node combinations
        analysis["node_combinations"] = self._analyze_node_combinations(collection)
        
        # Analyze node sequences
        analysis["node_sequences"] = self._analyze_node_sequences(collection)
        
        # Categorize nodes
        analysis["node_categories"] = self._categorize_node_usage(collection)
        
        return analysis
    
    # Helper methods
    
    def _prepare_transactions(self, collection: WorkflowCollection) -> List[List[str]]:
        """Prepare transaction data for pattern mining."""
        transactions = []
        
        for workflow in collection.workflows:
            # Create transaction from node types
            transaction = [node.type for node in workflow.nodes]
            
            # Add workflow-level features
            if workflow.active:
                transaction.append("ACTIVE_WORKFLOW")
            
            if len(workflow.tags) > 0:
                transaction.extend([f"TAG_{tag.upper()}" for tag in workflow.tags])
            
            # Add structural features
            if len(workflow.nodes) > 10:
                transaction.append("LARGE_WORKFLOW")
            elif len(workflow.nodes) < 5:
                transaction.append("SMALL_WORKFLOW")
            
            if workflow.has_error_handling():
                transaction.append("HAS_ERROR_HANDLING")
            
            transactions.append(transaction)
        
        return transactions
    
    def _find_workflows_with_pattern(self, collection: WorkflowCollection, pattern: frozenset) -> List[str]:
        """Find workflows that contain a specific pattern."""
        matching_workflows = []
        
        for workflow in collection.workflows:
            workflow_items = set()
            
            # Add node types
            workflow_items.update([node.type for node in workflow.nodes])
            
            # Add workflow features
            if workflow.active:
                workflow_items.add("ACTIVE_WORKFLOW")
            
            workflow_items.update([f"TAG_{tag.upper()}" for tag in workflow.tags])
            
            if len(workflow.nodes) > 10:
                workflow_items.add("LARGE_WORKFLOW")
            elif len(workflow.nodes) < 5:
                workflow_items.add("SMALL_WORKFLOW")
            
            if workflow.has_error_handling():
                workflow_items.add("HAS_ERROR_HANDLING")
            
            # Check if pattern is subset of workflow items
            if pattern.issubset(workflow_items):
                matching_workflows.append(workflow.id or workflow.name)
        
        return matching_workflows
    
    def _extract_execution_sequence(self, workflow: N8nWorkflow) -> List[str]:
        """Extract the execution sequence from a workflow."""
        if not workflow.connections:
            return [node.type for node in workflow.nodes]
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for conn in workflow.connections:
            adjacency[conn.source_node].append(conn.target_node)
        
        # Find entry points (nodes with no incoming connections)
        all_nodes = {node.id for node in workflow.nodes}
        nodes_with_incoming = {conn.target_node for conn in workflow.connections}
        entry_points = all_nodes - nodes_with_incoming
        
        if not entry_points:
            # If no clear entry points, start from first node
            entry_points = {workflow.nodes[0].id} if workflow.nodes else set()
        
        # Perform DFS to get execution sequence
        sequence = []
        visited = set()
        
        def dfs(node_id):
            if node_id in visited:
                return
            
            visited.add(node_id)
            
            # Find node type
            node = workflow.get_node_by_id(node_id)
            if node:
                sequence.append(node.type)
            
            # Visit children
            for child in adjacency.get(node_id, []):
                dfs(child)
        
        for entry_point in entry_points:
            dfs(entry_point)
        
        return sequence
    
    def _find_common_subsequences(self, sequences: List[List[str]]) -> List[List[str]]:
        """Find common subsequences across multiple sequences."""
        if not sequences:
            return []
        
        common_subsequences = []
        min_length = 2
        max_length = 5
        
        # Generate all possible subsequences
        all_subsequences = defaultdict(int)
        
        for sequence in sequences:
            for length in range(min_length, min(len(sequence) + 1, max_length + 1)):
                for i in range(len(sequence) - length + 1):
                    subseq = tuple(sequence[i:i + length])
                    all_subsequences[subseq] += 1
        
        # Filter subsequences that appear in multiple workflows
        min_support = max(2, len(sequences) * 0.2)  # At least 20% of workflows
        
        for subseq, count in all_subsequences.items():
            if count >= min_support:
                common_subsequences.append(list(subseq))
        
        # Sort by frequency
        common_subsequences.sort(key=lambda x: all_subsequences[tuple(x)], reverse=True)
        
        return common_subsequences[:20]  # Return top 20
    
    def _extract_clustering_features(self, collection: WorkflowCollection) -> Tuple[np.ndarray, List[str]]:
        """Extract features for workflow clustering."""
        features = []
        workflow_ids = []
        
        for workflow in collection.workflows:
            feature_vector = []
            
            # Basic structural features
            feature_vector.append(len(workflow.nodes))
            feature_vector.append(len(workflow.connections))
            feature_vector.append(len(workflow.node_types))
            feature_vector.append(workflow.complexity_score)
            
            # Node type features (one-hot encoding for common types)
            common_types = [
                'n8n-nodes-base.httpRequest',
                'n8n-nodes-base.webhook',
                'n8n-nodes-base.if',
                'n8n-nodes-base.function',
                'n8n-nodes-base.set',
                'n8n-nodes-base.json'
            ]
            
            for node_type in common_types:
                feature_vector.append(1 if node_type in workflow.node_types else 0)
            
            # Workflow characteristics
            feature_vector.append(1 if workflow.active else 0)
            feature_vector.append(len(workflow.tags))
            feature_vector.append(1 if workflow.has_error_handling() else 0)
            
            features.append(feature_vector)
            workflow_ids.append(workflow.id or workflow.name)
        
        return np.array(features), workflow_ids
    
    def _determine_optimal_clusters(self, features: np.ndarray) -> int:
        """Determine optimal number of clusters using silhouette analysis."""
        max_clusters = min(10, features.shape[0] - 1)
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            score = silhouette_score(features, cluster_labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def _perform_clustering(self, features: np.ndarray, workflow_ids: List[str], n_clusters: int) -> List[WorkflowCluster]:
        """Perform clustering using K-means algorithm."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_workflows = [
                workflow_ids[i] for i, label in enumerate(cluster_labels) 
                if label == cluster_id
            ]
            
            cluster = WorkflowCluster(
                cluster_id=cluster_id,
                workflows=cluster_workflows,
                centroid=kmeans.cluster_centers_[cluster_id]
            )
            clusters.append(cluster)
        
        return clusters
    
    def _analyze_cluster_characteristics(self, collection: WorkflowCollection, workflow_ids: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of workflows in a cluster."""
        cluster_workflows = [
            workflow for workflow in collection.workflows 
            if (workflow.id or workflow.name) in workflow_ids
        ]
        
        if not cluster_workflows:
            return {}
        
        characteristics = {}
        
        # Average metrics
        characteristics["avg_nodes"] = np.mean([len(w.nodes) for w in cluster_workflows])
        characteristics["avg_connections"] = np.mean([len(w.connections) for w in cluster_workflows])
        characteristics["avg_complexity"] = np.mean([w.complexity_score for w in cluster_workflows])
        
        # Common node types
        all_node_types = []
        for workflow in cluster_workflows:
            all_node_types.extend([node.type for node in workflow.nodes])
        
        node_counts = Counter(all_node_types)
        characteristics["common_node_types"] = dict(node_counts.most_common(5))
        
        # Common tags
        all_tags = []
        for workflow in cluster_workflows:
            all_tags.extend(workflow.tags)
        
        if all_tags:
            tag_counts = Counter(all_tags)
            characteristics["common_tags"] = dict(tag_counts.most_common(3))
        
        # Activity status
        active_count = sum(1 for w in cluster_workflows if w.active)
        characteristics["active_percentage"] = (active_count / len(cluster_workflows)) * 100
        
        return characteristics
    
    def _group_by_structure(self, collection: WorkflowCollection) -> Dict[str, List[N8nWorkflow]]:
        """Group workflows by structural similarity."""
        structure_groups = defaultdict(list)
        
        for workflow in collection.workflows:
            # Create a structural signature
            signature = self._create_structural_signature(workflow)
            structure_groups[signature].append(workflow)
        
        return dict(structure_groups)
    
    def _create_structural_signature(self, workflow: N8nWorkflow) -> str:
        """Create a structural signature for a workflow."""
        # Sort node types to create consistent signature
        node_types = sorted([node.type for node in workflow.nodes])
        
        # Create signature from node types and basic structure
        signature_parts = [
            f"nodes_{len(workflow.nodes)}",
            f"connections_{len(workflow.connections)}",
            f"types_{'_'.join(node_types[:5])}"  # Use first 5 types
        ]
        
        return "|".join(signature_parts)
    
    def _create_template_from_group(self, workflows: List[N8nWorkflow]) -> Dict[str, Any]:
        """Create a template from a group of similar workflows."""
        template = {
            "template_id": f"template_{hash(tuple(w.id or w.name for w in workflows)) % 10000}",
            "workflow_count": len(workflows),
            "workflows": [w.id or w.name for w in workflows],
            "common_structure": {},
            "variations": {}
        }
        
        # Analyze common structure
        all_node_types = []
        for workflow in workflows:
            all_node_types.append([node.type for node in workflow.nodes])
        
        # Find common node types
        common_types = set(all_node_types[0])
        for node_list in all_node_types[1:]:
            common_types &= set(node_list)
        
        template["common_structure"]["node_types"] = list(common_types)
        template["common_structure"]["avg_nodes"] = np.mean([len(w.nodes) for w in workflows])
        template["common_structure"]["avg_connections"] = np.mean([len(w.connections) for w in workflows])
        
        return template
    
    def _analyze_node_combinations(self, collection: WorkflowCollection) -> Dict[str, int]:
        """Analyze common node type combinations."""
        combinations = defaultdict(int)
        
        for workflow in collection.workflows:
            node_types = [node.type for node in workflow.nodes]
            
            # Generate pairs
            for i in range(len(node_types)):
                for j in range(i + 1, len(node_types)):
                    pair = tuple(sorted([node_types[i], node_types[j]]))
                    combinations[f"{pair[0]} + {pair[1]}"] += 1
        
        # Return top combinations
        return dict(Counter(combinations).most_common(10))
    
    def _analyze_node_sequences(self, collection: WorkflowCollection) -> Dict[str, int]:
        """Analyze common node sequences."""
        sequences = defaultdict(int)
        
        for workflow in collection.workflows:
            sequence = self._extract_execution_sequence(workflow)
            
            # Generate subsequences of length 2-3
            for length in [2, 3]:
                for i in range(len(sequence) - length + 1):
                    subseq = " -> ".join(sequence[i:i + length])
                    sequences[subseq] += 1
        
        # Return top sequences
        return dict(Counter(sequences).most_common(10))
    
    def _categorize_node_usage(self, collection: WorkflowCollection) -> Dict[str, List[str]]:
        """Categorize nodes by their usage patterns."""
        categories = {
            "triggers": [],
            "actions": [],
            "transformations": [],
            "conditions": [],
            "integrations": []
        }
        
        # Count usage of each node type
        node_counts = Counter()
        for workflow in collection.workflows:
            for node in workflow.nodes:
                node_counts[node.type] += 1
        
        # Categorize based on type and usage
        for node_type, count in node_counts.items():
            node_type_lower = node_type.lower()
            
            if any(trigger in node_type_lower for trigger in ["trigger", "webhook", "cron", "start"]):
                categories["triggers"].append(f"{node_type} ({count})")
            elif any(action in node_type_lower for action in ["send", "post", "create", "update", "delete"]):
                categories["actions"].append(f"{node_type} ({count})")
            elif any(transform in node_type_lower for transform in ["function", "set", "json", "xml"]):
                categories["transformations"].append(f"{node_type} ({count})")
            elif any(condition in node_type_lower for condition in ["if", "switch", "filter"]):
                categories["conditions"].append(f"{node_type} ({count})")
            else:
                categories["integrations"].append(f"{node_type} ({count})")
        
        return categories

