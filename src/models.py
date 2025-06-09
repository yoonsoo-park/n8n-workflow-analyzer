"""
Core data models for n8n workflow analysis.

This module defines the data structures used to represent n8n workflows,
nodes, connections, and analysis results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json


@dataclass
class WorkflowNode:
    """Represents a single node in an n8n workflow."""
    
    id: str
    name: str
    type: str
    type_version: Optional[float] = None
    position: Optional[Dict[str, float]] = None
    parameters: Optional[Dict[str, Any]] = None
    credentials: Optional[Dict[str, Any]] = None
    webhook_id: Optional[str] = None
    disabled: bool = False
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.position is None:
            self.position = {"x": 0, "y": 0}
        if self.parameters is None:
            self.parameters = {}
        if self.credentials is None:
            self.credentials = {}


@dataclass
class WorkflowConnection:
    """Represents a connection between two nodes in an n8n workflow."""
    
    source_node: str
    target_node: str
    source_output: str = "main"
    target_input: str = "main"
    source_output_index: int = 0
    target_input_index: int = 0
    
    def __str__(self) -> str:
        return f"{self.source_node}[{self.source_output}:{self.source_output_index}] -> {self.target_node}[{self.target_input}:{self.target_input_index}]"


@dataclass
class WorkflowSettings:
    """Represents workflow-level settings and configuration."""
    
    timezone: Optional[str] = None
    save_manual_executions: bool = True
    save_execution_progress: bool = False
    save_data_error_execution: str = "all"
    save_data_success_execution: str = "all"
    execution_timeout: int = -1
    caller_policy: str = "workflowsFromSameOwner"


@dataclass
class N8nWorkflow:
    """Represents a complete n8n workflow."""
    
    id: Optional[str]
    name: str
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    active: bool = False
    settings: Optional[WorkflowSettings] = None
    static_data: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    pin_data: Optional[Dict[str, Any]] = None
    version_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.settings is None:
            self.settings = WorkflowSettings()
        if self.static_data is None:
            self.static_data = {}
    
    @property
    def node_count(self) -> int:
        """Return the number of nodes in the workflow."""
        return len(self.nodes)
    
    @property
    def connection_count(self) -> int:
        """Return the number of connections in the workflow."""
        return len(self.connections)
    
    @property
    def node_types(self) -> Set[str]:
        """Return a set of all node types used in the workflow."""
        return {node.type for node in self.nodes}
    
    @property
    def complexity_score(self) -> float:
        """Calculate a basic complexity score for the workflow."""
        # Simple complexity metric based on nodes, connections, and branching
        base_score = self.node_count + self.connection_count
        
        # Add complexity for multiple outputs/inputs
        branching_factor = 0
        for node in self.nodes:
            outgoing_connections = [c for c in self.connections if c.source_node == node.id]
            if len(outgoing_connections) > 1:
                branching_factor += len(outgoing_connections) - 1
        
        return base_score + (branching_factor * 0.5)
    
    def get_node_by_id(self, node_id: str) -> Optional[WorkflowNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_connections_from_node(self, node_id: str) -> List[WorkflowConnection]:
        """Get all connections originating from a specific node."""
        return [conn for conn in self.connections if conn.source_node == node_id]
    
    def get_connections_to_node(self, node_id: str) -> List[WorkflowConnection]:
        """Get all connections targeting a specific node."""
        return [conn for conn in self.connections if conn.target_node == node_id]
    
    def has_error_handling(self) -> bool:
        """Check if the workflow has any error handling mechanisms."""
        # Look for error handling patterns
        for node in self.nodes:
            # Check for error workflow settings
            if node.parameters and node.parameters.get("continueOnFail", False):
                return True
            # Check for error handling nodes
            if "error" in node.type.lower() or "catch" in node.type.lower():
                return True
        return False


@dataclass
class WorkflowAnalysisResult:
    """Container for workflow analysis results."""
    
    workflow_id: str
    workflow_name: str
    analysis_timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    performance_data: Optional[Dict[str, Any]] = None
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the analysis results."""
        self.metrics[name] = value
    
    def add_pattern(self, pattern: str) -> None:
        """Add a discovered pattern to the results."""
        self.patterns.append(pattern)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to the results."""
        self.recommendations.append(recommendation)


@dataclass
class WorkflowCollection:
    """Container for multiple workflows and collection-level analysis."""
    
    workflows: List[N8nWorkflow]
    collection_name: str = "Unnamed Collection"
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def total_workflows(self) -> int:
        """Return the total number of workflows in the collection."""
        return len(self.workflows)
    
    @property
    def total_nodes(self) -> int:
        """Return the total number of nodes across all workflows."""
        return sum(workflow.node_count for workflow in self.workflows)
    
    @property
    def total_connections(self) -> int:
        """Return the total number of connections across all workflows."""
        return sum(workflow.connection_count for workflow in self.workflows)
    
    @property
    def all_node_types(self) -> Set[str]:
        """Return all unique node types used across all workflows."""
        all_types = set()
        for workflow in self.workflows:
            all_types.update(workflow.node_types)
        return all_types
    
    def get_workflow_by_id(self, workflow_id: str) -> Optional[N8nWorkflow]:
        """Get a workflow by its ID."""
        for workflow in self.workflows:
            if workflow.id == workflow_id:
                return workflow
        return None
    
    def get_workflows_by_tag(self, tag: str) -> List[N8nWorkflow]:
        """Get all workflows that have a specific tag."""
        return [workflow for workflow in self.workflows if tag in workflow.tags]
    
    def get_workflows_using_node_type(self, node_type: str) -> List[N8nWorkflow]:
        """Get all workflows that use a specific node type."""
        return [workflow for workflow in self.workflows if node_type in workflow.node_types]

