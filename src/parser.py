"""
JSON parser for n8n workflows.

This module provides functionality to parse n8n workflow JSON files
and convert them into our internal data models.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

try:
    from .models import (
        N8nWorkflow, WorkflowNode, WorkflowConnection, 
        WorkflowSettings, WorkflowCollection
    )
except ImportError:
    from models import (
        N8nWorkflow, WorkflowNode, WorkflowConnection, 
        WorkflowSettings, WorkflowCollection
    )


logger = logging.getLogger(__name__)


class N8nWorkflowParser:
    """Parser for n8n workflow JSON files."""
    
    def __init__(self):
        """Initialize the parser."""
        self.errors = []
        self.warnings = []
    
    def parse_workflow_file(self, file_path: Union[str, Path]) -> Optional[N8nWorkflow]:
        """
        Parse a single n8n workflow JSON file.
        
        Args:
            file_path: Path to the workflow JSON file
            
        Returns:
            Parsed workflow object or None if parsing failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            return self.parse_workflow_dict(workflow_data)
            
        except FileNotFoundError:
            error_msg = f"Workflow file not found: {file_path}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in workflow file {file_path}: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
        except Exception as e:
            error_msg = f"Error parsing workflow file {file_path}: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
    
    def parse_workflow_dict(self, workflow_data: Dict[str, Any]) -> Optional[N8nWorkflow]:
        """
        Parse a workflow from a dictionary (JSON object).
        
        Args:
            workflow_data: Dictionary containing workflow data
            
        Returns:
            Parsed workflow object or None if parsing failed
        """
        try:
            # Extract basic workflow information
            workflow_id = workflow_data.get('id')
            name = workflow_data.get('name', 'Unnamed Workflow')
            active = workflow_data.get('active', False)
            tags = workflow_data.get('tags', [])
            
            # Parse nodes
            nodes_data = workflow_data.get('nodes', [])
            nodes = []
            for node_data in nodes_data:
                node = self._parse_node(node_data)
                if node:
                    nodes.append(node)
            
            # Parse connections
            connections_data = workflow_data.get('connections', {})
            connections = self._parse_connections(connections_data)
            
            # Parse settings
            settings_data = workflow_data.get('settings', {})
            settings = self._parse_settings(settings_data)
            
            # Parse other optional fields
            static_data = workflow_data.get('staticData', {})
            pin_data = workflow_data.get('pinData', {})
            version_id = workflow_data.get('versionId')
            
            # Parse timestamps if available
            created_at = self._parse_timestamp(workflow_data.get('createdAt'))
            updated_at = self._parse_timestamp(workflow_data.get('updatedAt'))
            
            # Create workflow object
            workflow = N8nWorkflow(
                id=workflow_id,
                name=name,
                nodes=nodes,
                connections=connections,
                active=active,
                settings=settings,
                static_data=static_data,
                tags=tags,
                pin_data=pin_data,
                version_id=version_id,
                created_at=created_at,
                updated_at=updated_at
            )
            
            return workflow
            
        except Exception as e:
            error_msg = f"Error parsing workflow data: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
    
    def _parse_node(self, node_data: Dict[str, Any]) -> Optional[WorkflowNode]:
        """Parse a single node from JSON data."""
        try:
            node_id = node_data.get('id')
            if not node_id:
                self.warnings.append("Node missing ID, skipping")
                return None
            
            name = node_data.get('name', '')
            node_type = node_data.get('type', '')
            type_version = node_data.get('typeVersion')
            
            # Parse position
            position_data = node_data.get('position', [0, 0])
            if isinstance(position_data, list) and len(position_data) >= 2:
                position = {"x": position_data[0], "y": position_data[1]}
            else:
                position = {"x": 0, "y": 0}
            
            # Parse parameters
            parameters = node_data.get('parameters', {})
            
            # Parse credentials
            credentials = node_data.get('credentials', {})
            
            # Parse other optional fields
            webhook_id = node_data.get('webhookId')
            disabled = node_data.get('disabled', False)
            notes = node_data.get('notes')
            
            return WorkflowNode(
                id=node_id,
                name=name,
                type=node_type,
                type_version=type_version,
                position=position,
                parameters=parameters,
                credentials=credentials,
                webhook_id=webhook_id,
                disabled=disabled,
                notes=notes
            )
            
        except Exception as e:
            warning_msg = f"Error parsing node: {e}"
            logger.warning(warning_msg)
            self.warnings.append(warning_msg)
            return None
    
    def _parse_connections(self, connections_data: Dict[str, Any]) -> List[WorkflowConnection]:
        """Parse connections from JSON data."""
        connections = []
        
        try:
            for source_node, outputs in connections_data.items():
                for output_name, output_connections in outputs.items():
                    if isinstance(output_connections, list):
                        for i, connection_list in enumerate(output_connections):
                            if isinstance(connection_list, list):
                                for connection in connection_list:
                                    conn = self._parse_single_connection(
                                        source_node, output_name, i, connection
                                    )
                                    if conn:
                                        connections.append(conn)
            
        except Exception as e:
            warning_msg = f"Error parsing connections: {e}"
            logger.warning(warning_msg)
            self.warnings.append(warning_msg)
        
        return connections
    
    def _parse_single_connection(
        self, 
        source_node: str, 
        output_name: str, 
        output_index: int, 
        connection_data: Dict[str, Any]
    ) -> Optional[WorkflowConnection]:
        """Parse a single connection from JSON data."""
        try:
            target_node = connection_data.get('node')
            target_input = connection_data.get('type', 'main')
            target_input_index = connection_data.get('index', 0)
            
            if not target_node:
                self.warnings.append("Connection missing target node")
                return None
            
            return WorkflowConnection(
                source_node=source_node,
                target_node=target_node,
                source_output=output_name,
                target_input=target_input,
                source_output_index=output_index,
                target_input_index=target_input_index
            )
            
        except Exception as e:
            warning_msg = f"Error parsing single connection: {e}"
            logger.warning(warning_msg)
            self.warnings.append(warning_msg)
            return None
    
    def _parse_settings(self, settings_data: Dict[str, Any]) -> WorkflowSettings:
        """Parse workflow settings from JSON data."""
        return WorkflowSettings(
            timezone=settings_data.get('timezone'),
            save_manual_executions=settings_data.get('saveManualExecutions', True),
            save_execution_progress=settings_data.get('saveExecutionProgress', False),
            save_data_error_execution=settings_data.get('saveDataErrorExecution', 'all'),
            save_data_success_execution=settings_data.get('saveDataSuccessExecution', 'all'),
            execution_timeout=settings_data.get('executionTimeout', -1),
            caller_policy=settings_data.get('callerPolicy', 'workflowsFromSameOwner')
        )
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse a timestamp string into a datetime object."""
        if not timestamp_str:
            return None
        
        try:
            # Try different timestamp formats
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # If none of the formats work, try parsing as ISO format
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
        except Exception as e:
            warning_msg = f"Could not parse timestamp '{timestamp_str}': {e}"
            logger.warning(warning_msg)
            self.warnings.append(warning_msg)
            return None
    
    def parse_workflow_collection(self, directory_path: Union[str, Path]) -> WorkflowCollection:
        """
        Parse all workflow JSON files in a directory.
        
        Args:
            directory_path: Path to directory containing workflow JSON files
            
        Returns:
            Collection of parsed workflows
        """
        directory = Path(directory_path)
        workflows = []
        
        if not directory.exists():
            error_msg = f"Directory not found: {directory_path}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return WorkflowCollection(workflows=workflows, collection_name="Empty Collection")
        
        # Find all JSON files in the directory
        json_files = list(directory.glob("*.json"))
        
        logger.info(f"Found {len(json_files)} JSON files in {directory_path}")
        
        for json_file in json_files:
            workflow = self.parse_workflow_file(json_file)
            if workflow:
                workflows.append(workflow)
                logger.info(f"Successfully parsed workflow: {workflow.name}")
            else:
                logger.warning(f"Failed to parse workflow file: {json_file}")
        
        collection_name = f"Collection from {directory.name}"
        return WorkflowCollection(workflows=workflows, collection_name=collection_name)
    
    def get_parsing_summary(self) -> Dict[str, Any]:
        """Get a summary of parsing results including errors and warnings."""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }
    
    def clear_messages(self) -> None:
        """Clear all error and warning messages."""
        self.errors.clear()
        self.warnings.clear()


def create_sample_workflow() -> N8nWorkflow:
    """Create a sample n8n workflow for testing purposes."""
    
    # Create sample nodes
    nodes = [
        WorkflowNode(
            id="start",
            name="Start",
            type="n8n-nodes-base.start",
            position={"x": 100, "y": 100}
        ),
        WorkflowNode(
            id="http_request",
            name="HTTP Request",
            type="n8n-nodes-base.httpRequest",
            position={"x": 300, "y": 100},
            parameters={
                "url": "https://api.example.com/data",
                "method": "GET"
            }
        ),
        WorkflowNode(
            id="json_parse",
            name="JSON Parse",
            type="n8n-nodes-base.json",
            position={"x": 500, "y": 100},
            parameters={
                "operation": "parse"
            }
        ),
        WorkflowNode(
            id="set_data",
            name="Set Data",
            type="n8n-nodes-base.set",
            position={"x": 700, "y": 100},
            parameters={
                "values": {
                    "string": [
                        {
                            "name": "processed",
                            "value": "true"
                        }
                    ]
                }
            }
        )
    ]
    
    # Create sample connections
    connections = [
        WorkflowConnection("start", "http_request"),
        WorkflowConnection("http_request", "json_parse"),
        WorkflowConnection("json_parse", "set_data")
    ]
    
    # Create workflow
    workflow = N8nWorkflow(
        id="sample_workflow_001",
        name="Sample Data Processing Workflow",
        nodes=nodes,
        connections=connections,
        active=True,
        tags=["sample", "data-processing", "api"]
    )
    
    return workflow

