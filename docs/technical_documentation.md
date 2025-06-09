# n8n Workflow Analyzer - Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Models](#data-models)
4. [Workflow Analysis Engine](#workflow-analysis-engine)
5. [Pattern Mining Engine](#pattern-mining-engine)
6. [Network Analysis Engine](#network-analysis-engine)
7. [Visualization Engine](#visualization-engine)
8. [API Server](#api-server)
9. [Web Interface](#web-interface)
10. [Performance Considerations](#performance-considerations)
11. [Security Considerations](#security-considerations)
12. [Extending the System](#extending-the-system)
13. [Deployment Guide](#deployment-guide)
14. [API Reference](#api-reference)

## Architecture Overview

The n8n Workflow Analyzer is built with a modular architecture that separates concerns and allows for independent development and scaling of components. The system follows a layered architecture pattern with the following main layers:

1. **Data Layer**: Handles workflow data parsing, storage, and retrieval
2. **Analysis Layer**: Implements various analysis algorithms
3. **Visualization Layer**: Generates interactive visualizations
4. **API Layer**: Provides RESTful API access to system capabilities
5. **Presentation Layer**: Implements the web-based user interface

### System Components

The system consists of the following main components:

- **Core Analysis Engine**: Python modules for workflow parsing, analysis, pattern mining, and network analysis
- **Visualization Engine**: Python modules for generating interactive visualizations
- **API Server**: Flask-based RESTful API for accessing analysis capabilities
- **Web Interface**: React-based user interface for interacting with the system

### Component Interactions

The components interact through well-defined interfaces:

1. The **Web Interface** communicates with the **API Server** through HTTP requests
2. The **API Server** coordinates analysis tasks and returns results
3. The **Core Analysis Engine** processes workflows and generates analysis results
4. The **Visualization Engine** transforms analysis results into interactive visualizations

## Core Components

### Workflow Parser

The workflow parser is responsible for reading n8n workflow JSON files and converting them into internal data structures for analysis. It handles:

- JSON parsing and validation
- Data structure conversion
- Error handling and reporting

Implementation: `src/parser.py`

### Configuration Manager

The configuration manager handles system-wide and analysis-specific configuration settings. It provides:

- Default configuration values
- Configuration validation
- Configuration persistence

Implementation: `src/config.py`

### Data Storage

The data storage component manages persistent storage of:

- Uploaded workflow files
- Analysis results
- Generated visualizations
- User sessions and preferences

Implementation: Various modules in `src/`

## Data Models

The n8n Workflow Analyzer uses several data models to represent workflows and analysis results:

### N8nWorkflow

Represents a single n8n workflow with its nodes, connections, and settings.

```python
class N8nWorkflow:
    id: str
    name: str
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    settings: WorkflowSettings
    version: int
    active: bool
```

### WorkflowNode

Represents a single node in an n8n workflow.

```python
class WorkflowNode:
    id: str
    name: str
    type: str
    position: Dict[str, float]
    parameters: Dict[str, Any]
```

### WorkflowConnection

Represents a connection between two nodes in an n8n workflow.

```python
class WorkflowConnection:
    source_node: str
    target_node: str
```

### WorkflowCollection

Represents a collection of n8n workflows for batch analysis.

```python
class WorkflowCollection:
    workflows: List[N8nWorkflow]
    total_workflows: int
```

### WorkflowAnalysisResult

Represents the result of analyzing a workflow.

```python
class WorkflowAnalysisResult:
    workflow_id: str
    metrics: Dict[str, Any]
    patterns: List[Dict[str, Any]]
    recommendations: List[str]
```

Implementation: `src/models.py`

## Workflow Analysis Engine

The workflow analysis engine is responsible for analyzing individual workflows and collections of workflows. It calculates various metrics and identifies patterns and anomalies.

### Analysis Metrics

The engine calculates several metrics for each workflow:

- **Structural Complexity**: Based on the number of nodes and connections
- **Cognitive Complexity**: Estimates the cognitive load required to understand the workflow
- **Cyclomatic Complexity**: Counts the number of linearly independent paths
- **Error Handling Coverage**: Percentage of nodes with error handling
- **Branching Factor**: Average number of outgoing connections per node

### Analysis Process

The analysis process consists of the following steps:

1. Parse workflow JSON into internal data structures
2. Calculate basic metrics (node count, connection count, etc.)
3. Analyze workflow structure (entry points, exit points, branches)
4. Calculate complexity metrics
5. Identify patterns and anti-patterns
6. Generate recommendations
7. Return analysis results

Implementation: `src/analysis/workflow_analyzer.py`

## Pattern Mining Engine

The pattern mining engine discovers frequent patterns and association rules across collections of workflows. It uses algorithms from the field of frequent pattern mining.

### Frequent Pattern Mining

The engine implements the FP-Growth algorithm for discovering frequent patterns:

1. Convert workflows to transaction format
2. Build FP-Tree data structure
3. Mine frequent patterns from the FP-Tree
4. Filter and rank patterns by support

### Association Rule Mining

The engine generates association rules from frequent patterns:

1. For each frequent pattern, generate all possible rules
2. Calculate confidence, lift, and other metrics for each rule
3. Filter rules based on minimum confidence and lift thresholds
4. Rank rules by confidence and lift

Implementation: `src/mining/pattern_miner.py`

## Network Analysis Engine

The network analysis engine treats workflows as networks of connected nodes and applies network analysis techniques to extract insights.

### Network Construction

The engine constructs network representations of workflows:

1. Convert workflows to NetworkX graph objects
2. Nodes represent workflow nodes
3. Edges represent connections between nodes
4. Node attributes include node type, name, and parameters
5. Edge attributes include connection type and metadata

### Network Metrics

The engine calculates several network metrics:

- **Density**: Ratio of actual connections to possible connections
- **Average Degree**: Average number of connections per node
- **Diameter**: Longest shortest path between any two nodes
- **Average Path Length**: Average shortest path between all pairs of nodes
- **Clustering Coefficient**: Degree to which nodes tend to cluster together

### Community Detection

The engine identifies communities or modules within workflow networks:

1. Apply community detection algorithms (Louvain, Girvan-Newman)
2. Identify densely connected groups of nodes
3. Calculate modularity and other community metrics
4. Visualize community structure

Implementation: `src/network/network_analyzer.py`

## Visualization Engine

The visualization engine generates interactive visualizations of workflows, analysis results, and patterns.

### Visualization Types

The engine supports several types of visualizations:

- **Workflow Diagrams**: Visual representations of individual workflows
- **Complexity Charts**: Visualizations of complexity metrics
- **Pattern Networks**: Networks of frequent patterns and associations
- **Workflow Networks**: Network visualizations of workflow collections
- **Statistical Dashboards**: Comprehensive dashboards with multiple charts

### Visualization Technologies

The engine uses several technologies for visualization:

- **Plotly**: For interactive charts and graphs
- **NetworkX + Plotly**: For network visualizations
- **Matplotlib**: For static visualizations
- **HTML + JavaScript**: For interactive dashboards

Implementation: `src/visualization/visualization_manager.py`

## API Server

The API server provides RESTful API access to the n8n Workflow Analyzer capabilities. It is built with Flask and follows RESTful API design principles.

### API Endpoints

The API provides the following main endpoints:

- `POST /api/upload`: Upload workflow files
- `GET /api/workflows`: List uploaded workflows
- `GET /api/workflow/{id}`: Get workflow details
- `POST /api/analyze/{id}`: Analyze a specific workflow
- `GET /api/results/{id}`: Get analysis results
- `GET /api/visualizations/{id}/{filename}`: Get visualization files

### Authentication and Authorization

The API implements token-based authentication:

1. Clients authenticate with username/password to obtain a token
2. Subsequent requests include the token in the Authorization header
3. The API validates the token for each request
4. Access control is enforced based on user roles and permissions

### Error Handling

The API implements consistent error handling:

1. All errors return appropriate HTTP status codes
2. Error responses include detailed error messages
3. Validation errors include field-specific error information
4. Internal errors are logged for debugging

Implementation: `api_server/src/main.py` and related files

## Web Interface

The web interface provides a user-friendly way to interact with the n8n Workflow Analyzer. It is built with React and follows modern web application design principles.

### User Interface Components

The interface consists of several main components:

- **Navigation**: Sidebar navigation menu
- **Upload**: File upload interface
- **Sessions**: Session management interface
- **Analysis**: Analysis configuration and results interface
- **Visualizations**: Interactive visualization gallery
- **Settings**: User and system settings

### State Management

The interface uses React's state management capabilities:

1. Component-level state for UI-specific state
2. Context API for shared state
3. API client for data fetching and mutation

### Responsive Design

The interface is designed to work on various devices:

1. Responsive layout adapts to screen size
2. Mobile-friendly touch interactions
3. Accessible design for all users

Implementation: `web_ui/src/` and related files

## Performance Considerations

The n8n Workflow Analyzer is designed to handle large collections of workflows efficiently. Several performance optimizations are implemented:

### Data Processing Optimizations

- **Lazy Loading**: Load and process data only when needed
- **Incremental Processing**: Process workflows incrementally
- **Caching**: Cache intermediate results to avoid redundant computation
- **Batch Processing**: Process workflows in batches for better resource utilization

### Algorithmic Optimizations

- **Efficient Algorithms**: Use efficient algorithms for pattern mining and network analysis
- **Early Termination**: Terminate processing early when possible
- **Approximation**: Use approximation algorithms for large datasets
- **Parallelization**: Parallelize computation where possible

### Resource Management

- **Memory Management**: Limit memory usage for large datasets
- **Timeout Handling**: Implement timeouts for long-running operations
- **Resource Limits**: Enforce limits on resource usage

## Security Considerations

The n8n Workflow Analyzer implements several security measures:

### Data Security

- **Encryption**: Encrypt sensitive data at rest and in transit
- **Access Control**: Implement fine-grained access control
- **Data Isolation**: Isolate data between users and sessions

### API Security

- **Authentication**: Require authentication for all API requests
- **Authorization**: Enforce authorization for sensitive operations
- **Rate Limiting**: Implement rate limiting to prevent abuse
- **Input Validation**: Validate all input data to prevent injection attacks

### Web Security

- **HTTPS**: Require HTTPS for all communication
- **CSRF Protection**: Implement CSRF protection
- **Content Security Policy**: Implement CSP to prevent XSS attacks
- **Secure Cookies**: Use secure and HttpOnly cookies

## Extending the System

The n8n Workflow Analyzer is designed to be extensible. Several extension points are provided:

### Adding New Analysis Metrics

To add a new analysis metric:

1. Implement the metric calculation in `src/analysis/workflow_analyzer.py`
2. Update the `WorkflowAnalysisResult` class to include the new metric
3. Update the API to expose the new metric
4. Update the web interface to display the new metric

### Adding New Visualization Types

To add a new visualization type:

1. Implement the visualization generation in `src/visualization/visualization_manager.py`
2. Update the API to expose the new visualization
3. Update the web interface to display the new visualization

### Adding New API Endpoints

To add a new API endpoint:

1. Implement the endpoint in the appropriate route file in `api_server/src/routes/`
2. Update the API documentation
3. Update the web interface to use the new endpoint

## Deployment Guide

The n8n Workflow Analyzer can be deployed in various environments:

### Local Deployment

For local development and testing:

1. Clone the repository
2. Install dependencies
3. Start the API server and web interface
4. Access the application at `http://localhost:5173`

### Docker Deployment

For containerized deployment:

1. Build the Docker images:
   ```bash
   docker-compose build
   ```

2. Start the containers:
   ```bash
   docker-compose up -d
   ```

3. Access the application at `http://localhost:8080`

### Cloud Deployment

For cloud deployment (e.g., AWS, Azure, GCP):

1. Set up cloud infrastructure (VMs, containers, databases)
2. Deploy the API server and web interface
3. Configure networking and security
4. Set up monitoring and logging

## API Reference

### Authentication

#### Login

```
POST /api/auth/login
```

Request:
```json
{
  "username": "string",
  "password": "string"
}
```

Response:
```json
{
  "token": "string",
  "expires_at": "string"
}
```

### Workflows

#### Upload Workflow

```
POST /api/upload
```

Request:
```
Content-Type: multipart/form-data
file: <workflow.json>
```

Response:
```json
{
  "workflow_id": "string",
  "name": "string",
  "status": "string"
}
```

#### List Workflows

```
GET /api/workflows
```

Response:
```json
{
  "workflows": [
    {
      "id": "string",
      "name": "string",
      "node_count": "integer",
      "connection_count": "integer"
    }
  ]
}
```

#### Get Workflow

```
GET /api/workflow/{id}
```

Response:
```json
{
  "id": "string",
  "name": "string",
  "nodes": [
    {
      "id": "string",
      "name": "string",
      "type": "string"
    }
  ],
  "connections": [
    {
      "source": "string",
      "target": "string"
    }
  ]
}
```

### Analysis

#### Analyze Workflow

```
POST /api/analyze/{id}
```

Request:
```json
{
  "analysis_type": "string",
  "parameters": {
    "key": "value"
  }
}
```

Response:
```json
{
  "analysis_id": "string",
  "status": "string",
  "estimated_time": "integer"
}
```

#### Get Analysis Results

```
GET /api/results/{id}
```

Response:
```json
{
  "workflow_id": "string",
  "metrics": {
    "complexity": {
      "structural_complexity": "number",
      "cognitive_complexity": "number",
      "cyclomatic_complexity": "number"
    },
    "structure": {
      "error_handling_coverage": "number",
      "branching_factor": "number"
    }
  },
  "patterns": [
    {
      "pattern": "string",
      "support": "number"
    }
  ],
  "recommendations": [
    "string"
  ],
  "visualizations": {
    "workflow_diagram": "string",
    "complexity_chart": "string"
  }
}
```

### Visualizations

#### Get Visualization

```
GET /api/visualizations/{id}/{filename}
```

Response:
```
Content-Type: image/png or text/html
<visualization content>
```

