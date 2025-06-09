# n8n Workflow Analyzer - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Uploading Workflows](#uploading-workflows)
4. [Analyzing Workflows](#analyzing-workflows)
5. [Understanding Analysis Results](#understanding-analysis-results)
6. [Pattern Mining](#pattern-mining)
7. [Network Analysis](#network-analysis)
8. [Visualizations](#visualizations)
9. [Exporting Results](#exporting-results)
10. [API Integration](#api-integration)
11. [Troubleshooting](#troubleshooting)
12. [Glossary](#glossary)

## Introduction

The n8n Workflow Analyzer is a comprehensive tool designed to help n8n users and administrators gain deeper insights into their workflow collections. This user guide provides detailed instructions on how to use the analyzer effectively to extract valuable information from your n8n workflows.

### What is n8n?

n8n is a fair-code licensed workflow automation tool that enables you to connect different services and applications through a visual interface. It allows you to create complex automation workflows without writing code, making it accessible to both technical and non-technical users.

### Why Analyze n8n Workflows?

As organizations build more workflows in n8n, several challenges emerge:

- **Complexity Management**: Workflows can become complex and difficult to understand
- **Standardization**: Ensuring consistent patterns and best practices across workflows
- **Optimization**: Identifying inefficient or redundant workflows
- **Governance**: Managing and monitoring workflow deployments
- **Knowledge Transfer**: Documenting and sharing workflow knowledge

The n8n Workflow Analyzer addresses these challenges by providing tools to analyze, visualize, and understand your workflow collections.

## Getting Started

### System Requirements

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection
- n8n workflow JSON files

### Accessing the Application

The n8n Workflow Analyzer is available as a web application. You can access it by:

1. Opening your web browser
2. Navigating to the application URL (e.g., `http://localhost:5173` for local development)
3. Logging in with your credentials (if required)

### Interface Overview

The application interface consists of several key areas:

- **Navigation Menu**: Access different sections of the application
- **Upload Area**: Upload workflow files for analysis
- **Sessions List**: View and manage analysis sessions
- **Analysis Dashboard**: View and explore analysis results
- **Visualization Gallery**: Access interactive visualizations

## Uploading Workflows

### Supported File Formats

The n8n Workflow Analyzer supports the following file formats:

- Individual n8n workflow JSON files
- ZIP archives containing multiple workflow JSON files
- n8n export files (containing multiple workflows)

### Uploading Files

To upload workflow files:

1. Navigate to the "Upload" section
2. Click the "Upload Files" button or drag and drop files into the designated area
3. Select the files you want to upload
4. Click "Upload" to begin the upload process

### Creating Analysis Sessions

After uploading workflows, you can create an analysis session:

1. Enter a name for the session (optional)
2. Select the workflows to include in the session
3. Configure analysis options (if needed)
4. Click "Create Session" to begin the analysis

## Analyzing Workflows

### Analysis Types

The n8n Workflow Analyzer provides several types of analysis:

- **Basic Analysis**: Analyzes individual workflows for structure, complexity, and patterns
- **Collection Analysis**: Analyzes relationships and patterns across multiple workflows
- **Pattern Mining**: Discovers frequent patterns and association rules
- **Network Analysis**: Analyzes the network structure of workflow collections

### Running Analysis

To run an analysis:

1. Select an analysis session from the Sessions list
2. Choose the type of analysis you want to run
3. Configure analysis parameters (if needed)
4. Click "Run Analysis" to begin the analysis process

### Analysis Parameters

You can customize the analysis by adjusting various parameters:

- **Complexity Metrics**: Choose which complexity metrics to include
- **Pattern Mining Settings**: Adjust support and confidence thresholds
- **Visualization Options**: Configure visualization settings

## Understanding Analysis Results

### Workflow Metrics

The analyzer calculates several metrics for each workflow:

- **Structural Complexity**: Measures the complexity of the workflow structure
- **Cognitive Complexity**: Estimates the cognitive load required to understand the workflow
- **Cyclomatic Complexity**: Counts the number of linearly independent paths through the workflow
- **Error Handling Coverage**: Measures the percentage of nodes with error handling
- **Branching Factor**: Measures the average number of outgoing connections per node

### Interpretation Guidelines

When interpreting analysis results, consider the following guidelines:

- **Complexity Scores**: Higher scores indicate more complex workflows
  - Low: 0-10
  - Medium: 11-30
  - High: 31+
- **Error Handling Coverage**: Higher percentages indicate better error handling
  - Poor: 0-30%
  - Adequate: 31-70%
  - Good: 71-100%
- **Branching Factor**: Higher values indicate more branching
  - Linear: 0-1
  - Moderate: 1.1-2
  - High: 2.1+

## Pattern Mining

### Understanding Pattern Mining

Pattern mining discovers frequent patterns and association rules across workflows. This helps identify:

- Common workflow structures
- Frequently co-occurring nodes
- Potential standardization opportunities
- Hidden relationships between workflow elements

### Frequent Patterns

Frequent patterns are sets of nodes or connections that appear together in multiple workflows. The analyzer identifies these patterns and ranks them by:

- **Support**: The percentage of workflows containing the pattern
- **Size**: The number of elements in the pattern

### Association Rules

Association rules identify relationships between workflow elements, such as "if node A is present, then node B is likely to be present." The analyzer evaluates rules based on:

- **Confidence**: The probability that the rule is true
- **Lift**: The strength of the relationship between elements
- **Support**: The frequency of the rule in the dataset

### Using Pattern Mining Results

Pattern mining results can be used to:

- Standardize workflow designs
- Identify common patterns for reuse
- Detect anomalies or deviations from standards
- Create workflow templates

## Network Analysis

### Understanding Network Analysis

Network analysis treats workflows as networks of connected nodes. This approach reveals:

- Structural properties of workflows
- Relationships between workflows
- Central or critical nodes
- Community structures

### Network Metrics

The analyzer calculates several network metrics:

- **Density**: The ratio of actual connections to possible connections
- **Average Degree**: The average number of connections per node
- **Diameter**: The longest shortest path between any two nodes
- **Average Path Length**: The average shortest path between all pairs of nodes
- **Clustering Coefficient**: The degree to which nodes tend to cluster together

### Interpreting Network Results

Network analysis results provide insights into workflow structure:

- **High Density**: Indicates complex, highly interconnected workflows
- **Low Average Path Length**: Suggests efficient workflows with direct paths
- **High Clustering**: Indicates modular workflows with functional grouping

## Visualizations

### Available Visualizations

The n8n Workflow Analyzer provides several interactive visualizations:

- **Workflow Diagrams**: Visual representations of individual workflows
- **Complexity Charts**: Visualizations of complexity metrics
- **Pattern Networks**: Networks of frequent patterns and associations
- **Workflow Networks**: Network visualizations of workflow collections
- **Statistical Dashboards**: Comprehensive dashboards with multiple charts

### Interacting with Visualizations

Most visualizations are interactive and allow you to:

- Zoom in and out
- Pan across the visualization
- Hover over elements for additional information
- Click on elements to drill down
- Filter data
- Change visualization parameters

### Customizing Visualizations

You can customize visualizations by:

1. Clicking the "Customize" button on the visualization
2. Adjusting parameters such as colors, layouts, and data filters
3. Applying the changes to update the visualization

## Exporting Results

### Export Formats

The analyzer supports exporting results in various formats:

- **JSON**: Raw analysis data
- **CSV**: Tabular data for spreadsheet applications
- **HTML**: Interactive visualizations
- **PNG/SVG**: Static visualizations
- **PDF**: Comprehensive reports

### Exporting Data

To export analysis results:

1. Navigate to the analysis results page
2. Click the "Export" button
3. Select the export format
4. Choose what to include in the export
5. Click "Export" to download the file

### Sharing Results

You can share analysis results by:

- Exporting and sending the files
- Generating shareable links (if supported)
- Scheduling automated reports (if configured)

## API Integration

### API Overview

The n8n Workflow Analyzer provides a RESTful API for integration with other systems. The API allows you to:

- Upload workflows
- Run analyses
- Retrieve results
- Access visualizations

### Authentication

To use the API, you need to authenticate:

```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

This returns an access token that you must include in subsequent requests.

### Example API Calls

#### Uploading a Workflow

```bash
curl -X POST http://localhost:5000/api/upload \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@path/to/workflow.json"
```

#### Running an Analysis

```bash
curl -X POST http://localhost:5000/api/analyze/SESSION_ID \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"analysis_type": "basic"}'
```

#### Retrieving Results

```bash
curl -X GET http://localhost:5000/api/results/SESSION_ID \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Troubleshooting

### Common Issues

#### Upload Errors

- **File Too Large**: Reduce file size or increase upload limit
- **Invalid Format**: Ensure files are valid n8n workflow JSON
- **Upload Timeout**: Try uploading smaller batches

#### Analysis Errors

- **Analysis Timeout**: Reduce the number of workflows or simplify analysis
- **Memory Error**: Reduce data size or increase server resources
- **Invalid Results**: Check workflow JSON for errors or inconsistencies

#### Visualization Errors

- **Visualization Not Loading**: Check browser compatibility and JavaScript settings
- **Incomplete Visualization**: Ensure analysis completed successfully
- **Performance Issues**: Reduce data size or simplify visualization

### Getting Help

If you encounter issues not covered in this guide:

- Check the application logs
- Consult the FAQ section
- Contact support with error details

## Glossary

- **n8n**: A fair-code licensed workflow automation tool
- **Workflow**: A sequence of connected nodes that perform automation tasks
- **Node**: A single step in a workflow that performs a specific function
- **Connection**: A link between nodes that defines the flow of data
- **Pattern**: A recurring structure or arrangement of nodes and connections
- **Association Rule**: A relationship between workflow elements (e.g., "if A then B")
- **Support**: The percentage of workflows containing a pattern
- **Confidence**: The probability that a rule is true
- **Lift**: The strength of a relationship between elements
- **Network Analysis**: The study of workflows as networks of connected nodes
- **Density**: The ratio of actual connections to possible connections
- **Clustering Coefficient**: The degree to which nodes tend to cluster together

