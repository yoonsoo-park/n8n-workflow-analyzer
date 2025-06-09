# n8n Workflow Analyzer

A comprehensive system for analyzing large sets of n8n workflows, extracting insights through data mining and network analysis, and providing interactive visualizations and reports.

## Overview

The n8n Workflow Analyzer is a powerful tool designed to help n8n users and administrators gain deeper insights into their workflow collections. By analyzing the structure, patterns, and relationships within and across workflows, this system provides valuable information for optimization, standardization, and governance of n8n automation environments.

Key features include:

- **Workflow Structure Analysis**: Analyze individual workflows for complexity, error handling, and structural patterns
- **Pattern Mining**: Discover frequent patterns and association rules across workflow collections
- **Network Analysis**: Visualize and analyze the relationships between workflows as a network
- **Interactive Visualizations**: Generate interactive charts, graphs, and dashboards for exploring analysis results
- **Web Interface**: User-friendly web application for uploading, analyzing, and exploring workflows
- **API Access**: RESTful API for integrating analysis capabilities into other systems

## Installation

### Prerequisites

- Python 3.11 or higher
- Node.js 20.x or higher
- n8n workflow JSON files

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/n8n-workflow-analyzer.git
cd n8n-workflow-analyzer
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:

```bash
cd web_ui
npm install
```

4. Install backend dependencies:

```bash
cd api_server
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The n8n Workflow Analyzer provides several command-line tools for analyzing workflows:

#### Basic Workflow Analysis

```bash
python analyze_workflow.py --input path/to/workflow.json --output path/to/output
```

#### Batch Analysis

```bash
python analyze_collection.py --input path/to/workflows/directory --output path/to/output
```

#### Pattern Mining

```bash
python mine_patterns.py --input path/to/workflows/directory --output path/to/output
```

#### Network Analysis

```bash
python analyze_network.py --input path/to/workflows/directory --output path/to/output
```

### Web Interface

The n8n Workflow Analyzer includes a web interface for easy interaction:

1. Start the backend API server:

```bash
cd api_server
python src/main.py
```

2. Start the frontend development server:

```bash
cd web_ui
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

### API Usage

The n8n Workflow Analyzer provides a RESTful API for integration with other systems:

```bash
# Start the API server
cd api_server
python src/main.py
```

API endpoints:

- `POST /api/upload`: Upload workflow files
- `GET /api/workflows`: List uploaded workflows
- `GET /api/workflow/{id}`: Get workflow details
- `POST /api/analyze/{id}`: Analyze a specific workflow
- `GET /api/results/{id}`: Get analysis results
- `GET /api/visualizations/{id}/{filename}`: Get visualization files

## Architecture

The n8n Workflow Analyzer is built with a modular architecture:

- **Core Analysis Engine**: Python modules for workflow parsing, analysis, pattern mining, and network analysis
- **Visualization Engine**: Python modules for generating interactive visualizations
- **API Server**: Flask-based RESTful API for accessing analysis capabilities
- **Web Interface**: React-based user interface for interacting with the system

## Development

### Project Structure

```
n8n_workflow_analyzer/
├── src/                      # Core Python modules
│   ├── models.py             # Data models
│   ├── parser.py             # Workflow JSON parser
│   ├── analysis/             # Analysis modules
│   ├── mining/               # Pattern mining modules
│   ├── network/              # Network analysis modules
│   └── visualization/        # Visualization modules
├── api_server/               # Flask API server
├── web_ui/                   # React frontend
├── data/                     # Data directory
│   ├── sample_workflows/     # Sample workflow files
│   ├── reports/              # Analysis reports
│   └── visualizations/       # Generated visualizations
├── tests/                    # Test suite
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

### Running Tests

```bash
python test_suite.py
```

### Performance Testing

```bash
python performance_test.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [n8n](https://n8n.io/) - Workflow automation tool
- [NetworkX](https://networkx.org/) - Network analysis library
- [Plotly](https://plotly.com/) - Interactive visualization library
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [React](https://reactjs.org/) - Frontend library

