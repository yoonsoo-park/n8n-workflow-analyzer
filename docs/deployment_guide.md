# n8n Workflow Analyzer - Deployment Guide

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration](#configuration)
7. [Security Considerations](#security-considerations)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Backup and Recovery](#backup-and-recovery)
10. [Scaling](#scaling)
11. [Troubleshooting](#troubleshooting)

## Introduction

This guide provides detailed instructions for deploying the n8n Workflow Analyzer in various environments. The system consists of two main components:

1. **API Server**: A Flask-based backend that provides the analysis capabilities
2. **Web Interface**: A React-based frontend that provides the user interface

These components can be deployed together or separately, depending on your requirements.

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores
- **RAM**: 4 GB
- **Disk**: 10 GB
- **Operating System**: Linux (Ubuntu 20.04 or later recommended)
- **Python**: 3.11 or later
- **Node.js**: 20.x or later

### Recommended Requirements

- **CPU**: 4+ cores
- **RAM**: 8+ GB
- **Disk**: 20+ GB SSD
- **Operating System**: Linux (Ubuntu 22.04 or later)
- **Python**: 3.11 or later
- **Node.js**: 20.x or later

### Software Dependencies

- **Python Packages**: See `requirements.txt`
- **Node.js Packages**: See `web_ui/package.json`
- **System Packages**: `build-essential`, `python3-dev`, `libffi-dev`

## Local Deployment

### Prerequisites

1. Install system dependencies:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-pip libffi-dev
```

2. Install Node.js:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

3. Clone the repository:

```bash
git clone https://github.com/yourusername/n8n-workflow-analyzer.git
cd n8n-workflow-analyzer
```

### Backend Deployment

1. Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install API server dependencies:

```bash
cd api_server
pip install -r requirements.txt
cd ..
```

4. Start the API server:

```bash
cd api_server
python src/main.py
```

The API server will be available at `http://localhost:5000`.

### Frontend Deployment

1. Install frontend dependencies:

```bash
cd web_ui
npm install
```

2. Start the development server:

```bash
npm run dev
```

The web interface will be available at `http://localhost:5173`.

### Production Deployment

For production deployment, you should use a production-ready web server:

1. Build the frontend:

```bash
cd web_ui
npm run build
```

2. Serve the frontend with a web server like Nginx:

```bash
sudo apt install -y nginx
sudo cp -r web_ui/dist/* /var/www/html/
```

3. Run the API server with Gunicorn:

```bash
cd api_server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.main:app
```

4. Configure Nginx to proxy API requests:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        root /var/www/html;
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Docker Deployment

### Prerequisites

1. Install Docker and Docker Compose:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/n8n-workflow-analyzer.git
cd n8n-workflow-analyzer
```

### Docker Compose Deployment

1. Create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  api:
    build:
      context: ./api_server
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
    environment:
      - FLASK_ENV=production
      - FLASK_APP=src/main.py
      - SECRET_KEY=your-secret-key

  web:
    build:
      context: ./web_ui
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - api
```

2. Create a Dockerfile for the API server (`api_server/Dockerfile`):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.main:app"]
```

3. Create a Dockerfile for the web interface (`web_ui/Dockerfile`):

```dockerfile
FROM node:20 as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

4. Create an Nginx configuration file (`web_ui/nginx.conf`):

```nginx
server {
    listen 80;
    server_name localhost;

    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://api:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

5. Build and start the containers:

```bash
docker-compose build
docker-compose up -d
```

The application will be available at `http://localhost`.

## Cloud Deployment

### AWS Deployment

#### Prerequisites

1. AWS account with appropriate permissions
2. AWS CLI installed and configured
3. Docker installed

#### Elastic Beanstalk Deployment

1. Initialize Elastic Beanstalk application:

```bash
eb init -p docker n8n-workflow-analyzer
```

2. Create an Elastic Beanstalk environment:

```bash
eb create n8n-workflow-analyzer-env
```

3. Deploy the application:

```bash
eb deploy
```

#### ECS Deployment

1. Create an ECR repository:

```bash
aws ecr create-repository --repository-name n8n-workflow-analyzer
```

2. Build and push Docker images:

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<region>.amazonaws.com
docker build -t <your-account-id>.dkr.ecr.<region>.amazonaws.com/n8n-workflow-analyzer:api ./api_server
docker build -t <your-account-id>.dkr.ecr.<region>.amazonaws.com/n8n-workflow-analyzer:web ./web_ui
docker push <your-account-id>.dkr.ecr.<region>.amazonaws.com/n8n-workflow-analyzer:api
docker push <your-account-id>.dkr.ecr.<region>.amazonaws.com/n8n-workflow-analyzer:web
```

3. Create an ECS cluster, task definitions, and services using the AWS console or CLI.

### Azure Deployment

#### Prerequisites

1. Azure account with appropriate permissions
2. Azure CLI installed and configured
3. Docker installed

#### Azure App Service Deployment

1. Create a resource group:

```bash
az group create --name n8n-workflow-analyzer --location eastus
```

2. Create an App Service plan:

```bash
az appservice plan create --name n8n-workflow-analyzer-plan --resource-group n8n-workflow-analyzer --sku B1 --is-linux
```

3. Create a web app for the API server:

```bash
az webapp create --resource-group n8n-workflow-analyzer --plan n8n-workflow-analyzer-plan --name n8n-workflow-analyzer-api --runtime "PYTHON|3.11"
```

4. Create a web app for the web interface:

```bash
az webapp create --resource-group n8n-workflow-analyzer --plan n8n-workflow-analyzer-plan --name n8n-workflow-analyzer-web --runtime "NODE|20-lts"
```

5. Deploy the API server:

```bash
cd api_server
az webapp up --name n8n-workflow-analyzer-api --resource-group n8n-workflow-analyzer
```

6. Deploy the web interface:

```bash
cd web_ui
npm run build
az webapp deployment source config-local-git --name n8n-workflow-analyzer-web --resource-group n8n-workflow-analyzer
git init
git add .
git commit -m "Initial commit"
git remote add azure <git-url-from-previous-command>
git push azure master
```

### Google Cloud Platform Deployment

#### Prerequisites

1. GCP account with appropriate permissions
2. Google Cloud SDK installed and configured
3. Docker installed

#### Google App Engine Deployment

1. Create an `app.yaml` file for the API server:

```yaml
runtime: python311
entrypoint: gunicorn -w 4 -b :$PORT src.main:app

env_variables:
  FLASK_ENV: "production"
  SECRET_KEY: "your-secret-key"
```

2. Create an `app.yaml` file for the web interface:

```yaml
runtime: nodejs20
handlers:
- url: /
  static_files: dist/index.html
  upload: dist/index.html

- url: /(.*)
  static_files: dist/\1
  upload: dist/(.*)
```

3. Deploy the API server:

```bash
cd api_server
gcloud app deploy
```

4. Deploy the web interface:

```bash
cd web_ui
npm run build
gcloud app deploy
```

## Configuration

### Environment Variables

The n8n Workflow Analyzer can be configured using environment variables:

#### API Server

- `FLASK_ENV`: Environment mode (`development` or `production`)
- `FLASK_APP`: Flask application entry point (`src/main.py`)
- `SECRET_KEY`: Secret key for session encryption
- `DATABASE_URL`: Database connection URL
- `UPLOAD_FOLDER`: Path to upload directory
- `MAX_CONTENT_LENGTH`: Maximum upload file size
- `CORS_ORIGINS`: Allowed CORS origins

#### Web Interface

- `VITE_API_URL`: URL of the API server
- `VITE_ENABLE_ANALYTICS`: Enable/disable analytics
- `VITE_ENVIRONMENT`: Environment mode (`development` or `production`)

### Configuration Files

#### API Server

The API server can be configured using a configuration file (`api_server/src/config.py`):

```python
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'data/uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
```

#### Web Interface

The web interface can be configured using environment files:

- `.env`: Default environment variables
- `.env.development`: Development-specific variables
- `.env.production`: Production-specific variables

## Security Considerations

### API Security

1. **Authentication**: Implement token-based authentication for API access
2. **Authorization**: Implement role-based access control
3. **Rate Limiting**: Limit the number of requests per client
4. **Input Validation**: Validate all input data
5. **HTTPS**: Use HTTPS for all API communication

### Data Security

1. **Encryption**: Encrypt sensitive data at rest
2. **Access Control**: Implement proper access controls for data
3. **Data Isolation**: Isolate data between users
4. **Secure Storage**: Use secure storage for sensitive data
5. **Data Retention**: Implement data retention policies

### Web Security

1. **HTTPS**: Use HTTPS for all web communication
2. **Content Security Policy**: Implement CSP headers
3. **CSRF Protection**: Implement CSRF protection
4. **XSS Protection**: Implement XSS protection
5. **Secure Cookies**: Use secure and HttpOnly cookies

## Monitoring and Logging

### Logging

The n8n Workflow Analyzer uses Python's logging module for logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Monitoring

For production deployments, consider implementing monitoring:

1. **Health Checks**: Implement health check endpoints
2. **Metrics**: Collect and monitor system metrics
3. **Alerts**: Set up alerts for critical issues
4. **Dashboards**: Create monitoring dashboards

### Monitoring Tools

- **Prometheus**: For metrics collection
- **Grafana**: For metrics visualization
- **ELK Stack**: For log management
- **New Relic**: For application performance monitoring

## Backup and Recovery

### Data Backup

1. **Database Backup**: Regularly backup the database
2. **File Backup**: Regularly backup uploaded files and generated visualizations
3. **Configuration Backup**: Backup configuration files

### Backup Strategy

1. **Regular Backups**: Schedule regular backups
2. **Off-site Storage**: Store backups in a different location
3. **Backup Rotation**: Implement a backup rotation strategy
4. **Backup Testing**: Regularly test backup restoration

### Recovery Procedures

1. **Database Recovery**: Restore the database from backup
2. **File Recovery**: Restore files from backup
3. **Configuration Recovery**: Restore configuration files
4. **Application Recovery**: Redeploy the application

## Scaling

### Horizontal Scaling

1. **API Server Scaling**: Deploy multiple API server instances
2. **Web Interface Scaling**: Deploy multiple web interface instances
3. **Load Balancing**: Use a load balancer to distribute traffic

### Vertical Scaling

1. **Increase Resources**: Allocate more CPU, memory, and disk resources
2. **Optimize Performance**: Optimize code and database queries
3. **Caching**: Implement caching for frequently accessed data

### Database Scaling

1. **Connection Pooling**: Implement database connection pooling
2. **Sharding**: Implement database sharding for large datasets
3. **Replication**: Implement database replication for read scaling

## Troubleshooting

### Common Issues

#### API Server Issues

1. **Server Not Starting**: Check logs for errors
2. **Database Connection Issues**: Check database connection settings
3. **File Permission Issues**: Check file permissions for upload directory

#### Web Interface Issues

1. **Build Errors**: Check build logs for errors
2. **API Connection Issues**: Check API URL configuration
3. **CORS Issues**: Check CORS configuration

### Debugging

1. **Enable Debug Mode**: Set `FLASK_ENV=development` for API server
2. **Check Logs**: Check application logs for errors
3. **Use Browser DevTools**: Use browser developer tools for frontend debugging

### Getting Help

If you encounter issues not covered in this guide:

1. **Check Documentation**: Refer to the technical documentation
2. **Check Issue Tracker**: Check the GitHub issue tracker
3. **Contact Support**: Contact the support team

