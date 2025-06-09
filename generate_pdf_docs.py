#!/usr/bin/env python3
"""
Convert Markdown documentation to PDF format.

This script converts Markdown documentation files to PDF format using the
manus-md-to-pdf utility.

Usage:
    python3 generate_pdf_docs.py
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pdf_generator')

def convert_md_to_pdf(input_file, output_file):
    """Convert a Markdown file to PDF format."""
    try:
        logger.info(f"Converting {input_file} to {output_file}")
        subprocess.run(['manus-md-to-pdf', input_file, output_file], check=True)
        logger.info(f"Successfully converted {input_file} to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_file} to {output_file}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def main():
    """Main function to convert all Markdown documentation to PDF."""
    # Define directories
    docs_dir = Path('docs')
    pdf_dir = Path('docs/pdf')
    
    # Create PDF directory if it doesn't exist
    pdf_dir.mkdir(exist_ok=True, parents=True)
    
    # Define documentation files to convert
    doc_files = [
        ('README.md', 'n8n_workflow_analyzer.pdf'),
        ('docs/user_guide.md', 'user_guide.pdf'),
        ('docs/technical_documentation.md', 'technical_documentation.pdf'),
        ('docs/deployment_guide.md', 'deployment_guide.pdf')
    ]
    
    # Convert each file
    success_count = 0
    for input_file, output_name in doc_files:
        input_path = Path(input_file)
        output_path = pdf_dir / output_name
        
        if not input_path.exists():
            logger.warning(f"Input file {input_path} does not exist. Skipping.")
            continue
        
        if convert_md_to_pdf(str(input_path), str(output_path)):
            success_count += 1
    
    # Report results
    logger.info(f"Converted {success_count} of {len(doc_files)} documentation files to PDF format")
    logger.info(f"PDF files are available in {pdf_dir}")

if __name__ == '__main__':
    main()

