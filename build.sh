#!/bin/bash
# Force clean Python 3.10 environment
python3.10 -m pip install --upgrade pip
python3.10 -m pip uninstall -y groq  # Remove any corrupted installs

# Core dependency (must come before requirements.txt)
python3.10 -m pip install --no-cache-dir python-dotenv==1.0.1

# Main installation (Groq pinned to avoid conflicts)
python3.10 -m pip install --no-cache-dir groq==0.4.0
python3.10 -m pip install --no-cache-dir -r requirements.txt

# Verify critical installations
python3.10 -c "import groq, dotenv; print(f'Groq version: {groq.__version__}, dotenv: {dotenv.__version__}')"