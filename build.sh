#!/bin/bash
# Force clean Python 3.10 environment
python3.10 -m pip install --upgrade pip
python3.10 -m pip uninstall -y groq  # Remove any corrupted installs

# Install with explicit version (tested working)
python3.10 -m pip install --no-cache-dir groq==0.4.0
python3.10 -m pip install --no-cache-dir -r requirements.txt

# Verify installation
python3.10 -c "import groq; print(f'Groq version: {groq.__version__}')"