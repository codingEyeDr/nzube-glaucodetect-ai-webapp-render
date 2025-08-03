#!/bin/bash
# Force clean Python 3.10 environment
python3.10 -m pip install --upgrade pip

# Critical dependency order
python3.10 -m pip uninstall -y tensorflow keras  # Clean any existing installations
python3.10 -m pip install --no-cache-dir tensorflow==2.16.1  # Must match your model's version
python3.10 -m pip install --no-cache-dir protobuf==3.20.3  # Required for TF 2.16

# Core dependencies
python3.10 -m pip install --no-cache-dir python-dotenv==1.0.1
python3.10 -m pip install --no-cache-dir groq==0.4.0

# Main installation
python3.10 -m pip install --no-cache-dir -r requirements.txt

# Verify installations
python3.10 -c "
import tensorflow as tf;
import groq, dotenv;
print(f'TensorFlow: {tf.__version__}');
print(f'Groq: {groq.__version__}');
print(f'Dotenv: {dotenv.__version__}');
print('All critical dependencies installed successfully');
"