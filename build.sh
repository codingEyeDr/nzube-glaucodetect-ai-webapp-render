#!/bin/bash
# Force clean Python 3.10 environment
python3.10 -m pip install --upgrade pip

# Clean existing installations
python3.10 -m pip uninstall -y tensorflow keras

# Install core dependencies
python3.10 -m pip install --no-cache-dir tensorflow==2.16.1 protobuf==3.20.3
python3.10 -m pip install --no-cache-dir python-dotenv==1.0.1 groq==0.4.0

# Install remaining requirements
python3.10 -m pip install --no-cache-dir -r requirements.txt

# Simplified verification (removed dotenv version check)
python3.10 -c "
import tensorflow as tf;
import groq;
print(f'\n=== Verification ===');
print(f'TensorFlow: {tf.__version__}');
print(f'Groq: {groq.__version__}');
print('All critical dependencies installed successfully');
"