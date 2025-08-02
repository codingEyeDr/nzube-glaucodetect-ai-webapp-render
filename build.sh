#!/bin/bash
python3.10 -m pip install --upgrade pip
python3.10 -m pip install --no-cache-dir -r requirements.txt
python3.10 -m pip install --force-reinstall groq==0.3.0  # Add this line