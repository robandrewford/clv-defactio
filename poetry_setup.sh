#!/bin/bash

# Remove the Poetry environment if it exists
poetry env remove --all

# Clear Poetry's cache
rm -rf ~/Library/Caches/pypoetry
rm -rf ~/.cache/pypoetry

# Install pyenv if you haven't already
brew install pyenv

# Install Python 3.11.7
pyenv install 3.11.7

# Set it as your local version for this project
pyenv local 3.11.7

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Initialize Poetry in your existing project
poetry init

# Remove the existing virtual environment if any
poetry env remove python

# Create new environment with Python 3.11.7
poetry env use $(pyenv which python3.11)

# Create new conda environment with Python 3.11.7
conda create -n clv-defactio python=3.11.7

# Activate the environment
conda activate clv-defactio

# Install dependencies
poetry install

# Create virtual environment in project directory
poetry config virtualenvs.in-project true

# Use parallel installer for better performance
poetry config installer.parallel true

# Enable modern installation method
poetry config installer.modern-installation true

# Add new dependencies
poetry add package-name

# Add dev dependencies
poetry add --group dev package-name

# Update dependencies
poetry update

# Run commands in virtual environment
poetry run python script.py

# Export requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt