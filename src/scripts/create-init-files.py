import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.parent

# Define directories that need __init__.py files
dirs = [
    project_root / 'src',
    project_root / 'src/data',
    project_root / 'src/pipeline',
    project_root / 'src/config',
    project_root / 'src/utils',
    project_root / 'src/visualization',
    project_root / 'src/scripts',
    project_root / 'tests'
]

def create_init_files():
    """Create __init__.py files in specified directories if they don't exist."""
    for directory in dirs:
        init_file = directory / '__init__.py'
        if not init_file.exists():
            print(f"Creating {init_file}")
            init_file.touch()
        else:
            print(f"__init__.py already exists in {directory}")

if __name__ == "__main__":
    create_init_files()
