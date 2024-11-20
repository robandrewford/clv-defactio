import importlib
import sys
from pathlib import Path

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {str(e)}")
        return False

def get_package_names():
    # Read requirements file
    requirements_path = Path(__file__).parents[2] / "requirements.txt"
    
    try:
        with open(requirements_path) as f:
            # Strip version numbers and other markers from requirements
            packages = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove version specifiers and other markers
                    package = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
                    packages.append(package)
            return packages
    except FileNotFoundError:
        print("❌ Error: requirements.txt not found!")
        return []

def main():
    print("🔍 Testing imports for all dependencies...\n")
    
    packages = get_package_names()
    if not packages:
        return False

    # Common package name mappings (add more as needed)
    package_mappings = {
        'python-dotenv': 'dotenv',
        # Add other mappings here if needed
    }

    failed_imports = []
    
    for package in packages:
        # Use mapping if it exists, otherwise use package name
        import_name = package_mappings.get(package, package)
        if not test_import(import_name):
            failed_imports.append(package)
    
    print("\n" + "="*50)
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} package(s) failed to import:")
        for package in failed_imports:
            print(f"   - {package}")
        print("\nTry reinstalling these packages or check for any additional dependencies.")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 