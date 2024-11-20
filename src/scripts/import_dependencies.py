import importlib
import sys
from pathlib import Path
from typing import List, Tuple
import os
import time

def get_terminal_size() -> Tuple[int, int]:
    """Get terminal width and height."""
    try:
        columns, lines = os.get_terminal_size()
        return max(columns, 80), max(lines, 20)  # Minimum size
    except OSError:
        return 80, 20  # Default size

def show_scrolling_results(results: List[Tuple[str, bool, str]], page_size: int = None, auto_advance: bool = True):
    """Display results in a scrollable format."""
    if page_size is None:
        _, terminal_height = get_terminal_size()
        page_size = terminal_height - 5

    total_results = len(results)
    current_page = 0
    total_pages = (total_results + page_size - 1) // page_size
    
    while True:
        # Clear screen
        print("\033[2J\033[H", end="")
        
        # Print header
        print("🔍 Package Import Results")
        print("=" * 50)
        
        # Print current page of results
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_results)
        
        for pkg, success, message in results[start_idx:end_idx]:
            status = "✅" if success else "❌"
            print(f"{status} {pkg}: {message if not success else ''}")
        
        # Print footer with navigation instructions
        print("\n" + "=" * 50)
        print(f"Page {current_page + 1} of {total_pages}")
        
        if auto_advance:
            print("Auto-advancing... Press Ctrl+C to stop")
            time.sleep(2)  # Wait 2 seconds before next page
            if current_page < total_pages - 1:
                current_page += 1
            else:
                break
        else:
            break

def test_import(module_name: str) -> Tuple[bool, str]:
    """Test importing a module."""
    try:
        importlib.import_module(module_name)
        return True, ""
    except ImportError as e:
        return False, str(e)

def get_package_names() -> List[str]:
    """Get package names from requirements.txt."""
    requirements_path = Path(__file__).parents[2] / "requirements.txt"
    
    try:
        with open(requirements_path) as f:
            packages = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
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

    # Common package name mappings
    package_mappings = {
        'python-dotenv': 'dotenv',
    }

    # Collect all results
    results = []
    failed_imports = []
    
    for package in packages:
        import_name = package_mappings.get(package, package)
        success, message = test_import(import_name)
        results.append((package, success, message))
        if not success:
            failed_imports.append(package)
    
    # Show scrollable results with auto-advance
    show_scrolling_results(results, auto_advance=True)
    
    # Print summary
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