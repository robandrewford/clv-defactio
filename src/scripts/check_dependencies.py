import pkg_resources
import sys
from pathlib import Path
import subprocess
import time
from typing import List, Tuple
import os

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
        print("📦 Package Check Results")
        print("=" * 50)
        
        # Print current page of results
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_results)
        
        for pkg, installed, message in results[start_idx:end_idx]:
            status = "✅" if installed else "❌"
            print(f"{status} {pkg}: {message if not installed else ''}")
        
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
            print("Navigation: [n]ext, [p]revious, [q]uit")
            try:
                cmd = input("> ").lower()
                if cmd == 'q':
                    break
                elif cmd == 'n' and current_page < total_pages - 1:
                    current_page += 1
                elif cmd == 'p' and current_page > 0:
                    current_page -= 1
            except KeyboardInterrupt:
                print("\nExiting results view...")
                break

def install_dependencies(auto_install=True):
    """Install dependencies from requirements.txt."""
    print("📦 Installing dependencies...")
    
    requirements_path = Path(__file__).parents[2] / "requirements.txt"
    
    try:
        # First try without dependencies
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path), "--no-deps"],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Then install with dependencies
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Successfully installed dependencies!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e.stderr}")
        return False

def check_dependencies(auto_install=True):
    # Read requirements file
    requirements_path = Path(__file__).parents[2] / "requirements.txt"
    
    try:
        with open(requirements_path) as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("❌ Error: requirements.txt not found!")
        return False

    # Check each requirement
    results = []
    missing = []
    conflicts = []
    
    for requirement in requirements:
        try:
            pkg_resources.require(requirement)
            results.append((requirement, True, ""))
        except pkg_resources.DistributionNotFound:
            missing.append(requirement)
            results.append((requirement, False, "Missing package"))
        except pkg_resources.VersionConflict as e:
            conflicts.append(requirement)
            results.append((requirement, False, f"Version conflict: {str(e)}"))

    # Show scrollable results
    show_scrolling_results(results)
    
    if conflicts:
        print("\n⚠️  Please resolve version conflicts manually for:")
        for pkg in conflicts:
            print(f"   - {pkg}")

    if missing:
        print("\n❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        
        if auto_install:
            if install_dependencies(missing):
                # Wait a moment for installations to complete
                time.sleep(1)
                # Recheck dependencies
                print("\n🔍 Verifying installations...")
                return check_dependencies(auto_install=False)
        else:
            print("\nTo install manually, run:")
            print(f"pip install {' '.join(missing)}")
            return False
    
    if not missing and not conflicts:
        print("\n✅ All dependencies are installed!")
        return True
    
    return False

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    sys.exit(0 if check_dependencies() else 1) 