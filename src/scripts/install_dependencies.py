import pkg_resources
import sys
from pathlib import Path
import subprocess

def install_dependencies(auto_install=True):
    """Install dependencies from requirements.txt."""
    print("📦 Installing dependencies...")
    
    requirements_path = Path(__file__).parents[2] / "requirements.txt"
    
    try:
        # Uninstall pandas-profiling if it exists
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "pandas-profiling"],
            capture_output=True,
            text=True
        )
        
        # Install exactly what's in requirements.txt
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path), "--no-deps"],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Then install dependencies but ignore already installed packages
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path), "--ignore-installed"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Successfully installed dependencies!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e.stderr}")
        return False

if __name__ == "__main__":
    sys.exit(0 if install_dependencies() else 1)