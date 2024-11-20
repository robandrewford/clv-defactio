import subprocess
import sys
from pathlib import Path

def freeze_local_packages():
    print("📦 Freezing local packages...")
    
    try:
        # Get local packages in requirements format
        process = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Write to requirements.txt
        requirements_path = Path(__file__).parents[2] / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(process.stdout)
            
        print("✅ Successfully froze local packages to requirements.txt")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error freezing packages: {e.stderr}")
        return False

if __name__ == "__main__":
    sys.exit(0 if freeze_local_packages() else 1)