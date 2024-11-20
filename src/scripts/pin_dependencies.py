import pkg_resources
import sys
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple
import argparse

def get_installed_versions() -> Dict[str, str]:
    """Get all installed packages and their versions."""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def read_requirements() -> List[str]:
    """Read current requirements file."""
    requirements_path = Path(__file__).parents[2] / "requirements.txt"
    if not requirements_path.exists():
        return []
    
    with open(requirements_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def parse_requirement(requirement: str) -> Tuple[str, Optional[str]]:
    """Extract base package name and version from requirement string."""
    parts = requirement.split('==')
    name = parts[0].split('>=')[0].split('<=')[0].split('[')[0].strip()
    version = parts[1] if len(parts) > 1 else None
    return name, version

def write_requirements(requirements: List[str]):
    """Write requirements back to file."""
    requirements_path = Path(__file__).parents[2] / "requirements.txt"
    with open(requirements_path, 'w') as f:
        f.write('\n'.join(sorted(requirements)) + '\n')

def get_current_pins() -> Dict[str, str]:
    """Get currently pinned versions from requirements.txt."""
    pins = {}
    for req in read_requirements():
        name, version = parse_requirement(req)
        if version:
            pins[name] = version
    return pins

def pin_dependencies(yes_to_all: bool = False):
    print("📌 Dependency Pinning Utility\n")
    
    # Get current requirements and installed versions
    current_requirements = read_requirements()
    installed_versions = get_installed_versions()
    current_pins = get_current_pins()
    
    if not current_requirements:
        print("❌ No requirements found in requirements.txt")
        return False
    
    # Process each requirement
    new_requirements = []
    skipped_packages = set()
    no_changes = []
    changes_made = []
    
    print("Analyzing changes...\n")
    
    for req in current_requirements:
        base_name, pinned_version = parse_requirement(req)
        
        if base_name not in installed_versions:
            print(f"⚠️  Warning: {base_name} not installed")
            new_requirements.append(req)
            continue
            
        current_version = installed_versions[base_name]
        
        # Check if package needs attention
        needs_attention = False
        if pinned_version:
            # Package is pinned, check if installed version differs
            if pinned_version != current_version:
                print(f"📢 Version mismatch for {base_name}:")
                print(f"   Pinned: {pinned_version}, Installed: {current_version}")
                needs_attention = True
        else:
            # Package is not pinned, check if it's newly added
            if base_name not in current_pins:
                print(f"📦 New package: {base_name} ({current_version})")
                needs_attention = True
        
        if needs_attention:
            if yes_to_all:
                new_requirements.append(f"{base_name}=={current_version}")
                changes_made.append(f"✅ Pinned {base_name} to version {current_version}")
            else:
                while True:
                    response = input(
                        f"   Pin to version {current_version}? [Y/n/s/a(ll)]: "
                    ).lower()
                    
                    if response == 'a':
                        yes_to_all = True
                        new_requirements.append(f"{base_name}=={current_version}")
                        changes_made.append(f"✅ Pinned {base_name} to version {current_version}")
                        break
                    elif response in ['', 'y']:
                        new_requirements.append(f"{base_name}=={current_version}")
                        print(f"✅ Pinned {base_name} to version {current_version}")
                        break
                    elif response == 'n':
                        new_requirements.append(base_name)
                        print(f"⏩ Kept {base_name} unpinned")
                        break
                    elif response == 's':
                        skipped_packages.add(base_name)
                        new_requirements.append(req)  # Keep original requirement
                        print(f"⏭️  Skipped {base_name}")
                        break
                    else:
                        print("Please enter 'y', 'n', 's', or 'a'")
        else:
            new_requirements.append(req)
            no_changes.append(base_name)
            
    # Write updated requirements
    write_requirements(new_requirements)
    
    print("\n" + "="*50)
    print("\n✅ Requirements file updated!")
    
    if yes_to_all and changes_made:
        print("\nChanges made:")
        for change in changes_made:
            print(f"   {change}")
    
    if no_changes:
        print(f"\nℹ️  {len(no_changes)} package(s) unchanged")
    
    if skipped_packages:
        print("\nℹ️  Skipped packages:")
        for pkg in sorted(skipped_packages):
            print(f"   - {pkg}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Pin Python package versions in requirements.txt')
    parser.add_argument('-y', '--yes', action='store_true', 
                       help='Automatically accept all version updates')
    args = parser.parse_args()

    try:
        return pin_dependencies(yes_to_all=args.yes)
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 