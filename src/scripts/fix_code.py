#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path
import ast
from typing import Set, List

class CodeFixer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content = Path(file_path).read_text()
        self.tree = ast.parse(self.content)
        self.used_names: Set[str] = set()
        self.imports: List[str] = []
        self.fixed_content = self.content

    def analyze_usage(self):
        """Analyze which names are actually used in the code"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                self.used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                self.used_names.add(node.attr)

    def fix_unused_imports(self):
        """Remove unused imports"""
        import_lines = []
        code_lines = []
        current_imports = []

        for line in self.content.split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                current_imports.append(line)
            else:
                if current_imports:
                    # Process the collected imports
                    for imp in current_imports:
                        if any(name in self.used_names for name in self._extract_names(imp)):
                            import_lines.append(imp)
                    current_imports = []
                code_lines.append(line)

        self.fixed_content = '\n'.join(import_lines + code_lines)

    def fix_undefined_names(self):
        """Add missing imports for undefined names"""
        common_imports = {
            'datetime': 'from datetime import datetime',
            'timedelta': 'from datetime import timedelta',
            'uuid': 'import uuid',
            'logging': 'import logging',
            'os': 'import os',
            'Feature': 'from ..base.feature import Feature',
            'RangeRule': 'from ..validation import RangeRule',
            'ProcessedData': 'from ..data.types import ProcessedData',
            'Model': 'from .base import Model',
            'StorageProvider': 'from .providers import StorageProvider',
            'GCSProvider': 'from .providers.gcs import GCSProvider',
            'S3Provider': 'from .providers.s3 import S3Provider'
        }

        needed_imports = set()
        for name in self.used_names:
            if name in common_imports and name not in self.content:
                needed_imports.add(common_imports[name])

        if needed_imports:
            import_block = '\n'.join(sorted(needed_imports))
            self.fixed_content = import_block + '\n\n' + self.fixed_content

    def fix_unused_variables(self):
        """Comment out unused variable assignments"""
        lines = self.fixed_content.split('\n')
        for i, line in enumerate(lines):
            if '=' in line:
                var_name = line.split('=')[0].strip()
                if var_name in self.used_names:
                    continue
                if not line.strip().startswith('#'):
                    lines[i] = f"# Unused: {line}"
        self.fixed_content = '\n'.join(lines)

    def _extract_names(self, import_line: str) -> List[str]:
        """Extract imported names from an import line"""
        if import_line.startswith('from'):
            match = re.search(r'import (.*?)$', import_line)
            if match:
                return [n.strip() for n in match.group(1).split(',')]
        else:
            match = re.search(r'import (.*?)$', import_line)
            if match:
                return [n.strip().split(' as ')[0] for n in match.group(1).split(',')]
        return []

    def fix_all(self):
        """Apply all fixes"""
        self.analyze_usage()
        self.fix_unused_imports()
        self.fix_undefined_names()
        self.fix_unused_variables()
        return self.fixed_content

def fix_file(file_path: str):
    """Fix a single file"""
    print(f"Fixing {file_path}...")
    fixer = CodeFixer(file_path)
    fixed_content = fixer.fix_all()
    
    # Backup original file
    backup_path = f"{file_path}.bak"
    Path(file_path).rename(backup_path)
    
    # Write fixed content
    Path(file_path).write_text(fixed_content)
    print(f"Fixed {file_path} (backup saved as {backup_path})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_code.py <file_or_directory>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isfile(path):
        fix_file(path)
    else:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    fix_file(os.path.join(root, file))

if __name__ == "__main__":
    main() 