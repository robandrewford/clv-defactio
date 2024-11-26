import subprocess
import sys

def run_tests(test_path=None):
    """Run pytest with optional test path"""
    cmd = ['pytest']
    if test_path:
        cmd.append(test_path)
    return subprocess.call(cmd)

def main():
    # Get the script name and any additional arguments
    script_name = sys.argv[0]
    args = sys.argv[1:]

    # Determine which test to run based on the script name
    test_paths = {
        'test-convergence': 'tests/test_clv/test_model_diagnostics.py',
        'test-performance': 'tests/test_clv/test_performance.py',
        'test-integration': 'tests/test_clv/test_integration.py',
    }

    test_path = test_paths.get(script_name.split('/')[-1])
    
    if test_path:
        sys.exit(run_tests(test_path))
    else:
        sys.exit(run_tests())

if __name__ == '__main__':
    main() 