"""
Script to run pytest in PyCharm Python Console
Execute this in PyCharm's Python Console to run tests
"""

import fix_win_DLL_loading_issue
import os
import sys
import subprocess
from pathlib import Path

# Set up the repository path
repo_path = Path(r"C:\Users\eleme\Documents\1Uni_Laptop\model_comparison_codes")
os.chdir(repo_path)

print(f"Current directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")

# Function to run pytest
def run_pytest(test_path="tests/", verbose=True, stop_on_first_fail=False):
    """
    Run pytest with specified options

    Args:
        test_path: Path to tests (default: "tests/")
        verbose: Enable verbose output (default: True)
        stop_on_first_fail: Stop on first failure (default: False)
    """
    cmd = [sys.executable, "-m", "pytest", test_path]

    if verbose:
        cmd.append("-v")

    if stop_on_first_fail:
        cmd.append("-x")

    # Add short traceback format
    cmd.extend(["--tb=short"])

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}")

        return result.returncode == 0

    except Exception as e:
        print(f"Error running pytest: {e}")
        return False

# Function to run specific test file
def run_single_test(test_file):
    """Run a single test file"""
    return run_pytest(f"tests/{test_file}")

# Function to run tests matching a pattern
def run_test_pattern(pattern):
    """Run tests matching a specific pattern"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-k", pattern, "-v", "--tb=short"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running pytest: {e}")
        return False

# Examples of usage:
print("=" * 50)
print("PYTEST RUNNER FOR MODEL COMPARISON CODES")
print("=" * 50)
print("\nAvailable functions:")
print("1. run_pytest() - Run all tests")
print("2. run_pytest('tests/test_confg.py') - Run specific file")
print("3. run_single_test('test_confg.py') - Run single test file")
print("4. run_test_pattern('confg') - Run tests matching pattern")
print("\nExamples:")
print("run_pytest()  # Run all tests")
print("run_single_test('test_confg.py')  # Test configuration")
print("run_single_test('test_plot_timeseries_saved_data.py')  # Test your timeseries plotting")
print("run_test_pattern('plot')  # Run all plotting tests")

# Quick test to see if pytest is available
try:
    import pytest
    print(f"\npytest version: {pytest.__version__}")
    print("pytest is available!")
except ImportError:
    print("\nWarning: pytest not installed. Install with: pip install pytest")

if __name__ == "__main__":
    test_path = "tests/"
    verbose = True
    stop_on_first_fail = False

    cmd = [sys.executable, "-m", "pytest", test_path]

    if verbose:
        cmd.append("-v")

    if stop_on_first_fail:
        cmd.append("-x")

    # Add short traceback format
    cmd.extend(["--tb=short"])

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}")

    except Exception as e:
        print(f"Error running pytest: {e}")
