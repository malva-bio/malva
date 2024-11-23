#!/usr/bin/env python
import os
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def main():
    # Clean existing builds and coverage files
    print("Cleaning previous builds and coverage files...")
    build_dir = Path("build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    for pattern in ["*.so", ".coverage*", "*.gcda", "*.gcno"]:
        for file in Path().glob(pattern):
            file.unlink()

    # Ensure poetry environment is active
    print("Setting up poetry environment...")
    run_command(["poetry", "install"])

    # Set environment for coverage-enabled build
    env = os.environ.copy()
    env["CFLAGS"] = "-coverage"
    env["CXXFLAGS"] = "-coverage"

    # Build with coverage support
    print("Building package with coverage support...")
    run_command(["poetry", "run", "pip", "install", "-e", "."], env=env)

    # Run tests with coverage
    print("Running tests with coverage...")
    run_command([
        "poetry", "run", "pytest",
        "--cov=malva",
        "--cov-report=xml",
        "--cov-report=html",
        "tests/"
    ])

    # Generate reports
    print("Generating coverage reports...")
    run_command(["poetry", "run", "coverage", "combine"])
    run_command(["poetry", "run", "coverage", "report"])
    run_command(["poetry", "run", "coverage", "html"])

    print("\nCoverage testing complete!")
    print("View detailed report at htmlcov/index.html")

if __name__ == "__main__":
    main()