#!/usr/bin/env python3
"""
Test script to simulate the CI/CD pipeline steps locally.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        print(f"❌ {description} - FAILED")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False


def main():
    """Run CI/CD pipeline simulation."""
    print("🚀 CI/CD Pipeline Simulation")
    print("=" * 60)

    # Change to project directory (parent of scripts)
    os.chdir(Path(__file__).parent.parent)

    # Pipeline steps in order
    steps = [
        # 1. Lint and Format Check
        (
            ["poetry", "run", "python", "-m", "black", "--check", "--diff", "src/"],
            "Black formatting check",
        ),
        (
            [
                "poetry",
                "run",
                "python",
                "-m",
                "isort",
                "--check-only",
                "--diff",
                "src/",
            ],
            "Import sorting check",
        ),
        (["poetry", "run", "python", "-m", "ruff", "check", "src/"], "Ruff linting"),
        # 2. Build Assets
        (["poetry", "build"], "Build Python package"),
        # 3. Unit Tests
        (
            ["poetry", "run", "python", "scripts/test_local_validation.py"],
            "Unit tests (validation)",
        ),
        # 4. Static Analysis
        (
            [
                "poetry",
                "run",
                "python",
                "-m",
                "bandit",
                "-r",
                "src/",
                "-ll",
                "--skip",
                "B101",
            ],
            "Security analysis with Bandit",
        ),
        (
            [
                "poetry",
                "run",
                "python",
                "-m",
                "mypy",
                "src/",
                "--ignore-missing-imports",
            ],
            "Type checking with MyPy",
        ),
    ]

    passed = 0
    failed = 0

    for cmd, description in steps:
        if run_command(cmd, description):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Pipeline Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All pipeline steps passed! Ready for deployment.")

        # Show next steps
        print("\n🚀 Next Steps:")
        print(
            "1. Commit your changes: git add . && git commit -m 'feat: update CI/CD pipeline'"
        )
        print("2. Push to feature branch: git push origin feature/ci-cd-update")
        print("3. Create pull request to main branch")
        print("4. GitHub Actions will run the full pipeline")

        return 0
    print("⚠️  Some pipeline steps failed. Please fix issues before pushing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
