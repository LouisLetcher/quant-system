#!/usr/bin/env python
"""
Changelog Generator for Quant Trading System

This script automatically generates a CHANGELOG.md file by:
1. Parsing git commit history
2. Grouping commits by type (feature, fix, docs, etc.)
3. Formatting them into a readable changelog
4. Prepending to the existing CHANGELOG.md file

Usage:
    python scripts/generate_changelog.py [--since TAG] [--version VERSION]
"""

import argparse
import os
import re
import subprocess
from datetime import datetime
from collections import defaultdict

# Configure commit types and their display names
COMMIT_TYPES = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "docs": "Documentation",
    "style": "Styling",
    "refactor": "Code Refactoring",
    "perf": "Performance Improvements",
    "test": "Tests",
    "build": "Build System",
    "ci": "CI/CD",
    "chore": "Chores",
    "revert": "Reverts",
    "other": "Other Changes"
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a changelog from git commits")
    parser.add_argument("--since", help="Generate changelog since this git tag/ref", default=None)
    parser.add_argument("--version", help="Version number for the new release", default=None)
    return parser.parse_args()

def get_git_log(since=None):
    """Get git commit logs since the specified tag or ref."""
    cmd = ["git", "log", "--pretty=format:%s|%h|%an|%ad", "--date=short"]
    if since:
        cmd.append(f"{since}..HEAD")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError as e:
        print(f"Error getting git log: {e}")
        return []

def parse_commit_message(message):
    """Parse a commit message to extract type, scope, and description."""
    # Match conventional commit format: type(scope): description
    pattern = r"^(?P<type>\w+)(?:\((?P<scope>[\w-]+)\))?: (?P<description>.+)$"
    match = re.match(pattern, message)
    
    if match:
        commit_type = match.group("type").lower()
        scope = match.group("scope") if match.group("scope") else None
        description = match.group("description")
        
        # Map to known types or use "other"
        if commit_type not in COMMIT_TYPES:
            commit_type = "other"
            
        return commit_type, scope, description
    
    # If not in conventional format, categorize as "other"
    return "other", None, message

def categorize_commits(commits):
    """Categorize commits by type."""
    categorized = defaultdict(list)
    
    for commit in commits:
        if not commit:
            continue
            
        parts = commit.split("|")
        if len(parts) < 4:
            continue
            
        message, hash_id, author, date = parts
        commit_type, scope, description = parse_commit_message(message)
        
        entry = {
            "description": description,
            "scope": scope,
            "hash": hash_id,
            "author": author,
            "date": date
        }
        
        categorized[commit_type].append(entry)
    
    return categorized

def format_changelog(categorized_commits, version=None):
    """Format categorized commits into a changelog."""
    if not version:
        version = f"v{datetime.now().strftime('%Y.%m.%d')}"
    
    date = datetime.now().strftime("%Y-%m-%d")
    changelog = [f"# {version} ({date})\n"]
    
    # Add each category of changes
    for commit_type, display_name in COMMIT_TYPES.items():
        commits = categorized_commits.get(commit_type, [])
        if not commits:
            continue
            
        changelog.append(f"## {display_name}\n")
        
        for commit in commits:
            description = commit["description"]
            scope = commit["scope"]
            hash_id = commit["hash"]
            
            # Format the entry
            entry = f"- {description}"
            if scope:
                entry = f"- **{scope}:** {description}"
                
            entry += f" ({hash_id})"
            changelog.append(entry)
        
        changelog.append("")  # Add blank line between sections
    
    return "\n".join(changelog)

def update_changelog_file(new_content, filename="CHANGELOG.md"):
    """Update the CHANGELOG.md file with new content."""
    existing_content = ""
    
    # Read existing content if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = f.read()
    
    # Combine new and existing content
    full_content = new_content
    if existing_content:
        # Check if there's content after the first section
        match = re.search(r"#\s+v[\d\.]+.*?\n\n(.*)", existing_content, re.DOTALL)
        if match:
            full_content += "\n\n" + match.group(1)
        else:
            full_content += "\n\n" + existing_content
    
    # Write back to file
    with open(filename, "w") as f:
        f.write(full_content)
    
    print(f"✅ Updated {filename}")

def main():
    """Main function to generate changelog."""
    args = parse_args()
    
    print("Generating changelog...")
    commits = get_git_log(args.since)
    
    if not commits or commits[0] == '':
        print("No commits found. Make sure the repository has commits.")
        return
    
    print(f"Found {len(commits)} commits")
    categorized = categorize_commits(commits)
    
    changelog = format_changelog(categorized, args.version)
    update_changelog_file(changelog)
    
    print("Changelog generated successfully!")

if __name__ == "__main__":
    main()
