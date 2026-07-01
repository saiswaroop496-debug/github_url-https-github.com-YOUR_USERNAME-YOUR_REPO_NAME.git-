# auto_deploy.py
"""
Self-deployment module. Call deploy() after train_test.py completes.
Automatically commits latest model artifacts and pushes to GitHub,
triggering a Streamlit Cloud redeploy.
"""

import subprocess
import sys
import os
import json
from datetime import datetime
from pathlib import Path


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and print output."""
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    if result.returncode != 0 and check:
        print(f"  [ERROR] STDERR: {result.stderr.strip()}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def check_git_installed():
    result = run("git --version", check=False)
    if result.returncode != 0:
        raise EnvironmentError("Git is not installed. Install from https://git-scm.com/")


def init_repo_if_needed(repo_dir: Path):
    """Initialize git repo if not already done."""
    git_dir = repo_dir / ".git"
    if not git_dir.exists():
        print("  [SETUP] Initializing git repository...")
        run("git init")
        run("git branch -M main")


def write_gitignore(repo_dir: Path):
    """Ensure .gitignore exists with correct exclusions."""
    gi_path = repo_dir / ".gitignore"
    entries = [
        ".glicko_cache/",
        "*.pkl",
        "__pycache__/",
        "*.pyc",
        ".env",
        "*.log",
        ".DS_Store",
    ]
    existing = gi_path.read_text() if gi_path.exists() else ""
    new_lines = [e for e in entries if e not in existing]
    if new_lines:
        with open(gi_path, "a") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"  [OK] .gitignore updated with {len(new_lines)} new entries")


def write_streamlit_secrets(api_key: str, repo_dir: Path):
    """Write secrets to .streamlit/secrets.toml (gitignored)."""
    secrets_dir = repo_dir / ".streamlit"
    secrets_dir.mkdir(exist_ok=True)

    secrets_path = secrets_dir / "secrets.toml"
    secrets_path.write_text(f'API_KEY = "{api_key}"\n')
    print("  [OK] .streamlit/secrets.toml written")

    # Make sure secrets.toml is never committed
    gi = repo_dir / ".gitignore"
    content = gi.read_text() if gi.exists() else ""
    if ".streamlit/secrets.toml" not in content:
        with open(gi, "a") as f:
            f.write("\n.streamlit/secrets.toml\n")


def save_model_metadata(metrics: dict, repo_dir: Path):
    """
    Save latest validation metrics as model_card.json.
    This file IS committed  Streamlit app reads it to display live stats.
    """
    card = {
        "version": "V7.0",
        "deployed_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": metrics.get("accuracy", 0),
        "log_loss": metrics.get("log_loss", 0),
        "brier": metrics.get("brier", 0),
        "ece": metrics.get("ece", 0),
        "fold_std": metrics.get("fold_std", 0),
        "draw_recall": metrics.get("draw_recall", 0),
        "n_matches": metrics.get("n_matches", 0),
    }
    card_path = repo_dir / "model_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    print(f"  [OK] model_card.json written: accuracy={card['accuracy']:.3f}")


def set_remote(github_url: str):
    """Set or update the GitHub remote origin."""
    result = run("git remote get-url origin", check=False)
    if result.returncode == 0:
        current = result.stdout.strip()
        if current != github_url:
            run(f"git remote set-url origin {github_url}")
            print(f"  [OK] Remote updated to {github_url}")
    else:
        run(f"git remote add origin {github_url}")
        print(f"  [OK] Remote set to {github_url}")


def commit_and_push(github_url: str, version_tag: str = "V6.0"):
    """Stage all changes, commit with auto-timestamp, push to main."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    commit_msg = f"[auto] {version_tag} model update  {timestamp}"

    run("git add .")

    # Check if there's anything to commit
    result = run("git diff --cached --quiet", check=False)
    if result.returncode == 0:
        print("  [INFO] Nothing new to commit  model unchanged since last push")
        return False

    run(f'git commit -m "{commit_msg}"')
    
    pat = os.environ.get("GITHUB_PAT")
    if pat:
        # Strip https:// and inject token
        push_url = github_url.replace("https://", f"https://{pat}@")
        run(f"git push -u {push_url} main")
    else:
        run("git push -u origin main")
        
    print(f"  [DEPLOY] Pushed to GitHub: {commit_msg}")
    return True


def deploy(
    metrics: dict,
    github_url: str,
    api_key: str = "",
    version_tag: str = "V7.0",
    repo_dir: Path = None,
):
    """
    Main entry point. Call this at the end of train_test.py.

    Args:
        metrics:      Dict from your walk-forward validation output
        github_url:   e.g. "https://github.com/yourname/worldcup-v6.git"
        api_key:      Your RapidAPI key (written to secrets.toml, never committed)
        version_tag:  e.g. "V6.0", "V6.1"
        repo_dir:     Path to project root (defaults to cwd)
    """
    if repo_dir is None:
        repo_dir = Path.cwd()

    os.chdir(repo_dir)

    print("\n" + "=" * 55)
    print("  [AUTO-DEPLOY] FIFA Quant Engine")
    print("=" * 55)

    try:
        check_git_installed()
        init_repo_if_needed(repo_dir)
        write_gitignore(repo_dir)

        if api_key:
            write_streamlit_secrets(api_key, repo_dir)

        save_model_metadata(metrics, repo_dir)
        set_remote(github_url)

        pushed = commit_and_push(github_url, version_tag)

        if pushed:
            print("\n  [OK] DEPLOYMENT COMPLETE")
            print("  Streamlit Cloud will auto-redeploy within ~60 seconds.")
            print("  Monitor at: https://share.streamlit.io/")
        else:
            print("\n  [OK] Repository up-to-date. No redeploy triggered.")

    except EnvironmentError as e:
        print(f"\n  [ERROR] Environment error: {e}")
        print("  Fix the issue above and re-run train_test.py")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n  [ERROR] Deploy failed: {e}")
        print("  Check your GitHub URL and SSH/token setup below.")
        _print_auth_help(github_url)
        sys.exit(1)

    print("=" * 55 + "\n")


def _print_auth_help(github_url: str):
    print("""
   GitHub Auth Setup (one-time only) 
  Option A  Personal Access Token (recommended):
    1. Go to GitHub  Settings  Developer Settings
        Personal Access Tokens  Tokens (classic)  Generate
    2. Select scope: repo (full control)
    3. Copy the token
    4. Run:  git credential-manager configure
       Then push once manually  Windows will prompt for token

  Option B  SSH Key:
    1. Run: ssh-keygen -t ed25519 -C "your@email.com"
    2. Add ~/.ssh/id_ed25519.pub to GitHub SSH keys
    3. Change remote URL:
""")
    ssh_url = github_url.replace("https://github.com/", "git@github.com:")
    if not ssh_url.endswith(".git"):
        ssh_url += ".git"
    print(f"       git remote set-url origin {ssh_url}")
