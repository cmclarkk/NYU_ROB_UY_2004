#!/usr/bin/env python3
"""
Interactive script to download a policy and launch the neural controller.

This script:
1. Prompts for a run number to download a specific policy from wandb
2. Downloads the policy if a run number is provided (or skips if Enter is pressed)
3. Changes to the home directory
4. Launches the neural controller with ros2 launch

Usage:
    python3 deploy.py
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DOWNLOAD_SCRIPT = SCRIPT_DIR / "download_latest_policy.py"
HOME_DIR = Path.home()


def download_policy():
    """Prompt user for run number and download policy if specified"""
    
    print("=" * 70)
    print("Policy Download")
    print("=" * 70)
    print()
    
    # Check if download script exists
    if not DOWNLOAD_SCRIPT.exists():
        print(f"⚠️  Download script not found: {DOWNLOAD_SCRIPT}")
        print("   Skipping policy download...")
        return True
    
    # Prompt for run number
    print("Enter run number to download a specific policy")
    print("(Press Enter to keep the current policy)")
    print()
    
    try:
        user_input = input("Run number: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n⚠️  Cancelled by user")
        return False
    
    # If user pressed Enter without typing anything, skip download
    if not user_input:
        print("\nℹ️  No run number provided. Using current policy.")
        return True
    
    # Validate input is a number
    try:
        run_number = int(user_input)
    except ValueError:
        print(f"\n❌ Invalid input: '{user_input}' is not a valid number")
        return False
    
    # Download the policy
    print(f"\n📥 Downloading policy from run {run_number}...")
    print()
    
    try:
        result = subprocess.run(
            ["python3", str(DOWNLOAD_SCRIPT), "--run_number", str(run_number)],
            cwd=str(SCRIPT_DIR),
            check=False
        )
        
        if result.returncode == 0:
            print("\n✅ Policy downloaded successfully!")
            return True
        else:
            print(f"\n❌ Failed to download policy (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ Error downloading policy: {e}")
        return False


def launch_neural_controller():
    """Launch the neural controller with ros2"""
    
    print("\n" + "=" * 70)
    print("Launching Neural Controller")
    print("=" * 70)
    print()
    print(f"📁 Changing to home directory: {HOME_DIR}")
    print(f"🚀 Running: ros2 launch neural_controller launch.py")
    print()
    print("=" * 70)
    print()
    
    try:
        # Change to home directory and launch
        os.chdir(HOME_DIR)
        
        # Run ros2 launch - this will run in foreground
        result = subprocess.run(
            ["ros2", "launch", "neural_controller", "launch.py"],
            check=False
        )
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Launch interrupted by user (Ctrl+C)")
        return 130
    except Exception as e:
        print(f"\n❌ Error launching neural controller: {e}")
        return 1


def main():
    """Main entry point"""
    
    print("\n" + "=" * 70)
    print("Neural Controller Launch Script")
    print("=" * 70)
    print()
    
    # Step 1: Download policy (optional)
    download_success = download_policy()
    
    if not download_success:
        print("\n❌ Policy download failed. Continue anyway? (y/N): ", end="")
        try:
            response = input().strip().lower()
            if response not in ['y', 'yes']:
                print("Aborting.")
                return 1
        except (KeyboardInterrupt, EOFError):
            print("\nAborting.")
            return 1
    
    # Step 2: Launch the neural controller
    return_code = launch_neural_controller()
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())
