#!/usr/bin/env python3
"""
Wrapper script that properly activates htp_env before running the planning interface.
This ensures Pinocchio 3.4.0 is available instead of the system's old version.
"""

import os
import sys
import subprocess

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.join(script_dir, "..", "..")
    
    # Path to the actual planning interface
    planning_interface_path = os.path.join(script_dir, "..", "iiwa_planning_interface.py")
    
    # Path to htp_env activation script
    htp_env_activate = os.path.expanduser("~/.virtualenvs/htp_env/bin/activate")
    
    if not os.path.exists(htp_env_activate):
        print("❌ htp_env not found at ~/.virtualenvs/htp_env/bin/activate")
        print("Falling back to system Python...")
        # Run without virtual environment
        subprocess.run([sys.executable, planning_interface_path])
        return
    
    # Create a bash command that activates htp_env and runs the planning interface
    bash_command = f"""
    source {htp_env_activate}
    cd {workspace_root}
    export PYTHONPATH={workspace_root}:$PYTHONPATH
    python3 {planning_interface_path}
    """
    
    print("🚀 Starting planning interface with htp_env...")
    print(f"Activating: {htp_env_activate}")
    
    # Execute the command
    result = subprocess.run(['bash', '-c', bash_command])
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
