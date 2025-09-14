# clear the remote directory on an Android device using ADB
"""
A utility script to clear all files in a specified local Windows directory using the native `del` command.
Overview:
- Constructs and executes a shell command `del /Q "<folder>\*"` via `subprocess.run` to silently delete all files in the target folder.
- Does not delete subdirectories or their contents; only files directly within the specified directory are removed.
- Despite the header comment referencing ADB, this script operates on the local filesystem and does not interact with an Android device.
Configuration:
- local_folder (str): Absolute path to the target directory whose files will be deleted. Update this value to point to the desired folder.
Platform and Requirements:
- Windows-only: Relies on the `del` command available in Windows Command Prompt.
- Python 3 with the standard library `subprocess` module.
Behavior and Side Effects:
- Irreversibly removes files in the specified folder.
- Requires appropriate filesystem permissions; otherwise, some deletions may fail silently or raise errors depending on system configuration.
Security and Safety Notes:
- Shell execution is enabled (`shell=True`); ensure `local_folder` is a trusted, controlled value to avoid command injection risks.
- Double-check the `local_folder` path before running to prevent accidental data loss.
- Consider adding logging, dry-run options, and error handling for production use.
Examples:
- To clear exports before regenerating build artifacts, set `local_folder` to the export directory, then run the script.
Potential Enhancements:
- Cross-platform support (e.g., using Python's `os` and `pathlib` to delete files portably).
- Optionally include recursive deletion of subdirectories (use with extreme caution).
- Add confirmation prompts, verbose output, and exception handling with return codes.
"""
import subprocess

# Define the path on the device to clear
local_folder = r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\exported"

command = f'del /Q "{local_folder}\\*"'

# Clear the local folder first
subprocess.run(command, shell=True)

