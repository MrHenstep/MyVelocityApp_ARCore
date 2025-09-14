import subprocess
"""
A2_files_pull.py
This script pulls a directory of files from an Android device to a local Windows PC using adb.
It constructs the device path from a base directory and a directory name, then uses subprocess.run
to execute `adb pull` and prints the command's STDOUT and STDERR for visibility.
Key behaviors:
- Defines:
    - directory_name: The subdirectory on the device to pull (e.g., "exported").
    - device_path: The full path on the Android device under /sdcard/... where files reside.
    - local_path: The destination directory on the host PC (Windows path).
- Optionally demonstrates how to list the remote device path with `adb shell ls` (commented out).
- Executes `adb pull <device_path> <local_path>` and prints both STDOUT and STDERR from the process.
Prerequisites:
- Android Debug Bridge (adb) installed and available on PATH.
- An Android device or emulator connected and authorized with USB debugging enabled.
- The specified device_path exists and is accessible (scoped storage and app permissions may apply).
- Sufficient permissions on the host machine to write to local_path.
Usage:
- Adjust directory_name and local_path as needed.
- Ensure the device path exists, optionally enabling the commented `adb shell ls` to verify.
- Run the script; it will attempt to pull the directory and print the results.
Outputs and side effects:
- No return value; prints the subprocess outputs to STDOUT.
- Creates or updates files under local_path by mirroring the directory from the device.
Error handling:
- Does not raise on non-zero exit codes by default; instead, prints STDERR for diagnostics.
- If adb is not found or fails to spawn, subprocess may raise an exception before printing.
"""

# Define the source path on the device and the destination on your PC
directory_name = "exported"
device_path = "/sdcard/Android/data/com.google.ar.core.codelab.rawdepth/files/" + directory_name    
local_path = r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app"

# Run the adb pull command
result = subprocess.run(["adb", "pull", device_path, local_path], capture_output=True, text=True)

# Check output
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)


