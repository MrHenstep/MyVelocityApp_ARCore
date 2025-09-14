# A0_files_clear_remote.py
# Purpose: Clear a specific remote directory on an Android device using ADB.
# Note: Requires Android Debug Bridge (adb) to be installed and available in PATH,
# and a device connected with USB debugging enabled.

import subprocess  # Used to execute the adb shell command

# Absolute path on the Android device to clear.
# Adjust this if your app/package or target directory is different.
device_path = "/sdcard/Android/data/com.google.ar.core.codelab.rawdepth/files/exported"

# Execute 'rm -rf <path>/*' on the device to remove all files and subdirectories inside.
# - "adb shell" runs the command on the connected Android device.
# - "rm -rf" forcefully and recursively removes files and directories.
# - capture_output=True captures stdout/stderr for logging.
# - text=True decodes output as strings instead of bytes.
result = subprocess.run(
    ["adb", "shell", "rm", "-rf", device_path + "/*"],
    capture_output=True,
    text=True
)

# Check the exit status; 0 indicates success.
if result.returncode == 0:
    print("Directory cleared successfully.")
else:
    # Print detailed diagnostics to help troubleshoot failures.
    print("Error clearing directory:", result.stderr)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
