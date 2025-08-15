# clear the remote directory on an Android device using ADB
import subprocess

# Define the path on the device to clear
device_path = "/sdcard/Android/data/com.google.ar.core.codelab.rawdepth/files/exported"

# Run the adb shell command to clear the remote directory
result = subprocess.run(["adb", "shell", "rm", "-rf", device_path + "/*"], capture_output=True, text=True)

if result.returncode == 0:
    print("Directory cleared successfully.")
else:
    print("Error clearing directory:", result.stderr)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)


