import subprocess

# Define the source path on the device and the destination on your PC
directory_name = "exported"
device_path = "/sdcard/Android/data/com.google.ar.core.codelab.rawdepth/files/" + directory_name    
local_path = r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app"

# run remote ls on the device to check if the path exists, print the output
result = subprocess.run(["adb", "shell", "ls", device_path], capture_output=True, text=True)
# if result.returncode != 0:
#     print("Error accessing device path:", result.stderr)

# # Check if the command was successful
# if result.returncode == 0:
#     print("Device path exists. Listing files:")
#     print(result.stdout)

# # Check output
# print("STDERR:", result.stderr)

# Run the adb pull command
result = subprocess.run(["adb", "pull", device_path, local_path], capture_output=True, text=True)

# Check output
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)


