# clear the remote directory on an Android device using ADB
import subprocess

# Define the path on the device to clear
local_folder = r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\exported"

command = f'del /Q "{local_folder}\\*"'

# Clear the local folder first
subprocess.run(command, shell=True)

