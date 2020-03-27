"""
Use this script to convert MP4 files into Wav files.
"""

import os
import subprocess

# Loop through the filesystem
for root, dirs, files in os.walk("D:\\CSCI 599 data\\Videos\\", topdown=False):
    # Loop through files
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
            
            # Using ffmpeg to convert the mp4 in wav
            # Example command: "ffmpeg -i C:/test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
#            print(name[:-3])
            command = "ffmpeg -i " +"\"" + root[0:] + name +"\"" + " " + "-ab 160k -ac 1 -ar 44100 "+"\"" + root[0:] +  name[:-3] + "wav"+"\""
#            print("")            
#            # Execute conversion
            try:
                subprocess.call(command, shell=True)
                print("Success")
#                
#            # Skip the file in case of error
            except ValueError:
                continue
