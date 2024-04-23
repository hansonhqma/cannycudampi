import cv2
import numpy as np
import sys

video_path = sys.argv[1]  # Replace with the path to your video

f = open('video_byte_data.dat','wb')
cap = cv2.VideoCapture(video_path)

frame_count = 0
height = 0 
width = 0
# Loop through each frame of the video
while frame_count < 512:
    # Read the next frame
    ret, frame = cap.read()
    
    # If there are no more frames to read, break out of the loop
    if not ret:
        break

    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale_frame,(3,3),0)

    # Convert the grayscale frame to a numpy array
    pixel_data = np.array(blurred)
    if height == 0:
        height, width = pixel_data.shape[:2]
    pixel_data = pixel_data.flatten()
    frame_bytes = bytearray(pixel_data)
    f.write(frame_bytes)
    frame_count += 1

# rename dat file with this information
print("Height: {}, Width: {}, FrameCount: {}".format(height, width, frame_count))

# Release the video capture object
cap.release()
f.close()