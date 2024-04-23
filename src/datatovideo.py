import cv2
import numpy as np
import sys

data_file = sys.argv[1]
f = open(data_file,'rb')

height,width,frame_count = map(int, data_file.split('.')[0].split('_')[:3])
# print(height,width,frame_count)
output_path = f"{height}_{width}_{frame_count}.mp4"  # Path where you want to save the output video

video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'h264'), 30.0, (width, height))

byte_data = f.read()
pixel_data = np.frombuffer(byte_data, dtype=np.uint8)
total_pixels = len(pixel_data)
frame_size = height*width

start = 0
end = frame_size
while end <= total_pixels:
    frame = pixel_data[start:end].reshape((height, width))
    values = np.array(frame, dtype=np.uint8)
    video.write(values)
    start += frame_size
    end += frame_size

video.release()
