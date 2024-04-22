from PIL import Image
import numpy as np

def convert_byte_to_grayscale(image_file):
    # Parse the width and height from the file name
    # Assume the file name is in the format {width}_{height}_res.dat
    width, height = map(int, image_file.split('_')[:2])

    # Open the file in binary mode for reading
    with open(image_file, 'rb') as file:
        # Read the entire file content as byte data
        byte_data = file.read()
        
        # Convert the byte data to a numpy array with dtype uint8
        pixel_data = np.frombuffer(byte_data, dtype=np.uint8)
        
        # Check if the total number of pixels matches the expected size
        expected_size = height * width
        if len(pixel_data) != expected_size:
            raise ValueError(f"Unexpected data size: expected {expected_size} bytes, but got {len(pixel_data)} bytes")
        
        # Reshape the pixel data into the appropriate dimensions
        pixel_data = pixel_data.reshape((height, width))

        # Create a grayscale image from the pixel data
        grayscale_img = Image.fromarray(pixel_data, mode='L')

        # Save the grayscale image
        output_image_path = f"{width}_{height}.png"
        grayscale_img.save(output_image_path)

        print(f"Grayscale image saved as '{output_image_path}'")

# Example usage
image_file = "2340_1755_res.dat"  # Change this to your byte data file path
convert_byte_to_grayscale(image_file)
