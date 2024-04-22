from PIL import Image

def grayscale_and_write_bytes(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to grayscale
        grayscale_img = img.convert("L")

        # Get the width and height of the grayscale image
        width, height = grayscale_img.size

        # Get the pixel data (as bytes) from the grayscale image
        pixel_data = list(grayscale_img.getdata())

        # Convert pixel data to bytes
        byte_data = bytes(pixel_data)
        
        # Name output file by width_height of image
        output_file = f"{width}_{height}.dat"

        # Write the byte data to the specified output file
        with open(output_file, 'wb') as file:
            file.write(byte_data)

        print(f"Byte data written to '{output_file}'")

# Example usage
image_path = "chess.jpg"  # Change this to your image file path
grayscale_and_write_bytes(image_path)
