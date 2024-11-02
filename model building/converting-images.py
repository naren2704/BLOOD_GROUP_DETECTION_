from PIL import Image
import os

# Define the source and destination directories
source_dir = r'C:\Users\SEC\Desktop\archive\dataset_blood_group\integrated'
destination_dir = r'C:\Users\SEC\Desktop\archive\dataset_blood_group\integrated_converted'

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop through all BMP files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.BMP'):
        # Open the BMP image
        bmp_image_path = os.path.join(source_dir, filename)
        with Image.open(bmp_image_path) as img:
            # Convert and save as JPEG
            jpeg_image_path = os.path.join(destination_dir, filename.replace('.BMP', '.jpg'))
            img.convert('RGB').save(jpeg_image_path, 'JPEG')
            print(f"Converted {bmp_image_path} to {jpeg_image_path}")
