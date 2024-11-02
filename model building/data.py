import os
import shutil

# Define the source directories for each blood group dataset
source_dirs = {
    'A-': r'C:\Users\SEC\Desktop\archive\dataset_blood_group\A-',
    'B-': r'C:\Users\SEC\Desktop\archive\dataset_blood_group\B-',
    'AB-': r'C:\Users\SEC\Desktop\archive\dataset_blood_group\AB-',
    'O-': r'C:\Users\SEC\Desktop\archive\dataset_blood_group\O-',
    'A+': r'C:\Users\SEC\Desktop\archive\dataset_blood_group\A+',  # Example additional groups
    'B+': r'C:\Users\SEC\Desktop\archive\dataset_blood_group\B+',
    'AB+': r'C:\Users\SEC\Desktop\archive\dataset_blood_group\AB+',
    'O+': r'C:\Users\SEC\Desktop\archive\dataset_blood_group\O+'
}

# Define the destination directory where all the integrated images will be stored
destination_dir = r'C:\Users\SEC\Desktop\archive\dataset_blood_group\integrated'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop through each source directory and move/copy the images to the integrated folder
for blood_group, source_dir in source_dirs.items():
    if os.path.exists(source_dir):  # Check if the source directory exists
        for filename in os.listdir(source_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg','.BMP')):  # Process only image files
                # Create a new filename to indicate the blood group (e.g., A_image1.jpg)
                new_filename = f"{blood_group}_{filename}"
                # Define the full path for the source and destination files
                source_path = os.path.join(source_dir, filename)
                destination_path = os.path.join(destination_dir, new_filename)
                
                # Copy the file to the destination
                shutil.copy(source_path, destination_path)
                print(f"Copied {source_path} to {destination_path}")
    else:
        print(f"Source directory {source_dir} does not exist.")

print("Dataset integration completed.")


