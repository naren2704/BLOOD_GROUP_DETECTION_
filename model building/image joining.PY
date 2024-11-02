import os
import shutil

# Path to the folder containing all images
source_dir = r'C:\Users\SEC\Desktop\archive\dataset_blood_group\integrated_converted'
# Destination directory for organized images
destination_dir = r'C:\Users\SEC\Desktop\archive\dataset_blood_group\organized'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Define blood groups and create subdirectories
blood_groups = ['A+', 'B+', 'AB+', 'O+','A-', 'B-', 'AB-', 'O-']

for blood_group in blood_groups:
    os.makedirs(os.path.join(destination_dir, blood_group), exist_ok=True)

# Move files to corresponding subdirectories
for filename in os.listdir(source_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Process only image files
        # Print the current file being processed
        print(f"Processing file: {filename}")
        
        # Determine the blood group from the filename
        for blood_group in blood_groups:
            if filename.startswith(blood_group):  # Check if filename starts with blood group
                # Source and destination paths
                source_path = os.path.join(source_dir, filename)
                destination_path = os.path.join(destination_dir, blood_group, filename)
                
                # Move the file
                shutil.move(source_path, destination_path)
                print(f"Moved {source_path} to {destination_path}")
                break  # Exit loop after moving
        else:
            print(f"Skipped {filename}: does not match any blood group.")
    else:
        print(f"Skipped {filename}: Unsupported file format.")

print("Image organization completed.")
