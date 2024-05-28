import os
from PIL import Image

def rename_images(directory):
    # Check if the specified directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in '{directory}'.")
        return
    
    # Rename each image file
    for i, filename in enumerate(image_files, start=1):
        # Construct the new filename
        new_filename = f"real_test_img_{i}.jpg"
        
        # Construct the full paths of the old and new files
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        try:
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'.")
        except Exception as e:
            print(f"Error renaming '{filename}': {e}")


rename_images(r"C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\test\real_test")