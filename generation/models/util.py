import os
from PIL import Image
import base64
import io
def check_and_create_directory(dir_path):
    """Check if a directory exists, and if not, create it.
    
    Args:
    - dir_path: The path of the directory to check and create if necessary.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def jpg_path2base64(img_path):
    # Load the image from the specified path
    with Image.open(img_path) as image:
        # Convert the image to JPEG format
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        
        # Encode these bytes to Base64
        img_str = base64.b64encode(buffered.getvalue())
        
        # Return the Base64 string, decoded to make it a readable string
        return img_str.decode('utf-8')