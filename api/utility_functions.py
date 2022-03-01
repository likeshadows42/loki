# ==============================================================================
#                             UTILITY FUNCTIONS
# ==============================================================================

# Module / package imports
import os
from tqdm                    import tqdm

# ------------------------------------------------------------------------------

def get_image_paths(root_path, file_types=('.jpg', '.png')):
    """
    Gets the full paths of all 'file_types' files in the 'root_path' directory
    and its subdirectories. If the 'root_path' provided points to a file, this
    functions simply returns that path as a list with 1 element.

    Inputs:
        1. root_path  - full path of a directory
        2. file_types - tuple containing strings of file extensions
            ([file_types=('.jpg', '.png')])

    Output:
        1. list containing full path of all files in the directory and its
            subdirectories

    Example call:
        ROOT_PATH = path/to/some/folder/dude
        all_images = get_image_paths(ROOT_PATH, file_types=('.png'))
    """
    # If the root path points to file, simply return it as a list with 1 element
    if os.path.isfile(root_path):
        return [root_path]
    
    # Gets all images in this root directory
    all_images = []
    for root, _junk, files in os.walk(root_path):
        # Processes files in the root directory - can be transformed into a list
        # comprehension but code might lose clarity
        for file in files:
            if file.lower().endswith(file_types):
                exact_path = os.path.join(root, file)
                all_images.append(exact_path)
      
    return all_images

# ------------------------------------------------------------------------------

def create_dir(dir_path):
  """
  Creates a directory at the specified directory path 'dir_path' IF it does not
  exist. Returns a status of 0 is the directory was successfully created and 
  returns a status of 1 if a directory already exists.

  Inputs:
    1. dir_path - directory path of new directory.
    
  Outputs:
    1. status - 0 to indicate success (directory creation) or 1 to indicate
       failure (directory already exists)
    
  Signature:
    status = create_dir(dir_path)
  """
  # Create directory
  try:
    os.makedirs(dir_path)
    status = 0
  except FileExistsError:
    # Directory already exists
    status = 1

  return status

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
