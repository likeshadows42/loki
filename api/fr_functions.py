# ==============================================================================
#                        FACE RECOGNITION FUNCTIONS
# ==============================================================================

# Module / package imports
import os
import shelve
import numpy              as np
import tensorflow         as tf

from tqdm                    import tqdm
from deepface                import DeepFace
from deepface.DeepFace       import build_model as build_verifier
from deepface.basemodels     import Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons        import functions, realtime, distance as dst
from utility_functions       import create_dir
from api_functions           import API_DIR

from deepface.detectors.FaceDetector import build_model as build_detector

tf_version = int(tf.__version__.split(".")[0])

if tf_version == 2:
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    from tensorflow.keras.preprocessing import image
else:
    from keras.preprocessing import image

api_root_dir = API_DIR

# ------------------------------------------------------------------------------

def batch_build_detectors(detector_names, show_prog_bar=True, verbose=True):
    """
    Builds batches of face detectors. The face detectors to be built are
    specified by the 'detector_names' list of names. If a face detectors cannot
    be built (or results in an error), it is simply skipped.
    
    Inputs:
        1. detector_names - list (of strings) of face detectors names
        2. show_prog_bar - boolean toggling a progress bar
            ([show_prog_bar=True])
        
    Outputs:
        1. dictionary of built face detector models
        
    Signature:
        detectors = batch_build_detectors(detector_names, show_prog_bar=True)
    """
    # Creates the progress bar
    n_detectors    = len(detector_names)
    disable_option = not show_prog_bar
    pbar           = tqdm(range(0, n_detectors), desc='Saving verifiers',
                        disable=disable_option)
    
    # 
    detectors      = {}
    for index, detector_name in zip(pbar, detector_names):
        try:
            cur_detector = build_detector(detector_name)
            detectors[detector_name] = cur_detector
            
        except Exception as excpt:
            pass
    
    return detectors

# ------------------------------------------------------------------------------

def batch_build_verifiers(verifier_names, show_prog_bar=True):
    """
    Builds batches of face verifiers. The face verifiers to be built are
    specified by the 'verifier_names' list of names. If a face verifier cannot
    be built (or results in an error), it is simply skipped.
    
    Inputs:
        1. verifier_names - list (of strings) of face verifier names
        2. show_prog_bar - boolean toggling a progress bar
            ([show_prog_bar=True])
        
    Outputs:
        1. dictionary of built face verifier models
        
    Signature:
        verifiers = batch_build_verifiers(verifier_names, show_prog_bar=True)
    """
    # Creates the progress bar
    n_verifiers    = len(verifier_names)
    disable_option = not show_prog_bar
    pbar           = tqdm(range(0, n_verifiers), desc='Building verifiers',
                        disable = disable_option)

    # 
    verifiers      = {}
    for index, verifier_name in zip(pbar, verifier_names):
        try:
            cur_verifier = build_verifier(verifier_name)
            verifiers[verifier_name] = cur_verifier

        except Exception as excpt:
            pass
        
    return verifiers

# ------------------------------------------------------------------------------

def saved_verifier_exists(verifier_name, save_dir=''):
    """
    Checks if a saved verifier exists. Does that by comparing the
    'verifier_name' against the name of all files in 'save_dir'. If 'save_dir'
    is not specified, uses the global variable 'api_root_dir' to determine the
    save directory.
    
    Inputs:
        1. verifier_name - string with the name of the verifier.
        2. save_dir - string with the full path of the save directory
            ([save_dir='']).
        
    Output:
        1. Boolean value indicating if saved verifier exists or not.
    
    Signature:
        verifier_exists = saved_verifier_exists(verifier_name, save_dir='')
    """
    if len(save_dir) == 0:
        global api_root_dir
        save_dir = os.path.join(api_root_dir, 'saved_models', 'verifiers')
    
    is_in = False
    for file_name in os.listdir(save_dir):
        is_in = is_in or verifier_name in file_name
    
    return is_in

# ------------------------------------------------------------------------------

def save_face_verifiers(verifiers, show_prog_bar=True, overwrite=False):
    """
    Saves all face verifier models specified in 'verifiers' using the
    'shelve' (persistent storage) module. Uses the global 'api_root_dir'
    variable to determine the root directory in which the models will be
    saved.
    
    This function saves each model individually and skips any model that 
    fails to save (or produces an error). For each model that already exists,
    this function does not overwrite it unless the 'overwrite' flag is set
    to True.
    
    Inputs:
        1. verifiers - dictionary containing the build face verifier models.
        2. show_prog_bar - boolean that toggles the progress bar on or off.
        3. overwrite - boolean that indicates if the function should overwrite
                        any saved models.
                        
    Outputs:
        1. returns a status flag of True if any model fails to save (otherwise
            returns False)
    
    Signature:
        status = save_face_verifiers(verifiers, show_prog_bar=True,
                                        overwrite=False)
    """
    # Create necessary directories if they do not exist based on
    # the global api root directory
    create_dir(os.path.join(api_root_dir, 'saved_models'))
    save_dir = os.path.join(api_root_dir, 'saved_models', 'verifiers')
    create_dir(save_dir)
    
    # Creates the progress bar
    n_verifiers    = len(verifiers)
    disable_option = not show_prog_bar
    pbar           = tqdm(range(0, n_verifiers), desc='Saving verifiers',
                        disable = disable_option)
    
    # Loops through each verifier
    no_errors_flag = False # False means no errors
    for index, verifier_items in zip(pbar, verifiers.items()):
        # Gets the name of the verifier and the verifier object
        name     = verifier_items[0]
        verifier = verifier_items[1]
        
        if not saved_verifier_exists(name) or overwrite:
            try:
                # Creates the file's full path
                file_fp  = os.path.join(save_dir, name)
        
                # Opens a persistent dictionary, saves the model
                # as a dictionary then closes it
                with shelve.open(file_fp) as d:
                    d[name]  = verifier
            
            except Exception as excpt:
                no_errors_flag = True
            
    return no_errors_flag

# ------------------------------------------------------------------------------

def load_face_verifier(verifier_names, save_dir='', show_prog_bar=True):
    """
    Loads all face verifier models specified in 'verifier_names'. Uses
    the global 'api_root_dir' variable to determine the root directory
    from which the models will be loaded if the 'save_dir' directory is
    empty (i.e. save_dir=''). Otherwise, attempts to load the models
    from 'save_dir'.
    
    This function loads each model individually and skips any model that 
    fails to load (or produces an error).
    
    Inputs:
        1. verifier_name - string with the name of the face verifiers.
        2. save_dir      - string with the full path of the save directory
            ([save_dir='']).
        3. show_prog_bar - boolean that toggles the progress bar on or off.
                        
    Outputs:
        1. returns a status flag of True if any model fails to save (otherwise
            returns False)
            
    Signature:
        models = load_face_verifier(verifier_names, save_dir='',
                                    show_prog_bar=True)
    """
    # 
    if len(save_dir) == 0:
        global api_root_dir
        save_dir = os.path.join(api_root_dir, 'saved_models', 'verifiers')
    #all_files = os.listdir(save_dir)
    
    # Ensures that the verifier_names is a list (even a single name is provided)
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]
    
    # Creates the progress bar
    n_verifiers    = len(verifier_names)
    disable_option = not show_prog_bar
    pbar           = tqdm(range(0, n_verifiers), desc='Loading verifiers',
                            disable = disable_option)
    
    models = {}
    for index, verifier_name in zip(pbar, verifier_names):
        # Checks if the face verifier model exists
        if saved_verifier_exists(verifier_name):
            # Loads the model
            file_fp = os.path.join(save_dir, verifier_name)
            with shelve.open(file_fp) as model:
                models[verifier_name] = model[verifier_name]
        else:
            # Otherwise, tries to build model from scratch & save it
            try:
                models[verifier_name] = build_verifier(verifier_name)
                save_face_verifiers(models[verifier_name], api_root_dir,
                                    show_prog_bar=False, overwrite=False)
            except Exception as excpt:
                pass
            
    return models

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
