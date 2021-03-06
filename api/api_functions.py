# ==============================================================================
#                             API FUNCTIONS
# ==============================================================================

# Module / package imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import cv2
import faiss
import pickle
import shelve

import numpy                as np
import tensorflow           as tf

from tqdm                    import tqdm
from uuid                    import uuid4
from deepface                import DeepFace
from deepface.commons        import functions, distance as dst
from deepface.DeepFace       import build_model as build_verifier
from deepface.basemodels     import Boosting
from deepface.detectors      import FaceDetector
from keras.preprocessing     import image

# These imports need to be relative to work with FastAPI but need to be absolute
# to work with ipynb?
from api_classes           import Representation

from .                               import global_variables as glb
from deepface.detectors.FaceDetector import build_model      as build_detector

# Local variables
api_root_dir = glb.API_DIR
svd_vrf_dir  = glb.SVD_VRF_DIR

# ------------------------------------------------------------------------------

def detect_faces(img_path, detector_backend = 'opencv', align = True,
                 return_type = 'both', face_detector = None):
    """
    Detects faces in an image (and optionally aligns them).
    
    Inputs:
        1. img_path - image path, base64 image or numpy array image
        2. detector_backend - string corresponding to detector ([opencv],
            ssd, dlib, mtcnn, retinaface, mediapipe).
        3. align - flag indicating if face should be aligned ([align=True]).
        4. return_type - string indicating if faces, regions or both should
            be returned ('faces', 'regions', ['both']).
            
    Outputs:
        If return_type='regions':
            Dictionary containing list of face detections. The face detections
            (or regions of interest - rois) are lists with the format
            [top-left x, top-left y, width, height]. The dictionary key is
            'regions'.
            
        If return_type='faces':
            Dictionary containing list of detected faces. Each detection is an
            image with 'target_size' size (the number of color channels is
            unchanged). The dictionary key is 'faces'.
            
        If return_type='both':
            Dictionary containing both of the above outputs: face detections and
            detected faces. The dictionary keys are 'faces' and 'regions'. 
    
    Signature:
        output = detect_faces(img_path, detector_backend = 'opencv',
                              align = True, return_type = 'both')
    """
    # Raises an error if return type is not 'faces', 'regions' or 'both'.
    # Otherwise, initializes lists.
    if return_type == 'faces' or return_type == 'regions' or \
        return_type == 'both':
        faces = []
        rois  = []
    else:
        raise ValueError("Return type should be 'faces', 'regions' or 'both'.")
    
    # Loads image. Image might be path, base64 or numpy array. Convert it to numpy
    # whatever it is.
    img = functions.load_image(img_path)

    # The detector is stored in a global variable in FaceDetector object.
    # This call should be completed very fast because it will return found in
    # memory and it will not build face detector model in each call (consider for
    # loops)
    if face_detector == None:
        face_detector = FaceDetector.build_model(detector_backend)
    detections = FaceDetector.detect_faces(face_detector, detector_backend,
                                            img, align)

    # Prints a warning and returns an empty dictionary and error if no faces were
    # found, otherwise processes faces & regions
    if len(detections) == 0:
        print('Face could not be detected or the image contains no faces.')

    else:
        # Loops through each face & region pair
        for face, roi in detections:

            # Only process images (faces) if the return type is 'faces' or 'both'
            if return_type == 'faces' or return_type == 'both':
                # Appends processed face
                faces.append(face)
    
            # Only process regions (rois) if the return type is 'regions' or 'both'
            if return_type == 'regions' or return_type == 'both':
                rois.append(roi)

  # ------------------------
  
    if return_type == 'faces':
        return {'faces':faces}
    elif return_type == 'regions':
        return {'regions':rois}
    else:
        assert return_type == 'both', "Return type should be 'both' here."
        return {'faces':faces, 'regions':rois}

# ------------------------------------------------------------------------------

def verify_faces(img1_path, img2_path = '', model_name = 'VGG-Face',
                 distance_metric = 'cosine', model = None,
                 enforce_detection = True, detector_backend = 'opencv',
                 align = True, prog_bar = True, normalization = 'base',
                 threshold = -1):

    """
    This function verifies if an image pair is the same person or different
    ones.

    Parameters:
        img1_path, img2_path: exact image path, numpy array or based64 encoded
        images could be passed. If you are going to call verify function for a
        list of image pairs, then you should pass an array instead of calling
        the function in for loops.

        e.g. img1_path = [
            ['img1.jpg', 'img2.jpg'],
            ['img2.jpg', 'img3.jpg']
        ]

        model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble
        distance_metric (string): cosine, euclidean, euclidean_l2
        model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.
            model = DeepFace.build_model('VGG-Face')
        enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.
        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib
        prog_bar (boolean): enable/disable a progress bar
    Returns:
        Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.
        {
            "verified": True
            , "distance": 0.2563
            , "max_threshold_to_verify": 0.40
            , "model": "VGG-Face"
            , "similarity_metric": "cosine"
        }
    """

    tic = time.time()

    img_list, bulkProcess = functions.initialize_input(img1_path, img2_path)

    resp_objects = []

    #--------------------------------

    if model_name == 'Ensemble':
        model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
        metrics = ["cosine", "euclidean", "euclidean_l2"]
    else:
        model_names = []; metrics = []
        model_names.append(model_name)
        metrics.append(distance_metric)

    #--------------------------------

    if model == None:
        if model_name == 'Ensemble':
            models = Boosting.loadModel()
        else:
            model = DeepFace.build_model(model_name)
            models = {}
            models[model_name] = model
    else:
        if model_name == 'Ensemble':
            Boosting.validate_model(model)
            models = model.copy()
        else:
            models = {}
            models[model_name] = model

    #------------------------------

    disable_option = (False if len(img_list) > 1 else True) or not prog_bar

    pbar = tqdm(range(0, len(img_list)), desc='Verification',
                      disable = disable_option)

    for index in pbar:
        instance = img_list[index]

        if type(instance) == list and len(instance) >= 2:
            img1_path = instance[0]; img2_path = instance[1]

            ensemble_features = []

            for i in model_names:
                custom_model = models[i]

                #img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'mtcnn'
                img1_representation = DeepFace.represent(img_path = img1_path,
                            model_name = model_name, model = custom_model,
                            enforce_detection = enforce_detection,
                            detector_backend = detector_backend,
                            align = align, normalization = normalization)

                img2_representation = DeepFace.represent(img_path = img2_path,
                            model_name = model_name, model = custom_model,
                            enforce_detection = enforce_detection,
                            detector_backend = detector_backend, align = align,
                            normalization = normalization)

                #----------------------
                #find distances between embeddings

                for j in metrics:

                    if j == 'cosine':
                        distance = dst.findCosineDistance(img1_representation,
                                                          img2_representation)
                    elif j == 'euclidean':
                        distance = dst.findEuclideanDistance(\
                                                        img1_representation,
                                                        img2_representation)
                    elif j == 'euclidean_l2':
                        distance = dst.findEuclideanDistance(\
                                    dst.l2_normalize(img1_representation),
                                    dst.l2_normalize(img2_representation))
                    else:
                        raise ValueError("Invalid distance_metric passed - ",
                                         distance_metric)

                    distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
                    #----------------------
                    #decision

                    if model_name != 'Ensemble':
                        if threshold < 0:
                            threshold = dst.findThreshold(i, j)

                        if distance <= threshold:
                            identified = True
                        else:
                            identified = False

                        resp_obj = {"verified": identified,
                                    "distance": distance,
                                    "threshold": threshold,
                                    "model": model_name,
                                    "detector_backend": detector_backend,
                                    "similarity_metric": distance_metric}

                        if bulkProcess == True:
                            resp_objects.append(resp_obj)
                        else:
                            return resp_obj

                    else: #Ensemble

                        #this returns same with OpenFace - euclidean_l2
                        if i == 'OpenFace' and j == 'euclidean':
                            continue
                        else:
                            ensemble_features.append(distance)

            #----------------------

            if model_name == 'Ensemble':

                boosted_tree = Boosting.build_gbm()
                prediction   = boosted_tree.predict(np.expand_dims(\
                                        np.array(ensemble_features), axis=0))[0]

                verified = np.argmax(prediction) == 1
                score = prediction[np.argmax(prediction)]

                resp_obj = {"verified": verified, "score": score,
                   "distance": ensemble_features,
                   "model": ["VGG-Face", "Facenet", "OpenFace", "DeepFace"],
                   "similarity_metric": ["cosine", "euclidean", "euclidean_l2"]}

                if bulkProcess == True:
                    resp_objects.append(resp_obj)
                else:
                    return resp_obj

            #----------------------

        else:
            raise ValueError("Invalid arguments passed to verify function: ",
                             instance)

    #-------------------------

    toc = time.time()

    if bulkProcess == True:

        resp_obj = {}

        for i in range(0, len(resp_objects)):
            resp_item = resp_objects[i]
            resp_obj["pair_%d" % (i+1)] = resp_item

        return resp_obj

# ------------------------------------------------------------------------------

def calculate_similarity(rep1, rep2,
                         metrics=('cosine', 'euclidean', 'euclidean_l2')):
    """
    Calculates the similarity between two face images, represented as two vector
    embeddings. If a metric in 'metrics' is not valid, the function raises a
    Value error.

    Inputs:
        1. rep1 - vector embedding representing the face in image 1

        2. rep2 - vector embedding representing the face in image 2

        3. metrics - tupple containing names of the distance metrics to be used
            ([metrics=('cosine', 'euclidean', 'euclidean_l2')]). Available
            metrics are 'cosine', 'euclidean' and 'euclidean_l2'. If the user
            only requires 1 metric be sure to pass it using this format:
                metrics=('<distance_metric>',), i.e. metrics=('cosine',)

    Outputs:
        1. dictionary containing the metric (key) and its corresponding distance
            (value) or 'similarity'

    Signature:
        distances = calculate_similarity(rep1, rep2,
                         metrics=('cosine', 'euclidean', 'euclidean_l2')
    """
    distances = {} # initializes output dictionary

    # Loops through each metric
    for metric in metrics:
        if metric == 'cosine':
            distances[metric] = dst.findCosineDistance(rep1, rep2)
        elif metric == 'euclidean':
            distances[metric] = dst.findEuclideanDistance(rep1, rep2)
        elif metric == 'euclidean_l2':
            distances[metric] = \
                dst.findEuclideanDistance(dst.l2_normalize(rep1),
                                          dst.l2_normalize(rep2))
        else:
            raise ValueError(f"Invalid metric passed: {metric}")

        distances[metric] = np.float64(distances[metric]) #causes trobule for euclideans in api calls if this is not set (issue #175)
        
    return distances

# ______________________________________________________________________________
#                       UTILITY & GENERAL USE FUNCTIONS
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

# ______________________________________________________________________________
#                DETECTORS & VERIFIERS BUILDING, SAVING & LOADING
# ------------------------------------------------------------------------------

def build_face_verifier(model_name='VGG-Face', distance_metric='cosine', 
                        model=None, verbose=0):
    """
    Builds the face verifier model with 'model_name'. Alternatively, a pre-built
    'model' can be passed. In that case, the function simply stores the model
    and metric names. Also handles the special case of the 'Ensemble' model.
    
    Inputs:
        1. model_name - name of face verifier model ([VGG-Face], Facenet,
                Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, Ensemble)
        2. model - built model (see DeepFace.build_model() for more information)
                or None ([model=None])
                
    Outputs:
        1. dictionary containing model names
        2. dictionary containing metric names
        3. dictionary containing models
        
    Signature:
        model_names, metric_names, models = 
                build_face_verifier(model_name='VGG-Face', model=None)
        
    Example:
        
    """
    
    if model == None:
        if model_name == 'Ensemble':
            if verbose:
                print("Ensemble learning enabled")
            models = Boosting.loadModel()
            
        else: #model is not ensemble
            model  = DeepFace.build_model(model_name)
            models = {}
            models[model_name] = model

    else: #model != None
        if verbose:
            print("Already built model is passed")
        
        if model_name == 'Ensemble':
            Boosting.validate_model(model)
            models = model.copy()
        else:
            models = {}
            models[model_name] = model


    if model_name == 'Ensemble':
        model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
        metric_names = ['cosine', 'euclidean', 'euclidean_l2']

    elif model_name != 'Ensemble':
        model_names = []; metric_names = []
        model_names.append(model_name)
        metric_names.append(distance_metric)
        
    return model_names, metric_names, models

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

def saved_verifier_exists(verifier_name, save_dir=svd_vrf_dir):
    """
    Checks if a saved verifier exists with name 'verifier_name'. Does that by
    comparing the 'verifier_name' against the name of all files in 'save_dir'.
    
    Inputs:
        1. verifier_name - string with the name of the verifier.
        2. save_dir - string with the full path of the save directory
            ([save_dir=svd_vrf_dir]).
        
    Output:
        1. Boolean value indicating if saved verifier exists or not.
    
    Signature:
        verifier_exists = saved_verifier_exists(verifier_name,
                                                save_dir=svd_vrf_dir)
    """
    # Save directory provided is not a directory
    if not os.path.isdir(save_dir):
        return False
    
    # Otherwise, check if verifier name is in the names of files in the
    # 'save_dir' directory
    is_in = False
    for file_name in os.listdir(save_dir):
        is_in = is_in or verifier_name in file_name
    
    return is_in

# ------------------------------------------------------------------------------

def save_face_verifiers(verifiers, save_dir=svd_vrf_dir, show_prog_bar=True,
                        overwrite=False):
    """
    Saves all face verifier models specified in 'verifiers' using the 'shelve'
    (persistent storage) module. The models are saved in the 'save_dir'
    directory and each model is saved individually. Models that fail to save (or
    produces an error) are skipped. For each model that already exists, this
    function does not overwrite it unless the 'overwrite' flag is set to True.
    
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
    # Checks if the save directory provided is a directory
    if not os.path.isdir(save_dir):
        return True
    
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
        
        if not saved_verifier_exists(name, save_dir=save_dir) or overwrite:
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

def load_face_verifier(verifier_names, save_dir=svd_vrf_dir,
                        show_prog_bar=True, verbose=False):
    """
    Loads all face verifier models specified in 'verifier_names'. Alternatively,
    'all' can be passed as a 'verifier_name' to load all saved models. The
    models are loaded from the 'save_dir' directory.
      
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
    # Checks if the save directory provided is a directory
    if not os.path.isdir(save_dir):
        raise OSError(f'Save directory does not exist ({save_dir})!')
    
    # Ensures that the verifier_names is a list (even a single name is provided)
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]
        
    # If 'all' was provided, use all model names
    if verifier_names[0].lower() == 'all':
        verifier_names = ['VGG-Face', 'OpenFace', 'Facenet', 'Facenet512',
                          'DeepFace', 'DeepID', 'ArcFace', 'Emotion', 'Age',
                          'Gender', 'Race']
    
    # Creates the progress bar
    n_verifiers    = len(verifier_names)
    disable_option = not show_prog_bar
    pbar           = tqdm(range(0, n_verifiers), desc='Loading verifiers',
                            disable = disable_option)
    
    models = {}
    for index, verifier_name in zip(pbar, verifier_names):       
        # Checks if the face verifier model exists
        if saved_verifier_exists(verifier_name, save_dir=save_dir):
            # Loads the model
            file_fp = os.path.join(save_dir, verifier_name)
            with shelve.open(file_fp) as model:
                if verbose:
                    print(f'[load_face_verifier] Loading model',
                          f'{verifier_name}: ', end='')
                try:
                    models[verifier_name] = model[verifier_name]
                    if verbose:
                        print('success!')
                except Exception as excpt:
                    if verbose:
                        print(f'failed! Reason: {excpt}\n',
                               '[load_face_verifier] Attempting to build',
                               ' & save model from scratch: ', sep='', end='')
                    try:
                        cur_model = build_verifier(verifier_name)
                        models[verifier_name] = cur_model
                        save_face_verifiers(cur_model, save_dir=save_dir,
                                            show_prog_bar=False, overwrite=True)
                        if verbose:
                            print('success!')
                    except Exception as excpt:
                        if verbose:
                            print(f'failed! Reason: {excpt}')
                                
        else:
            # Otherwise, tries to build model from scratch & save it
            if verbose:
                print(f'[load_face_verifier] Model {verifier_name} ',
                      'does not exist.', '\nAttempting to build & save model ',
                      'from scratch: ', sep='', end='')
            try:
                cur_model = build_verifier(verifier_name)
                models[verifier_name] = cur_model
                save_face_verifiers(cur_model, save_dir=save_dir,
                                    show_prog_bar=False, overwrite=False)
                if verbose:
                    print('success!')
            except Exception as excpt:
                if verbose:
                    print(f'failed! Reason: {excpt}')
    
    return models

# ------------------------------------------------------------------------------
        
def load_representation_db(file_path, verbose=False):
    """
    Loads a database (at 'file_path') containing representations of face images.
    The database is assumed to be a pickled Python object. The database is
    expected to be a list of Representation object (see help(Representation)
    for more information). If verbose is set to True, the loading processing is
    printed, with any errors being reported to the user.

    Inputs:
        1. file_path - string with file's full path
        2. verbose - flag indicating if the function should print information
            about the loading process and errors ([verbose=False])

    Output:
        1. database object (list of Representation objects)
    
    Signature:
        db = load_representation_db(file_path, verbose=False)
    """
    # Prints message
    if verbose:
        print('Opening database: ', end='')

    # Checks if path provided points to a valid database
    if os.path.isfile(file_path):
        # Try to open pickled database (list of objects)
        try:
            db = pickle.load(open(file_path, 'rb'))
            if verbose:
                print('success!')
        
        except (OSError, IOError) as e:
            if verbose:
                print(f'failed! Reason: {e}')
            db = []

    # If path does not point to a file, open an 'empty' database
    else:
        print(f'failed! Reason: database does not exist.')
        db = []

    return db
                
# ------------------------------------------------------------------------------

def process_face(img_path, target_size=(224, 224), normalization='base',
                    grayscale=False):
    """
    Applies some processing to an image of a face:
        1. Loads the image from an image path, base64 encoding or numpy array
        2. If the 'grayscale' flag is True, converts the image to grayscale
        3. Resizes the image based on the smallest factor (target_size /
            dimension) and zero pads the resized image to match the target size.
        4. If for some reason the image is still not the target size, resizes
            the modified image once again.
        5. Normalizes the image based on the normalization option:
            a. 'base': do nothing
            b. 'raw': restore input in scale of [0, 255]
            c. 'Facenet': 'raw' then subtract image mean and divide by image
                    standard deviation
            d. 'Facenet2018': 'raw' then divide by 127.5 and subtract 1
            e. 'VGGFace': 'raw' then mean subtraction based on VGGFace1 training
                    data
            f. 'VGGFace2': 'raw' then mean subtraction based on VGGFace2
                    training data
            g. 'ArcFace': based on a reference study, 'raw', then pixels are
                    normalized by subtracting 127.5 then dividing by 128.

    Inputs:
        1. img_path - image path, base64 image or numpy array
        2. target_size - tuple containing desired X and Y dimensions
            ([target_size=(224, 224)])
        3. normalization - defines a type of normalization
            ([normalization='base'])
        4. grayscale - flag indicating if image should be converted to grayscale
            ([grayscale=False])

    Output:
        1. processed image as a numpy array

    Signature:
        face_image = process_face(img_path, target_size=(224, 224),
                                  normalization='base', grayscale=False)
    """
    # Loads the face image. Image might be path, base64 or numpy array. Convert
    # it to numpy whatever it is.
    face = functions.load_image(img_path)
    
    # Ensures that both dimensions are >0, otherwise raises error
    if face.shape[0] == 0 or face.shape[1] == 0:
        raise ValueError(f'Detected face shape is {face.shape}.')

    # Converts to grayscale if face is 
    if grayscale:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
    # Resizes face
    if face.shape[0] > 0 and face.shape[1] > 0:
        factor_0 = target_size[0] / face.shape[0]
        factor_1 = target_size[1] / face.shape[1]
        factor   = min(factor_0, factor_1)

    dsize  = (int(face.shape[1] * factor), int(face.shape[0] * factor))
    face   = cv2.resize(face, dsize)

    # Then pad the other side to the target size by adding black pixels
    diff_0 = target_size[0] - face.shape[0]
    diff_1 = target_size[1] - face.shape[1]
                
    if not grayscale:
        # Put the base image in the middle of the padded image
        face = np.pad(face, ((diff_0 // 2, diff_0 - diff_0 // 2),
                             (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                             'constant')
    else:
        face = np.pad(face, ((diff_0 // 2, diff_0 - diff_0 // 2),
                             (diff_1 // 2, diff_1 - diff_1 // 2)),
                             'constant')

    # Double check if target image is not still the same size with target.
    if face.shape[0:2] != target_size:
        face = cv2.resize(face, target_size)

    # Normalizing the image pixels
    if normalization == 'base':
        face  = image.img_to_array(face) #what this line doing? must?
        face  = np.expand_dims(face, axis = 0)
        face /= 255 # normalize input in [0, 1]
    else:
        face = functions.normalize_input(face, normalization=normalization)
    
    return face

# ______________________________________________________________________________
#                   REPRESENTATION DATABASE RELATED FUNCTIONS
# ------------------------------------------------------------------------------

def find_image_in_db(img_path, db, shortcut=None):
    """
    Finds the corresponding Representation in the database 'db'. Uses the
    image's name obtained from its full path ('img_path') to match the
    Representation to the image. An optional dictionary ('shortcut') can be
    provided where each letter corresponds to an index to speed up the search.
    Consider a sorted database where the first image with a name starting with
    a 'g' is at position 156. Then a shortcut dictionary with the key/value pair
    'g':155 will ensure this function starts at the position 156 and wont waste
    time previous database entries.

    Inputs:
        1. img_path - image path, base64 image or numpy array
        2. db - database object (list of Representation objects)
        3. shortcut - optional dictionary with letter/index key/value pairs
            ([shortcut=None])

    Output:
        1. tuple containing the matching Representation object and the index
            corresponding to that object in the database (i.e. output
            representation == database[output index]). If no match is found,
            both representation and index are returned as empty lists. If
            multiple entries are found, multiple Representations and indexs are
            returned as lists.

    Signature:
        rep_objs, rep_idxs = find_image_in_db(img_path, db, shortcut=None)
    """
    rep_objs = [] # empty representation object
    rep_idxs = [] # no matching index

    # Database is empty
    if len(db) == 0:
        pass # do nothing

    # 'shortcut' dictionary is provided
    elif not shortcut == None:
        # Get image name
        img_name = img_path.split('/')[-1].lower()
        idx      = shortcut[img_name[0]]

        # 
        for i, rep in enumerate(db[idx::]):
            if img_name.lower() == rep.image_name:
                rep_objs.append(rep)
                rep_idxs.append(i + idx)

    # 'shortcut' dictionary is not provided
    else:
        # Get image name
        img_name = img_path.split('/')[-1].lower()

        # 
        for i, rep in enumerate(db):
            if img_name.lower() == rep.image_name:
                rep_objs.append(rep)
                rep_idxs.append(i)

    return (rep_objs, rep_idxs)

# ------------------------------------------------------------------------------

def create_new_representation(img_path, region, embeddings, tag='', uid='',
                                ignore_taglist=['--', '---']):
    """
    Creates a new representation object. For more information see
    help(Representation).

    Inputs:
        1. img_path - string containing image full path
        2. region - list of integers specifying face region on the original
            image
        3. embeddings - dictionary with face verifier name (key) and embedding
            (1-D numpy array) (item). Can have multiple verifier, embedding
            pairs (key, value pairs).
        4. tags - string containing a name tag for this representation
            ([tag=''])
        5. uid - string containing unique object identifier. If left empty ('')
            a unique object identifier is created using uuid4 from uuid library
            ([uid=''])
        6. ignore_taglist - list of strings that are treated as equivalent to ''
            (i.e. no tag) ([ignore_taglist=['--', '---']])
    
    Output:
        1. Representation object

    Signature:
        new_rep = create_new_representation(img_path, region, embeddings,
                                tag='', uid='', ignore_taglist=['--', '---'])
    """

    # If Unique IDentifier (UID) is not provided, generate one
    if len(uid) == 0 or uid == '':
        uid = uuid4()

    # If tag is in the ignore taglist, then it is considered as a "ignore" tag
    if tag in ignore_taglist:
        tag = ''

    # Returns the new representation
    return Representation(uid, image_name=img_path.split('/')[-1], 
                          image_fp=img_path, name_tag=tag, region=region,
                          embeddings=embeddings)

# ------------------------------------------------------------------------------

def update_representation(rep, embeddings):
    """
    Updates an existing representation object. For more information on the
    Representation object see help(Representation).

    Inputs:
        1. rep - Representation object
        2. embeddings - dictionary with face verifier name (key) and embedding
            (1-D numpy array) (item). Can have multiple verifier, embedding
            pairs (key, value pairs).
    
    Output:
        1. Representation object

    Signature:
        updated_rep = update_representation(rep, embeddings)
    """

    # Loops through each model name and embedding pair, adding or updating the 
    # corresponding value in the embeddings dictionary of the representation 
    # provided
    for model_name, embedding in embeddings.items():
        rep.embeddings[model_name] = embedding

    return rep

# ------------------------------------------------------------------------------

def get_embeddings_as_array(db, verifier_name):
    """
    Gets all of the embeddings for a given 'verifier_name' from the database
    'db' and returns it as a N x M numpy array where N is the number of
    embeddings and M is the number of elements of each embeddings.

    Inputs:
        1. db - database object (list of Representation objects)
        2. verifier_name - name of verifier

    Output:
        1. N x M numpy array where each row corresponds to a face image's
            embedding

    Signature:
        embeddings = get_embeddings_as_array(db, verifier_name)
    """
    embeddings = [] # initializes empty list
    for rep in db:
        embeddings.append(rep.embeddings[verifier_name])

    return np.array(embeddings)

# ------------------------------------------------------------------------------

def create_reps_from_dir(img_dir, verifier_models, detector_name='opencv',
                    align=True, verifier_names='VGG-Face', show_prog_bar=True,
                    normalization='base', tags=[], uids=[], verbose=False):
    """
    Creates a representations from images in a directory 'img_dir'. The
    representations are returned in a list, and the list of representations. If
    tags and/or unique identifiers (uids) are provided, make sure that they 
    correspond to the sorted (ascending) image names contained in 'img_dir'.

    Inputs:
        1. img_dir - string with the full path to the directory containing the
            images.

        2. verifier_models - dictionary containing model name (key) and model
            object (value).

        3. detector_name - string with the face detector name ([opencv], ssd,
            dlib, mtcnn, retinaface).
        
        4. align - boolean flag to indicate if face images should be aligned.
            Improves face recognition performance at the cost of some speed
            ([True], False).

        5. verifier_names - string with the face verifier name ([VGG-Face],
            OpenFace, Facenet, Facenet512, DeepFace, DeepID, ArcFace).

        6. show_prog_bar - boolean that toggle the progress bar on or off
            ([True], False).

        7. normalization - normalizes the face image and may increase face
            recognition performance depending on the normalization type and the
            face verifier model ([base], raw, Facenet, Facenet2018, VGGFace,
            VGGFace2, ArcFace).

        8. tags - list of strings where each string corresponds to a tag for the
            i-th image, i.e. tags[0] is the tag for the first image in the
            sorted list of image name obtain from 'img_dir' directory. If an
            empty list is provided, this is skipped during the representation
            creation process ([tags=[]]).

        9. uids - list of strings where each string corresponds to a unique
            identifier (UID) for the i-th image, i.e. uids[0] is the UID for the
            first image in the sorted list of image name obtain from 'img_dir'
            directory. If an empty list is provided, a UID is created for each
            image during the representation creation process ([uids=[]]).
            
        10. verbose - boolean to toggle function warnings and other messages
            ([True], False).

        Note: the 'tags' and 'uids' lists (inputs 8 and 9) must have the same
        number of elements (length) and must match the number of images in
        'img_dir'. If not, these inputs will be treated as empty lists (i.e.
        ignored).

    Outputs:
        1. list of Representation objects. For more information about the
            Representation class attributes and methods, use
            help(Representation)

    Signature:
        rep_db = create_reps_from_dir(img_dir, verifier_models, 
                        detector_name='opencv', align=True,
                        verifier_names='VGG-Face', show_prog_bar=True,
                        normalization='base', tags=[], uids=[], verbose=False)
    """
    # Initializes skip flags and database (list of Representation objects)
    skip_tag = False
    skip_uid = False
    rep_db   = []
    
    # Assuming img_dir is a directory containing images
    img_paths = get_image_paths(img_dir)
    img_paths.sort()

    # No images found, return empty database
    if len(img_paths) == 0:
        return []

    # If tags list does not have the same number of elements as the images (i.e.
    # 1 tag per image), ignore it
    if len(tags) != len(img_paths):
        if verbose:
            print('[create_reps_from_dir] Number of tags and image paths',
                  'must match. Ignoring tags list.')
        skip_tag = True

    # If uids list does not have the same number of elements as the images (i.e.
    # 1 UID per image), ignore it
    if len(uids) != len(img_paths):
        if verbose:
            print('[create_reps_from_dir] Number of UIDs and image paths',
                  'must match. Ignoring uids list.')
        skip_uid = True

    # Creates the progress bar
    n_imgs  = len(img_paths)
    disable = not show_prog_bar
    pbar    = tqdm(range(0, n_imgs), desc='Creating representations',
                    disable=disable)

    # Loops through each image in the 'img_dir' directory
    for pb_idx, i, img_path in zip(pbar, range(0, n_imgs), img_paths):
        # Calculate the face image embedding
        region, embeddings = calc_embedding(img_path, verifier_models,
                                            align=align,
                                            detector_name=detector_name, 
                                            verifier_names=verifier_names,
                                            normalization=normalization)

        # Determines if tag was provided and should be used when creating this
        # representation
        if skip_tag:
            tag = ''
        else:
            tag = tags[i]

        # Determines if UID was provided and should be used when creating this
        # representation
        if skip_uid:
            uid = ''
        else:
            uid = uids[i]

        # Create a new representation and adds it to the database
        rep_db.append(create_new_representation(img_path, region, embeddings,
                                                tag=tag, uid=uid))

    # Return representation database
    return rep_db

# ______________________________________________________________________________
#                           VERIFY RELATED FUNCTIONS
# ------------------------------------------------------------------------------

def create_faiss_index(embeddings, metric='cosine'):
    """
    Create faiss indexes from 'embeddings'. Embeddings are expected to be a
    N x M numpy array with N embeddings each with M elements.

    Inputs:
        1. embeddings - N x M numpy array
        2. metric - string specifying the distance/similarity metric ([cosine],
            euclidean, euclidean_l2)

    Output:
        1. index object with embeddings added

    Signature:
        index = create_faiss_index(embeddings, metric='cosine')
    """
    # Create the appropriate index based on the distance metric provided. If the
    # distance metric provided is not valid (not cosine, euclidean or
    # euclidean_l2), an error is raised.
    if metric == 'cosine':
        index = faiss.IndexFlatIP(embeddings.shape[1])
    elif metric in ['euclidean', 'euclidean_l2']:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    else:
        raise ValueError(f'{metric} is an invalid metric!',
                          'Expected cosine, euclidean or euclidean_l2.',
                          sep='\n')
    
    # Applies L2 normalization in the cases of cosine or euclidean_l2 metrics
    if metric in ['cosine', 'euclidean_l2']:
        embeddings = dst.l2_normalize(embeddings)

    # Adds the embeddings to the index object
    index.add(embeddings)

    return index

# ------------------------------------------------------------------------------

def load_faiss_indexes(idx_path, embeddings=None, metric='cosine',
                        verbose=False):
    """
    If 'embeddings' == None, attempts to load a faiss index object saved at
    'idx_path'. If 'embeddings' is not None and the path provided does not point
    to a valid file, creates a new index object using 'embeddings'. Finally, if
    the file path provided is valid and 'embeddings' is not None, attempts to
    load the index object and add the embeddings provided. Once again, on
    failure, a new index object is created.

    Input:
        1. idx_path - full path to file containing index object
        2. embeddings - N x M numpy array or None ([embeddings=None])
        3. metric - string specifying the distance/similarity metric ([cosine],
            euclidean, euclidean_l2)
        4. verbose - boolean flag controlling if the function prints creation
            and error messages ([verbose=True])

    Output:
        1. faiss index object
    
    Signature:
        index = load_faiss_indexes(idx_path, embeddings=None, metric='cosine',
                                    verbose=False)
    """
    # If embeddings is None, attempts to load the index object. If the loading
    # process fails, None is returned.
    if embeddings is None:
        # Attempts to load index path
        if verbose:
            print('Loading index object: ', sep='', end='')

        try:
            # Loading index object
            index = faiss.read_index(idx_path)
            if verbose:
                print(f'success!')

        except Exception as excpt:
            # Loading failed. Print exception if verbose == True and create new
            # index object
            if verbose:
                print(f'failed! Reason: {excpt}')
            index = None

    # Index path is not file (or does not exist) so attempt to create a new
    # index object and add the embeddings provided (remember, they are not None)
    elif not os.path.isfile(idx_path):
        if verbose:
            print('Creating index object: ', sep='', end='')

        try:
            index = create_faiss_index(embeddings, metric=metric)
            if verbose:
                print(f'success!')

        except Exception as excpt:
            if verbose:
                print(f'failed! Reason: {excpt}')
            index = None

    # Otherwise, attempts to load the index object at the file path provided
    else:
        # Attempts to load index path
        if verbose:
            print('Loading index object: ', sep='', end='')

        try:
            # Loading index object
            index = faiss.read_index(idx_path)
            index.add(embeddings)
            if verbose:
                print(f'success!')

        except Exception as excpt:
            # Loading failed. Print exception if verbose == True and create new
            # index object
            if verbose:
                print(f'failed! Reason: {excpt}')
                print('Creating new indexes.')
            
            index = create_faiss_index(embeddings, metric=metric)

    return index

# ------------------------------------------------------------------------------

def verify_with_faiss(target_embd, index, db, top_x=3, model_name='VGG-Face',
                      metric='cosine', threshold=-1):
    """
    Uses the Faiss library to speed up the verification process. The 'index'
    object is a faiss-specific object with the embeddings of the whole database
    'db'. Using faiss fast search, the 'top_x' nearest neighbour embeddings are
    obtained and a subset of the original database is created.

    The similarity (and matches) are then calculated using the subset database,
    which is only 'top_x' large. Note that the speed gains are more noticeable
    for larger databases and small values of 'top_x'. Also note that if there
    are more than 'top_x' matches, then only 'top_x' will be returned.

    Inputs:
        1. target_embd - 1-D numpy array containing the embedding.
        2. index - faiss-specific index object. Use 'load_faiss_indexes'
                function to create it.
        3. db - database (list of Representation objects)
        4. top_x - natural number indicating the number of nearest neighbours to
                filter the database ([top_x=3])
        5. model_name - string specifying the model name (['VGG-Face']).
        6. metric - string specifying the distance metric to be used
                (['cosine'], 'euclidean', 'l2_euclidean').
        7. threshold - if a negative float is provided, then the threshold is
                calculated automatically based on the model name provided.
                Otherwise, the threshold provided is used ([threshold=-1]).
    
    Output:
        1. dictionary containing the indexes of matches (key: idxs), the
            threshold value used (key: threshold) and the distances calculated
            using the specified metric (key: distances). Note that if no match
            is found, then the 'indexes' will have a length of zero (i.e. will
            be empty).

    Signature:
        similarity_obj = verify_with_faiss(target_embd, index, db, top_x=3,
                                    model_name='VGG-Face', metric='cosine',
                                    threshold=-1)
    """
    # Obtains the 'top_x' nearest neighbours
    distances, neighbors = index.search(target_embd, top_x)

    # Obtains the embeddings for the 'filtered' (top_x) representations
    filtered_reps  = [db[i] for i in np.array(neighbors[0], dtype=int)]
    embds          = get_embeddings_as_array(filtered_reps, model_name)

    # Calculates the matches to the target embedding (from the top_x filtered
    # set of representations)
    similarity_obj = calc_similarity(target_embd, embds, metric=metric,
                                     model_name=model_name, threshold=threshold)

    return similarity_obj

# ______________________________________________________________________________
#                   SIMILARITY & DISTANCE RELATED FUNCTIONS
# ------------------------------------------------------------------------------

def calc_cosine_similarity(A, B):
    """
    Calculates the cosine similarity metric between matrices A and B. If A is a
    vector, it is converted into a matrix so that the cosine metric can be
    calculated normally.

    Inputs:
        1. A - N x M matrix with N embeddings with M elements
        2. A - I x M matrix with I embeddings with M elements

    Outputs:
        1. matrix of cosine similarity metric between A and B

    Signature:
        csm = calc_cosine_similarity(A, B)
    """
    if A.ndim == 1:
        A = A[np.newaxis, :]

    num = np.dot(A, B.T)
    p1  = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis]
    p2  = np.sqrt(np.sum(B**2, axis=1))[np.newaxis, :]
    return 1 - (num / (p1 * p2))

# ------------------------------------------------------------------------------

def calc_euclidean_similarity(A, B, l2_normalize=False):
    """
    Calculates the Euclidean similarity metric between matrices A and B. If A is
    a vector, it is converted into a matrix by repeating (and stacking) it
    horizontally until it has the correct dimensions. If 'l2_normalize' is set
    to True, then the function applies L2 normalization to the inputs before
    calculating the Euclidean similarity (distance).

    Inputs:
        1. A - N x M matrix with N embeddings with M elements
        2. A - I x M matrix with I embeddings with M elements
        3. l2_normalize - boolean to indicate if the inputs should be L2
            normalized before calculating the similarity ([l2_normalize=True])

    Outputs:
        1. matrix of Euclidean similarity metric between A and B

    Signature:
        edm = calc_euclidean_similarity(A, B, l2_normalize=False)
    """
    # Applies l2 normalization to the inputs if necessary
    if l2_normalize:
        if A.ndim == 1:
            A = A / np.sqrt(np.sum(np.multiply(A, A)))
        else:
            A = np.transpose(A.T / np.linalg.norm(A, axis=1))

        if B.ndim == 1:
            B = B / np.sqrt(np.sum(np.multiply(B, B)))
        else:
            B = np.transpose(B.T / np.linalg.norm(B, axis=1))

    # 'Repeats vertically' vector A until it is a matrix with appropriate
    # dimensions 
    if A.ndim == 1:
        A = np.tile(A, (B.shape[0], 1))

    # Calcultes and returns the Euclidean distance
    return np.sqrt(np.sum((A - B) * (A - B), axis=1))

# ------------------------------------------------------------------------------

def calc_similarity(tgt_embd, embds, metric='cosine', model_name='VGG-Face',
                    threshold=-1):
    """
    Calculates the similarity (distance) between both embeddings ('tgt_embd'
    and 'embds') using the 'metric' distance metric. If the 'threshold' < 0 then
    it is automatically determined based on the 'model_name' provided. If a 
    custom threshold is specified, then the 'model_name' input is unused.

    Note that 'embds' can be a N x M matrix (N embeddings each with M elements)
    and 'tgt_embd' can only be a 1 x M embedding.

    Inputs:
        1. tgt_embd - 1-D numpy array containing the embedding.
        2. embds - 1-D or 2-D numpy array containing the embedding(s).
        3. metric - string specifying the distance metric to be used
            (['cosine'], 'euclidean', 'l2_euclidean').
        4. model_name - string specifying the model name (['VGG-Face']).
        5. threshold - if a negative float is provided, then the threshold is
            calculated automatically based on the model name provided.
            Otherwise, the threshold provided is used ([threshold=-1]).

    Output:
        1. dictionary containing the indexes of matches (key: idxs), the
            threshold value used (key: threshold) and the distances (using the
            specified metric) of the matches (key: distances). Note that if no
            match is found, then the 'indexes' will have a length of zero (i.e.
            will be empty).
    
    Signature:
        similarity_obj = calc_similarity(tgt_embd, embds, metric='cosine',
                                         model_name='VGG-Face', threshold=-1)
    """
    # Calculates the distance based on the metric provided, otherwise raises a
    # value error
    if metric == 'cosine':
        distances = calc_cosine_similarity(tgt_embd, embds)
    elif metric == 'euclidean':
        distances = calc_euclidean_similarity(tgt_embd, embds)
    elif metric == 'euclidean_l2':
        distances = calc_euclidean_similarity(tgt_embd, embds,
                                              l2_normalize=True)
    else:
        raise ValueError(f'Invalid metric passed: {metric}')

    # If threshold is negative, determine threshold automatically based on the
    # model name and distance metric
    if threshold < 0:
        threshold = dst.findThreshold(model_name, metric)
    
    # Processes distances
    distances = distances.flatten() # flattens into 1-D array

    # Makes a decision
    decision  = (distances <= threshold)
    idxs      = np.where(decision)[0]

    # Determines the corresponding distances of each match (decision==True)
    dists = []
    j     = 0
    for i in range(len(distances)):
        if i in idxs:
            dists.append(distances[i])
            j += 1
        
        if j == len(idxs):
            break

    # Sort the indexes by smallest distance
    srt_dists = []
    srt_idxs  = []
    for dist, idx in sorted(zip(dists, idxs)):
        srt_idxs.append(idx)
        srt_dists.append(dist)

    return {'idxs': np.array(srt_idxs),
            'threshold': threshold,
            'distances': np.array(srt_dists)}

# ------------------------------------------------------------------------------

def calc_embedding(img_path, verifier_models, detector_name='opencv',
                    align=True, verifier_names='VGG-Face',
                    normalization='base'):
    """
    Calculates the embedding (1-D numpy array) of a face image. Each embedding
    is associated with a face verifier model. Multiple verifiers can be passed
    (as model names) to this function so that multiple embeddings are calculated
    in a single function call.

    Inputs:
        1. img_dir - string with the full path to the directory containing the
            images.
        2. verifier_models - dictionary containing pre-built models (key: model
            name, value: built model object)
        3. detector_name - string with the chosen face detector name ([opencv],
            ssd, dlib, mtcnn, retinaface)
        4. align - boolean indicating if alignment of face images should be
            performed (may improve recognition performance around 1-2%)
            ([align=True])
        5. verifier_names - string or list of strings with face verifier name(s)
            ([VGG-Face], OpenFace, Facenet, Facenet512, DeepFace, DeepID, Dlib,
            ArcFace)
        6. normalization - string indicating the type of image normalization to
            be performed ([base], raw, Facenet, Facenet2018, VGGFace, VGGFace2,
            ArcFace)
        
    Outputs:
        1. region - list of lists of 4 integers specifying the faces' region on
            the original image. The 4 integers correspond to the top-left
            corner's and bottom-right corner's x & y coordinates respectively
        2. embeddings - dictionary containing the embedding (value) for each
            face verifier model provided in 'verifier_names' (key)

        The outputs are returned as a tuple.

    Signature:
        region, embeddings = calc_embedding(img_path, verifier_models,
                                            detector_name='opencv', align=True,
                                            verifier_names='VGG-Face',
                                            normalization='base')
    """
    # Converts verifier names into a list if it is a single entry
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]

    # Tries to detect faces & align:
    try:
        output = detect_faces(img_path, detector_backend=detector_name,
                                align=align, return_type='both',
                                face_detector=None)
    except:
        print('[calc_embedding] Error: face detection failed!')
        return ([], {})

    # TODO: MAKE THE FUNCTION ACCEPT MULTIPLE FACES IN ONE IMAGE
    # Since we assume there is only 1 face (and region):
    face   = output['faces'][0]
    region = output['regions'][0]

    # For each verifier model provided
    embeddings={}
    for verifier_name in verifier_names:
        try:
            # Gets the current verifier model
            model = verifier_models[verifier_name]

            # Determine target size
            input_x, input_y = functions.find_input_shape(model)

            # Process face
            processed_face = process_face(face, target_size=(input_x, input_y),
                                   normalization=normalization, grayscale=False)

            # Calculate embeddings
            embeddings[verifier_name] = model.predict(processed_face)[0]
        except Exception as excpt:
            print(f'[calc_embedding] Error when calculting {verifier_name}. ',
                  f'Reason: {excpt}', sep='')

    return (region, embeddings)

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def get_matches_from_similarity(similarity_obj, db, verifier_name):

    # Initializes all required lists
    mtch_uids  = [] # unique ids
    mtch_tags  = [] # name tags
    mtch_names = [] # image names
    mtch_fps   = [] # image full paths (fp)
    mtch_rgns  = [] # regions
    mtch_embds = [] # embeddings

    for i in similarity_obj['idxs']:
        rep = db[i]
        mtch_uids.append(rep.unique_id)
        mtch_tags.append(rep.name_tag)
        mtch_names.append(rep.image_name)
        mtch_fps.append(rep.image_fp)
        mtch_rgns.append(rep.region)
        mtch_embds.append(list(rep.embeddings[verifier_name]))

    return {'unique_ids':mtch_uids  , 'name_tags':mtch_tags,
            'image_names':mtch_names, 'image_fps':mtch_fps ,
            'regions':mtch_rgns     , 'embeddings':mtch_embds,
            'distances':list(similarity_obj['distances']),
            'threshold':similarity_obj['threshold']}
