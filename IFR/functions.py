# ==============================================================================
#                                   FUNCTIONS
# ==============================================================================

# Module / package imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import time
import cv2

import numpy                 as np

from tqdm                    import tqdm
from deepface                import DeepFace
from deepface.basemodels     import Boosting
from deepface.detectors      import FaceDetector
from keras.preprocessing     import image

from deepface.commons                import functions, distance as dst
from deepface.DeepFace               import build_model        as build_verifier
from deepface.detectors.FaceDetector import build_model        as build_detector

# ______________________________________________________________________________
#                           OTHER / OLDISH? FUNCTIONS
# ------------------------------------------------------------------------------

def detect_faces(img_path, detector_backend='opencv', align=True,
                 return_type='both', face_detector=None):
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

                    # Issue #175: causes trobule for euclideans in api calls if
                    # this is not set
                    distance = np.float64(distance)
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

# ------------------------------------------------------------------------------

def get_property_from_database(db, param, do_sort=False, suppress_error=True):
    """
    Gets a specific property 'param' from each representation the in the
    database 'db'. If the flag 'do_sort' is True, then the output is sorted. If
    the 'suppress_error' flag is set to True, then if the chosen property
    'param' does not exist an empty list is returned. Otherwise, an Assertion
    error is raised.

    Inputs:
        1. db - list of representation objects.
        2. param - string with the property / parameter name (unique_id,
            name_tag, image_name, image_fp, region or embeddings).
        3. do_sort - boolean flag controlling if sorting of all properties
            should be performed ([False], True).
        4. suppress_error - boolean flag controlling if errors should be
            suppressed, i.e. if the chosen property does not exist, an empty
            list should be returned instead of an exception (False, [True]).

    Output:
        1. list containing the chosen property from each representation. The
            list is sorted if 'do_sort' is set to True. The list will be empty
            if the representation database has a length of zero or if
            'suppress_error' is True and a non-existant property 'param' is
            chosen.

    Signature:
        propty = get_property_from_database(db, param, do_sort=False,
                                            suppress_error=True)
    """
    # Gets the names in the database depending on the database's size
    if len(db) == 0:   # no representations
        propty = []

    elif len(db) == 1: # single representation
        if   param == 'unique_id':
            propty = [db[0].unique_id]
        elif param == 'image_name':
            propty = [db[0].image_name]
        elif param == 'image_fp':
            propty = [db[0].image_fp]
        elif param == 'group_no':
            propty = [db[0].group_no]
        elif param == 'name_tag':
            propty = [db[0].name_tag]
        elif param == 'region':
            propty = [db[0].region]
        elif param == 'embeddings':
            propty = [db[0].embeddings]
        else:
            if suppress_error:
                propty = []
            else:
                raise AttributeError('Representation does not have '
                                   + 'the chosen property.')

    elif len(db) > 1:  # many representations
        # Loops through each representation in the database and gets the
        # specified property / parameter
        if   param == 'unique_id':
            propty = []
            for rep in db:
                propty.append(rep.unique_id)

        elif param == 'image_name':
            propty = []
            for rep in db:
                propty.append(rep.image_name)
            
        elif param == 'image_fp':
            propty = []
            for rep in db:
                propty.append(rep.image_fp)

        elif param == 'group_no':
            propty = []
            for rep in db:
                propty.append(str(rep.group_no))
            propty = np.unique(propty)  # only keep unique groups
            propty = sorted(propty, key=lambda x: int(x))   # and sort them
            
        elif param == 'name_tag':
            propty = []
            for rep in db:
                propty.append(rep.name_tag)
            propty = np.unique(propty) # only keep unique name tags
            
        elif param == 'region':
            propty = []
            for rep in db:
                propty.append(rep.region)
            
        elif param == 'embeddings':
            propty = []
            for rep in db:
                propty.append(rep.embeddings)
            
        else:
            if suppress_error:
                propty = []
            else:
                raise AttributeError('Representation does not have '
                                   + 'the chosen property.')

        # Sorts the property if do_sort flag is set to True
        if do_sort:
            propty.sort()
    
    else: # this should never happen (negative size for a database? preposterous!)
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    return list(propty)

# ------------------------------------------------------------------------------

def string_is_valid_uuid4(uuid_string):
    """
    Checks if the string provided 'uuid_string' is a valid uuid4 or not.

    Input:
        1. uuid_string - string representation of uuid including dashes ('-')
             [string].
    
    Output:
        1. boolean indicating if the string is a valid uuid4 or not.

    Signature:
        is_valid = string_is_valid_uuid4(uuid_string)
    """
    uuid4hex = re.compile('^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z', re.I)
    match = uuid4hex.match(uuid_string)
    return bool(match)

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

    # Checks if the face detector was able to find any face
    if len(output['faces']) == 0:
        print('[calc_embedding] Error: face detection failed!')
        return ([], {})
    else:
        pass # do nothing

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

def get_matches_from_similarity(similarity_obj, db, verifier_name, verbose=True):
    """
    Gets all matches from the database 'db' based on the current similairty
    object 'similairty_obj' and the face verifier 'verifier_name'. This object
    is obtained from 'calc_similarity()' function (see help(calc_similarity) for
    more information).

    Inputs:
        1. similarity_obj - dictionary containing the indexes of matches (key:
            idxs), the threshold value used (key: threshold) and the distances
            of the matches (key: distances).
        2. db - list of Representations
        3. verifier_name - string specifying the face verifier used (VGG-Face,
            Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace)

    Output:
        1. dictionary containing:
            - unique_ids : list of unique ids
            - name_tags  : list of name tags
            - image_names: list of image names
            - image_fps  : list of image full paths
            - regions    : list of regions, where each region is a list
                            containing the (X, Y) coordinates of the top-left
                            and bottom-right corners of the rectangle that
                            contains the detected face
            - embeddings : list of face verifier names (strings) for which this
                            Representation has embeddings for
            - distances  : list of distances / similarity
            - threshold  : float indicating the cutoff value for the decision

            The length of each list is equal to the number of matches, so that
            each index correspond to a single match (i.e. index 0 corresponds to
            match 1, index 1 to match 2, etc...)

    Signature:
        match_object = get_matches_from_similarity(similarity_obj, db,
                                                   verifier_name)
    """

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
        if verbose:
            mtch_embds.append(list(rep.embeddings[verifier_name]))
        else:
            mtch_embds.append([])

    return {'unique_ids':mtch_uids  , 'name_tags':mtch_tags,
            'image_names':mtch_names, 'image_fps':mtch_fps ,
            'regions':mtch_rgns     , 'embeddings':mtch_embds,
            'distances':list(similarity_obj['distances']),
            'threshold':similarity_obj['threshold']}

# ------------------------------------------------------------------------------
