# ==============================================================================
#                          CUSTOM FUNCTIONS
# ==============================================================================

# Module / package imports
import os
import cv2
import time
import numpy              as np
import tensorflow         as tf

from tqdm                    import tqdm
from deepface                import DeepFace
from deepface.basemodels     import Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons        import functions, realtime, distance as dst
from deepface.detectors      import FaceDetector

tf_version = int(tf.__version__.split(".")[0])

if tf_version == 2:
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    from tensorflow.keras.preprocessing import image
else:
    from keras.preprocessing import image

# ==============================================================================

def get_image_paths(root_path, file_types=('.jpg', '.png')):
    """
    Gets the full paths of all 'file_types' files in the 'root_path' directory
    and its subdirectories.

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

# ==============================================================================

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
    detections    = FaceDetector.detect_faces(face_detector, detector_backend,
                                              img, align)

    # Prints a warning and returns an empty dictionary an error if no faces were
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

# ==============================================================================

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

# =================================================================

def process_single_face(img_path, target_size = (224, 224), grayscale=False):
    # Loads the face image. Image might be path, base64 or numpy array. Convert
    # it to numpy whatever it is.
    face = functions.load_image(img_path)
    
    # Ensures that both dimensions are >0, otherwise raises error
    if face.shape[0] == 0 or face.shape[1] == 0:
        raise ValueError(f'Detected face shape is {face.shape}.')

    # Converts to grayscale
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
    face  = image.img_to_array(face) #what this line doing? must?
    face  = np.expand_dims(face, axis = 0)
    face /= 255 # normalize input in [0, 1]
    
    return face

# =================================================================

def calc_representations(img_paths, model_name='VGG-Face', model=None,
                         grayscale=False, align=True, normalization='base',
                         unique_id_start=0):
    
    # Currently, this function does not support multiple models or Ensemble
    if model_name == 'Ensemble':
        print('[WARNING] Ensemble and multiple models currently unsupported! Returning empty representation.')
        return []
    
    # Builds the face verifier model - uses input variables to store result from this
    # function, maybe using a different name is clearer?
    model_names, junk, models = build_face_verifier(model_name=model_name, model=model)
    
    
    representations = []
    unique_id       = unique_id_start
    
    # Loops through each image
    for img_path in img_paths:
        embeddings = {} # initialize embeddings dictionary
        img_name   = img_path.split('/')[-1] # gets image name
        
        # For the current image, loops through each model and create its representation
        for name, model in zip(model_names, models.values()):
            # Find current model's input shape and process the face image
            input_shape_x, input_shape_y = functions.find_input_shape(model)
            face = process_single_face(img_path, target_size = (input_shape_x, input_shape_y),
                                       grayscale=grayscale)
            
            # Normalizes input and finds the embedding
            face = functions.normalize_input(img=face, normalization=normalization)
            embeddings[name] = model.predict(face)[0].tolist()
            
        # Create representation and increment unique id
        representations.append(Representation(unique_id, img_name,
                                              image_fp=img_path,
                                              embeddings=embeddings))
        unique_id += 1
    
    return representations, unique_id

# =================================================================

def verify_from_reps(target_reps, gallery_reps, model_names, distance_metrics=['cosine'], threshold=-1, prog_bar=True):

    """
    
    """
    # Starts function timer
    tic = time.time()
    
    #------------------------------
    
    if not isinstance(model_names, list):
        model_names = [model_names]

    #------------------------------

    disable_option = not prog_bar
    n_tgts         = len(target_reps)
    n_gall         = len(gallery_reps)
    pbar           = tqdm(range(0, n_tgts * n_gall),
                          desc='Verification', disable=disable_option)
    bulk_process   = False if n_tgts * n_gall != 1 else True
    resp_objects   = []

    for index in pbar:
        ensemble_features = []

        for model_name in model_names:
            # Get embeddings and names from the representations
            target_rep   = target_reps[index // n_gall].embeddings[model_name]
            target_name  = target_reps[index // n_gall].image_name
            gallery_rep  = gallery_reps[np.mod(index, n_gall)].embeddings[model_name]
            gallery_name = gallery_reps[np.mod(index, n_gall)].image_name
            
            #----------------------
            #find distances between embeddings

            for metric in distance_metrics:
                if metric == 'cosine':
                    distance = dst.findCosineDistance(target_rep, gallery_rep)
                elif metric == 'euclidean':
                    distance = dst.findEuclideanDistance(target_rep, gallery_rep)
                elif metric == 'euclidean_l2':
                    distance = dst.findEuclideanDistance(dst.l2_normalize(target_rep), dst.l2_normalize(gallery_rep))
                else:
                    raise ValueError("Invalid distance_metric passed - ", metric)

                distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
                #----------------------
                
                # Makes a decision  EDIT THE RESPONSE OBJECT
                if model_name != 'Ensemble':
                    
                    # Finds the threshold if the one provided is not a number (int or float)
                    # or if it is a negative number
                    if type(threshold) == int or float:
                        if threshold < 0:
                            threshold = dst.findThreshold(model_name, metric)
                    else:
                        threshold = dst.findThreshold(model_name, metric)

                    # Makes the decision
                    if distance <= threshold:
                        identified = True
                    else:
                        identified = False

                    resp_obj = {'target': target_name, 'ref': gallery_name,
                                'verified': identified, 'distance': distance,
                                'threshold': threshold, 'model': model_name,
                                'similarity_metric': metric}

                    resp_objects.append(resp_obj)

                else: #Ensemble

                    #this returns same with OpenFace - euclidean_l2
                    if model_name == 'OpenFace' and metric == 'euclidean':
                        continue
                    else:
                        ensemble_features.append(distance)

        #----------------------

        if model_name == 'Ensemble':
            boosted_tree = Boosting.build_gbm()
            prediction   = boosted_tree.predict(np.expand_dims(\
                                        np.array(ensemble_features), axis=0))[0]
            verified     = np.argmax(prediction) == 1
            score        = prediction[np.argmax(prediction)]
            resp_obj     = {'target': target_name, 'ref': gallery_name,
                            'verified': verified, 'score': score,
                            'distance': ensemble_features,
                            'model': ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace'],
                            'similarity_metric': ['cosine', 'euclidean', 'euclidean_l2']}

            resp_objects.append(resp_obj)

    #-------------------------

    resp_obj = {}

    for i in range(0, len(resp_objects)):
        resp_item   = resp_objects[i]
        target_name = resp_item['target'].split('.')[0]
        ref_name    = resp_item['ref'].split('.')[0]
        resp_obj[f'pair_{i+1:05}_{target_name}_vs_{ref_name}'] = resp_item
        
    toc = time.time()

    return resp_obj, toc

# =================================================================

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



# =============================================================================

class Representation():
    """
    Class to store the model-specific embeddings for an image.
    
    Attributes:
        After __init__:
            1. unique_id  - unique identifier number that represents a face
            2. image_name - image name
            3. image_fp   - image full path
            4. embeddings - model-specific vector representation of the face
        
    Methods:
        1. show_info() - prints the representation information in a condensed,
            easy-to-read form
    """
    def __init__(self, unique_id, image_name, image_fp='', embeddings={}):
        self.unique_id  = unique_id
        self.image_name = image_name
        self.image_fp   = image_fp
        self.embeddings = embeddings
        
    def show_info(self):
        print('Unique ID'.ljust(25) + f': {self.unique_id}',
              'Original image name'.ljust(25) + f': {self.image_name}',
              'Original image full path'.ljust(25) + f': {self.image_fp}',
              sep='\n')
        
        if self.embeddings: # embeddings dictionary is NOT empty
            print('Embeddings:')
            for key, value in self.embeddings.items():
                print(f'  > {key}: [{value[0]}, {value[1]}, {value[2]}, ... , {value[-1]}] (len={len(value)})')
                
        else:
            print('No embedding found!')

