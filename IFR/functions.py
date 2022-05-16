# ==============================================================================
#                                   FUNCTIONS
# ==============================================================================

# Module / package imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import cv2
import imagesize

import numpy                 as np

from tqdm                    import tqdm
from filecmp                 import cmp
from deepface                import DeepFace
from deepface.basemodels     import Boosting
from deepface.detectors      import FaceDetector
from keras.preprocessing     import image

from deepface.commons                import functions, distance as dst
from deepface.DeepFace               import build_model        as build_verifier
from deepface.detectors.FaceDetector import build_model        as build_detector

# ______________________________________________________________________________
#                           UTILITY & GENERAL USE
# ------------------------------------------------------------------------------

def get_image_paths(root_path, file_types=('.jpg', '.png')):
    """
    Gets the full paths of all 'file_types' files in the 'root_path' directory
    and its subdirectories. If the 'root_path' provided points to a file, this
    functions simply returns that path as a list with 1 element.

    Inputs:
        1. root_path  - full path of a directory [string].

        2. file_types - file extensions to be considered [tuple of strings,
                        default=('.jpg', '.png')].
    Output:
        1. list containing full path of all files in the directory and its
            subdirectories

    Signature:
        all_images_fps = get_image_paths(root_path, file_types=('.jpg', '.png'))
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
        1. dir_path - full path of new directory [string].
    
    Outputs:
        1. boolean to indicate successfull directory creation (0) or failure
            because directory already exists (1)
    
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

def ensure_dirs_exist(directory_list, verbose=False):
    """
    Ensures that all directories in 'directory_list' exist. Note that
    'directory_list' should contain the full paths of each directory. This
    function tries to create each directory that does not exist.

    Inputs:
        1. directory_list - directory full paths [string or list of strings].

        2. verbose        - toggles if the function should output useful
                            information to the console [boolean, default=False].

    Output:
        1. None

    Signature:
        ensure_dirs_exist(directory_list, verbose=False)
    """
    # Loops through each directory's full path in 'directory_list'
    for dir_fp in directory_list:
        if verbose:
            print(f'[ensure_dirs_exist] Creating {dir_fp} directory: ', end='')
    
        if create_dir(dir_fp):
            if verbose:
                print('directory exists. Continuing...')
        else:
            if verbose:
                print('success.')

    return None

# ------------------------------------------------------------------------------

def has_same_img_size(fpath1, fpath2):
    """
    Determines if the 2 files with paths 'fpath1' and 'fpath2' have the same
    image size (i.e. the same width and height). If any of the 2 files are not a
    valid image file, returns False. On any error, also returns False.

    TODO: Finish doc
    """
    try:
        w1, h1 = imagesize.get(fpath1)
        w2, h2 = imagesize.get(fpath2)

        return True if w1 == w2 and h1 == h2 else False
    except:
        return False

# ------------------------------------------------------------------------------

def img_files_are_same(fpath1, fpath2):
    """
    Compares if both image files, with paths 'fpath1' and 'fpath2' respectively,
    are the same. They are considered to be the same if:
        1. they have the same size
        2. they have the same content
        3. the images have the same width and height

    Checks 1 and 2 are performed using filecmp library's cmp() method with
    shallow=False. Check 3 is performed using the efficient imagesize library
    which is able to extract the image's width and height without loading it
    fully into memory.

    Inputs:
        1. fpath1 - full path of the first image file [string].

        2. fpath2 - full path of the first image file [string].

    Output:
        1. flag indicating if both files are the same or not [boolean].

    Signature:
        is_same = img_files_are_same(fpath1, fpath2)
    """
    return True if cmp(fpath1, fpath2, shallow=False)\
            and has_same_img_size(fpath1, fpath2) else False

# ------------------------------------------------------------------------------

def remove_img_file_duplicates(trgt_dir, dont_delete=False):
    """
    Detects and removes (if dont_delete=False) all duplicate image files in a
    target directory 'trgt_dir'. Also returns a list with the name of all
    duplicate files, regardless if they were deleted or not. The algorithm works
    in the following way:

        1. The full path and file size of all files (in the directory) are
            obtained. A list with all unique file sizes is calculated.

        2. For each file size in the unique file size list:
            2.1. The indicies of all files with a matching file size are
                  obtained.

            2.2. If there are multiple matches, the first file (corresponding to
                  the first index) is set as the reference file for comparison.
                  Its width and height are calculated without loading the entire
                  image to memory.

            2.3. Every other match is compared to the reference file. The
                  comparison is made by using filecmp.cmp() (with
                  shallow=False). Their widths and heights are also calculated
                  and compared to the reference image's width and height.

            2.4. If any match is deemed the same (filecmp.cmp() results in True
                  and has the same width and height), the matching file is
                  considered a duplicate and is deleted (unless
                  dont_delete=True). The file's name is also stored in the
                  duplicate file names' list.

        3. Returns a list with the names of all duplicate files (regardless if
            they were deleted or not).

    Inputs:
        1. trgt_dir    - path to target directory [string].

        2. dont_delete - toggles if the function should delete the duplicate
                          files or not [boolean, default=False].

    Output:
        1. Returns the names of all duplicate files (regardless if they were
            deleted or not) [list of strings].

    Signature:
        dup_file_names = remove_img_file_duplicates(trgt_dir, dont_delete=False)
    """
    # Initialize duplicate files' name list
    dup_files = []

    # Obtains all file full paths ('all_files'), their file sizes ('all_sizes')
    # and a list of all unique file sizes ('unq_sizes')
    all_files = [os.path.join(trgt_dir, pth) for pth in os.listdir(trgt_dir)]
    all_sizes = np.array([os.path.getsize(pth) for pth in all_files])
    unq_sizes = np.unique(all_sizes)

    # Loops through all unique file sizes
    for sze in unq_sizes:
        # Gets the indices of all files with the same current file size
        ii = np.where(all_sizes == sze)[0]

        # If there are multiple matches, compare them to see if there are
        # duplicates. Otherwise, just continue
        if len(ii) > 1:
            # Sets the first index (file) as a reference file (for comparison)
            # and obtains their width and height
            refw, refh = imagesize.get(all_files[ii[0]])

            # Loops through each remaining file index
            for i in ii[1:]:
                # Calculates the current matched file's width and height
                wi, hi = imagesize.get(all_files[i])

                # Files have the same size, content and image size
                if cmp(all_files[ii[0]], all_files[i], shallow=False)\
                    and refw == wi and refh == hi:
                    # Appends the duplicate file's name
                    dup_files.append(all_files[i])

                    # Removes the duplicate file if dont_delete=False
                    if not dont_delete:
                        os.remove(all_files[i])

    return dup_files

# ------------------------------------------------------------------------------

def ext_is_valid(fpath, valid_exts=['.jpg', '.png', '.npy']):
    """
    Checks if 'fpath' extension is contained in the 'valid_exts' list, returning
    True if so and False otherwise. Note that 'fpath' can be a full or relative
    path or a file name.

    Inputs:
        1. fpath      - full / relative file path or file name [string].

        2. valid_exts - list of valid file extensions [list of strings].

    Output:
        1. returns a boolean indicating if 'fpath' extension is in the
            'valid_exts' list

    Signature:
        flag = ext_is_valid(fpath, valid_exts=['.jpg', '.png', '.npy'])
    """
    return fpath[fpath.rindex('.'):].lower() in valid_exts

# ------------------------------------------------------------------------------

def filter_files_by_ext(fpaths, valid_exts=['.jpg', '.png', '.npy']):
    """
    Filters file paths / names in 'fpaths' list if they have a valid extension.
    A valid extension is any extension in the 'valid_exts' list.

    Inputs:
        1. fpaths     - list of file paths / names [list of strings].

        2. valid_exts - list of valid file extensions [list of strings].

    Output:
        1. returns a filtered list of file paths / names, where each element has
            an extension contained in 'valid_exts' list.

    Signature:
        filtered_fpaths = filter_files_by_ext(fpaths,
                                            valid_exts=['.jpg', '.png', '.npy'])
    """
    # Initializes the valid file paths / names list
    valid_fpaths = []

    # Loops through each file path / name in the 'fpaths' list
    for fpath in fpaths:
        # Adds the current path / name to the 'valid_fpaths' list if it has a
        # valid extension
        if fpath[fpath.rindex('.'):].lower() in valid_exts:
            valid_fpaths.append(fpath)

    return valid_fpaths


# ______________________________________________________________________________
#                    FACE DETECTION / VERIFICATION RELATED
# ------------------------------------------------------------------------------

def detect_faces(img_path, use_detector='retinaface', align=True,
                 return_type='both', face_detector=None):
    """
    Detects faces in an image (and optionally aligns them).
    
    Inputs:
        1. img_path      - image path, base64 image or numpy array image [string
                            / base64 image / numpy array].
        
        2. use_detector  - face detector name. Options: opencv, ssd, dlib, mtcnn
                            or retinaface [string, default='retinaface'].
        
        3. align         - toggles if the faces should be aligned [boolean,
                            default=True].
        
        4. return_type   - controls if the function should return the faces,
                            regions or both. Options: faces, regions, both
                            [string, default='both'].

        5. face_detector - face detector object or None. If a face detector
                            object is provided, then the function does not build
                            one from scratch, improving execution time. For more
                            information see help(FaceDetector.build_model)
                            [face detector object / None].
            
    Outputs:
        If return_type='regions':
            Dictionary containing list of face detections. The face detections
            (or regions of interests - rois) are lists with the format
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
        output = detect_faces(img_path, use_detector='retinaface', align=True,
                              return_type='both', face_detector=None)
    """
    # Raises an error if return type is not 'faces', 'regions' or 'both'.
    # Otherwise, initializes lists.
    if   return_type == 'both':
        faces = []
        rois  = []
    
    elif return_type == 'faces':
        faces = []
        
    elif return_type == 'regions':
        rois  = []

    else:
        raise ValueError("Return type should be 'faces', 'regions' or 'both'.")
    
    # Loads image. Image might be path, base64 or numpy array. Convert it to
    # numpy whatever it is.
    img = functions.load_image(img_path)

    # The detector is stored in a global variable in FaceDetector object.
    # This call should be completed very fast because it will return found in
    # memory and it will not build face detector model in each call (consider
    # for loops)
    if face_detector == None:
        face_detector = FaceDetector.build_model(use_detector)

    # Tries to detect faces
    try:
        detections = FaceDetector.detect_faces(face_detector, use_detector, img,
                                                align)
    except Exception as excpt:
        print(f'[detect_faces] Face detection failed! (reason: {excpt})')

    # Prints a warning and returns an empty dictionary and error if no faces
    # were found, otherwise processes faces & regions
    if len(detections) == 0:
        print('[detect_faces] Face detection failed!',
              '(check if the image contains any faces)')

    else:
        # Loops through each face & region pair
        for face, roi in detections:
            # Only stores images (faces) if the return type is 'faces' or 'both'
            if return_type == 'faces' or return_type == 'both':
                faces.append(face)
    
            # Only stores regions (rois) if the return type is 'regions' or
            # 'both'
            if return_type == 'regions' or return_type == 'both':
                rois.append(roi)
  
    # Returns the appropriate dictionary based on 'return_type'
    if return_type == 'faces':
        return {'faces':faces}
    elif return_type == 'regions':
        return {'regions':rois}
    else:
        assert return_type == 'both', "Return type should be 'both'."
        return {'faces':faces, 'regions':rois}

# ------------------------------------------------------------------------------

def do_face_detection(img_path, detector_models={}, detector_name='retinaface',
                        align=True, verbose=False):
    """
    Performs the face detection step. The function loads the chosen face
    detector 'detector_name' from the 'detector_models' dictionary. Then, the
    function tries to detect faces in the image provided. If the face detection
    fails or finds no faces, then a None output is returned.

    Inputs:
        1. img_path        - image path, base64 image or numpy array image
                              [string / base64 image / numpy array].

        2. detector_models - all face detector models, where each face detector
                              name (key) corresponds to a built model object
                              (value) [dictionary, default={}].

        3. detector_name   - face detector name [string, default='retinaface'].

        4. align           - toggles if face alignment should be performed
                             [boolean, default=True].

        5. verbose         - toggles if the function should output information
                              to the console [boolean, default=True].

    Output:
        1. dictionary containing the faces detected (key:'faces') [list of numpy
            arrays] and face regions (key:'regions') [list of lists of 4
            integers].

    Signature:
        output = do_face_detection(img_path, detector_models={},
                        detector_name='retinaface', align=True, verbose=False)
    """
    # Initializes output object
    output = None

    # Loads the chosen face detector
    face_detector = detector_models[detector_name]

    # Tries to detect faces in the image provided and align (if required):
    try:
        if verbose:
            print('[do_face_detection] Detecting faces: ', end='')

        output = detect_faces(img_path, use_detector=detector_name,
                                align=align, return_type='both',
                                face_detector=face_detector)
    except Exception as excpt:
        if verbose:
            print(f'failed! (reason: {excpt}) (img: {img_path})')

    # Checks if the face detector was able to find any face if verbose is True
    if verbose:
        if len(output['faces']) == 0:
            print(f'failed! (reason: No face found.',
                   'Ensure the image provided has at least 1 face.)',
                  f'(img: {img_path})')
        else:
            print(f'success! (img: {img_path})')

    return output

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
        1. img_path      - image path, base64 image or numpy array

        2. target_size   - desired X and Y dimensions [tuple or 2 integers,
                            default=(224, 224)]
        
        3. normalization - defines a type of normalization [string,
                            default='base']
        
        4. grayscale     - toggles if an image should be converted to grayscale
                            [boolean, default=False]

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

# ------------------------------------------------------------------------------

def discard_small_regions(regions, image_size, pct=0.02):
    """
    Discards small detection regions in 'regions'. A region is considered
    'small' if its area is smaller than the area of the original image, given by
    'image_size', multiplied by a percentage factor 'pct' and rounded up to the
    nearest pixel (integer). The function then returns the filtered regions and
    an index list with the index of each valid region.

    Inputs:
        1. regions    - list of regions, where each region is a list of 4
                          integers corresponding to the coordinates of the top
                          left corner, width and height [list of list of 4
                          integers].

        2. image_size - width and height (or height and width) of the image
                          [tuple or list with 2 elements].

        3. pct        - percentage of image area as a decimal [float,
                          default=0.02].

    Output:
        1. filtered list of regions
        2. valid region index list

    Signature:
        filt_rgns, idxs = discard_small_regions(regions, image_size, pct=0.02)
    """
    # Initializes filtered regions and idxs list
    filt_regions = []
    idxs         = []

    # Calculates the area threshold, which is obtained from the area of the
    # original image (image_size[0] * image_size[1]) multipled by a percentage
    # 'pct'. This result is rounded up to the nearest pixel (i.e. integer) and
    # ensured to be greater than or equal to 1
    threshold = np.maximum(np.ceil(pct * image_size[0] * image_size[1]), 0)

    # Loops through each region in the 'regions' list
    for i, region in enumerate(regions):
        # Stores the current region if it is greater than or equal to the
        # threshold.
        if region[2] * region[3] >= threshold:
            filt_regions.append(region)
            idxs.append(i)

    # Returns the filtered regions and idxs list
    return filt_regions, idxs

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

def batch_build_detectors(detector_names, show_prog_bar=True, verbose=False):
    """
    Builds batches of face detectors. The face detectors to be built are
    specified by the 'detector_names' list of names. If a face detector cannot
    be built (or results in an error), it is simply skipped. All errors are
    suppressed unless verbose is True, in which case they are printed to the
    console.
    
    Inputs:
        1. detector_names - face detectors names [list of strings].

        2. show_prog_bar  - toggles if a progress bar is shown [boolean,
                            default=True].

        3. verbose        - toggles the amount of text output [boolean,
                            default=False].
        
    Output:
        1. dictionary of built face detector models
        
    Signature:
        detectors = batch_build_detectors(detector_names, show_prog_bar=True,
                                            verbose=False)
    """
    # Creates the progress bar
    n_detectors = len(detector_names)
    pbar        = tqdm(range(0, n_detectors), desc='Building detectors',
                        disable=(not show_prog_bar))
    
    # Builds the face detector models and stores them in a dictionary
    detectors = {}
    for index, detector_name in zip(pbar, detector_names):
        try:
            detectors[detector_name] = build_detector(detector_name)
            
        except Exception as excpt:
            if verbose:
                print('[batch_build_detectors]',
                     f'Building {detector_name} failed!',
                     f'(reason: {excpt})')
    
    return detectors

# ------------------------------------------------------------------------------

def batch_build_verifiers(verifier_names, show_prog_bar=True, verbose=False):
    """
    Builds batches of face verifiers. The face verifiers to be built are
    specified by the 'verifier_names' list of names. If a face verifier cannot
    be built (or results in an error), it is simply skipped. All errors are
    suppressed unless verbose is True, in which case they are printed to the
    console.
    
    Inputs:
        1. detector_names - face detectors names [list of strings].

        2. show_prog_bar  - toggles if a progress bar is shown [boolean,
                            default=True].

        3. verbose        - toggles the amount of text output [boolean,
                            default=False].
        
    Output:
        1. dictionary of built face verifier models
        
    Signature:
        verifiers = batch_build_verifiers(verifier_names, show_prog_bar=True,
                                            verbose=False)
    """
    # Creates the progress bar
    n_verifiers    = len(verifier_names)
    pbar           = tqdm(range(0, n_verifiers), desc='Building verifiers',
                            disable = (not show_prog_bar))

    # Builds the face verifier models and stores them in a dictionary
    verifiers = {}
    for index, verifier_name in zip(pbar, verifier_names):
        try:
            verifiers[verifier_name] = build_verifier(verifier_name)

        except Exception as excpt:
            if verbose:
                print('[batch_build_verifiers]',
                     f'Building {verifier_name} failed!',
                     f'(reason: {excpt})')
        
    return verifiers

# ------------------------------------------------------------------------------

def ensure_detectors_exists(models={}, detector_names=['retinaface'],
                                verbose=False):
    """
    Ensures that each face detector with a name in 'detector_names' exists in
    the 'models' dictionary. For each face detector, if it does not exist, this
    function tries to build it from scratch. If the building process fails, the
    detector is simply skipped and the return value will be False. If all
    detectors exist and/or are successfully built, then True is returned. The
    updated model dictionary is always returned as a second output.

    Inputs:
        1. models         - face detectors' name (key) and object (value)
                            [dictionary, default={}].

        2. detector_names - face detectors' name [string or list of strings,
                            default=['retinaface']].

        3. verbose        - toggles if the function should print useful
                            information to the console [boolean, default=False].

    Output:
        1. flag indicating if all face detectors exist and/or were successfully
            built (True) or not (False) [boolean].

        2. updated model dictionary [dictionary].

    Signature:
        ret, models = ensure_detectors_exists(models={}, verbose=False,
                                             detector_names=['retinaface'])
    """
    # Initializes no error flag
    no_errors = True

    # Ensures model names is a list
    if not isinstance(detector_names, list):
        detector_names = [detector_names]

    # Prints information if verbose is True
    if verbose:
        print('[ensure_detectors_exists] Checking existence of face detectors:')

    # Loops through each name in the detector names list
    for name in detector_names:
        if verbose:
            print(f'  > Checking {name}: ', end='')
        
        # First, checks if the current face detector exists in the 'models'
        # dictionary provided. This only checks for the key, not the integrity
        # of the respective detector object.
        try:
            test = models[name]
            if verbose:
                print('success!')
        except:
            if verbose:
                print('failed! (building detector:', end='')

            # On failure, tries to build the detector from scratch, skipping it
            # if the building process fails
            try:
                models[name] = build_detector(name)
                if verbose:
                    print('success!')
            except Exception as excpt:
                no_errors = False
                if verbose:
                    print(f'failed > reason: {excpt})')

    return (no_errors, models)

# ------------------------------------------------------------------------------

def ensure_verifiers_exists(models={}, verifier_names=['ArcFace'],
                                verbose=False):
    """
    Ensures that each face verifier with a name in 'verifier_names' exists in
    the 'models' dictionary. For each face verifier, if it does not exist, this
    function tries to build it from scratch. If the building process fails, the
    verifier is simply skipped and the return value will be False. If all
    verifiers exist and/or are successfully built, then True is returned. The
    updated model dictionary is always returned as a second output.

    Inputs:
        1. models         - face verifiers' name (key) and object (value)
                            [dictionary, default={}].

        2. verifier_names - face verifiers' name [string or list of strings,
                            default=['ArcFace']].

        3. verbose        - toggles if the function should print useful
                            information to the console [boolean, default=False].

    Output:
        1. flag indicating if all face verifiers exist and/or were successfully
            built (True) or not (False) [boolean].

        2. updated model dictionary [dictionary].

    Signature:
        ret, models = ensure_verifier_exists(models={}, verbose=False,
                                             verifier_names=['retinaface'])
    """
    # Initializes no error flag
    no_errors = True

    # Ensures model names is a list
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]

    # Prints information if verbose is True
    if verbose:
        print('[ensure_verifiers_exists] Checking existence of face verifiers:')

    # Loops through each name in the verifier names list
    for name in verifier_names:
        if verbose:
            print(f'  > Checking {name}: ', end='')
        
        # First, checks if the current face verifier exists in the 'models'
        # dictionary provided. This only checks for the key, not the integrity
        # of the respective verifier object.
        try:
            test = models[name]
            if verbose:
                print('success!')
        except:
            if verbose:
                print('failed! (building verifier:', end='')

            # On failure, tries to build the verifier from scratch, skipping it
            # if the building process fails
            try:
                models[name] = build_verifier(name)
                if verbose:
                    print('success!')
            except Exception as excpt:
                no_errors = False
                if verbose:
                    print(f'failed > reason: {excpt})')

    return (no_errors, models)

# ______________________________________________________________________________
#                           SIMILARITY & DISTANCE RELATED
# ------------------------------------------------------------------------------

def calc_cosine_similarity(A, B):
    """
    Calculates the cosine similarity metric between matrices A and B. If A is a
    vector, it is converted into a matrix so that the cosine metric can be
    calculated normally.

    Inputs:
        1. A - N x M matrix with N embeddings with M elements [numpy array].

        2. A - I x M matrix with I embeddings with M elements [numpy array].

    Outputs:
        1. matrix of cosine similarity metric between A and B

    Signature:
        csm = calc_cosine_similarity(A, B)
    """
    # Creates a new axis if necessary to ensure the sizes match
    if A.ndim == 1:
        A = A[np.newaxis, :]

    # Calculates (in a vectorized manner) elements required for the cosine
    # distance
    num = np.dot(A, B.T)
    p1  = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis]
    p2  = np.sqrt(np.sum(B**2, axis=1))[np.newaxis, :]

    # Returns the cosine distance metric
    return 1 - (num / (p1 * p2))

# ------------------------------------------------------------------------------

def calc_euclidean_similarity(A, B, l2_normalize=False):
    """
    Calculates the Euclidean similarity metric between matrices A and B. If A is
    a vector, it is converted into a matrix by repeating (and stacking) it
    horizontally until it has the correct dimensions. If 'l2_normalize' is set
    to True, then the function applies L2 normalization to the inputs before
    calculating the Euclidean similarity (distance). All of this is vectorized
    to improve execution times.

    Inputs:
        1. A            - N x M matrix with N embeddings with M elements
                            [numpy array].

        2. A            - I x M matrix with I embeddings with M elements
                            [numpy array].

        3. l2_normalize - toggles the L2 normalization of inputs before
                            calculating the similarity [boolean, default=True].

    Outputs:
        1. matrix of (possibly L2 normalized) Euclidean similarity metric
            between A and B.

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

def calc_similarity(tgt_embd, embds, metric='cosine', face_verifier='ArcFace',
                    threshold=-1):
    """
    Calculates the similarity (distance) between both embeddings ('tgt_embd'
    and 'embds') using the 'metric' distance metric. If the 'threshold' < 0 then
    it is automatically determined based on the 'face_verifier' provided. If a 
    custom threshold is specified, then the 'face_verifier' input is unused.

    Note that 'embds' can be a N x M matrix (N embeddings each with M elements)
    and 'tgt_embd' can only be a 1 x M embedding.

    Inputs:
        1. tgt_embd   - 1-D target embedding [numpy array].

        2. embds      - 1-D embedding or 2-D embedding matrix [numpy array].

        3. metric     - chosen distance metric. Options: cosine, euclidean or
                        l2_euclidean [string, default='cosine'].

        4. model_name - face verifier name [string, default='ArcFace'].

        5. threshold  - threshold provided. If it is negative, then the it is
                        automatically determined based on the face verifier
                        name [float, default=-1].

    Output:
        1. dictionary containing the indexes of matches (key: idxs), the
            threshold value used (key: threshold) and the distances (using the
            specified metric) of the matches (key: distances). Note that if no
            match is found, then the 'indexes' will have a length of zero (i.e.
            will be empty).
    
    Signature:
        similarity_obj = calc_similarity(tgt_embd, embds, metric='cosine',
                                         face_verifier='ArcFace', threshold=-1)
    """
    # Calculates the distance based on the metric provided, otherwise raises a
    # value error
    if   metric == 'cosine':
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
        threshold = dst.findThreshold(face_verifier, metric)
    
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

def calc_embeddings(faces, verifier_models, verifier_names=['ArcFace'],
                    normalization='base'):
    """
    Calculates the embeddings for each face verifier in 'verifier_names', for
    each face in 'faces'. An embedding is a vector representation of a face
    image, obtained from a face verifier.
    
    A 'normalization' is performed on each face image prior to embedding
    calculation.
    
    If a face verifier does not exist in the 'verifier_models' dictionary, or if
    the embedding calculation fails, it is skipped. This results in a dictionary
    with successfully calculated (verifier name, embedding) key, value pairs.

    Inputs:
        1. faces           - 3-D numpy arrays corresponding to face images
                                [list of numpy arrays].
        
        2. verifier_models - all face verifier models, where each face verifier
                                name (key) corresponds to a built model object
                                (value) [dictionary].
        
        3. verifier_names  - face verifiers' names [list of strings,
                                default=['ArcFace']].
        
        4. normalization   - type of normalization to be performed on the face
                                image prior to embedding calculation [string,
                                default='base'].

    Output:
        1. list of embedding dictionaries, where each element of the list
            corresponds to embeddings of the input faces. E.g. faces = [face1,
            face2, ...] then output = [face1 embeddings, face2 embeddings, ...].

    Signature:
        output = calc_embeddings(faces, verifier_models,
                            verifier_names=['ArcFace'], normalization='base')
    """
    # Converts verifier names into a list if it is a single entry
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]

    # Initializes 'embeddings' list
    embeddings = []

    # Loops through each detected face
    for face in faces:
        # Initializes current embeddings
        cur_embds = {}

        # Loops through each face verifier in verifier names
        for verifier_name in verifier_names:
            # Tries to calculate the face's embeddings using the current face
            # verifier model
            try:
                # Gets the current face verifier model
                verifier  = verifier_models[verifier_name]

                # Determine target size
                input_x, input_y = functions.find_input_shape(verifier)

                # Process face
                p_face = process_face(face, target_size=(input_x, input_y),
                                normalization=normalization, grayscale=False)

                # Calculates the embeddings
                cur_embds[verifier_name] = verifier.predict(p_face)[0]

            except Exception as excpt:
                print('[calc_embedding]',
                     f'Calculation of {verifier_name} embeddings failed!',
                     f'(reason: {excpt})')

        # Stores the calculated embeddings
        embeddings.append(cur_embds)

    return embeddings

# ------------------------------------------------------------------------------
