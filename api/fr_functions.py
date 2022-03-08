# ==============================================================================
#                        FACE RECOGNITION FUNCTIONS
# ==============================================================================

# Module / package imports
import os
import cv2
import faiss
import pickle
import shelve
import numpy              as np
import tensorflow         as tf

from tqdm                    import tqdm
from uuid                    import uuid4
from keras.preprocessing     import image
from deepface.DeepFace       import build_model as build_verifier
from deepface.commons        import functions, distance as dst
from utility_functions       import create_dir
from api_functions           import API_DIR, detect_faces
from api_classes             import Representation

from deepface.detectors.FaceDetector import build_model as build_detector

tf_version = int(tf.__version__.split(".")[0])

# if tf_version == 2:
#     import logging
#     tf.get_logger().setLevel(logging.ERROR)
#     from tensorflow.keras.preprocessing import image
# else:
#     from keras.preprocessing import image

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
    pbar           = tqdm(range(0, n_verifiers), desc='Saving verifiers', disable = disable_option)
    
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

def load_face_verifier(verifier_names, save_dir='', show_prog_bar=True,
                        verbose=False):
    """
    Loads all face verifier models specified in 'verifier_names'.
    Alternatively, 'all' can be passed as a 'verifier_name' to load all
    saved models. Uses the global 'api_root_dir' variable to determine
    the root directory from which the models will be loaded if the
    'save_dir' directory is empty (i.e. save_dir=''). Otherwise, attempts
    to load the models from 'save_dir'.
    
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
    
    # TODO: Add save_dir creation to ensure directory exists
    
    # Ensures that the verifier_names is a list (even a single name is provided)
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]
        
    # If 'all' was provided, use all model names
    if verifier_names[0].lower() == 'all':
        verifier_names = ['VGG-Face', 'OpenFace', 'Facenet', 'Facenet512',
                          'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'Emotion',
                          'Age', 'Gender', 'Race']
    
    # Creates the progress bar
    n_verifiers    = len(verifier_names)
    disable_option = not show_prog_bar
    pbar           = tqdm(range(0, n_verifiers), desc='Loading verifiers', disable = disable_option)
    
    models = {}
    for index, verifier_name in zip(pbar, verifier_names):       
        # Checks if the face verifier model exists
        if saved_verifier_exists(verifier_name):
            # Loads the model
            file_fp = os.path.join(save_dir, verifier_name)
            with shelve.open(file_fp) as model:
                if verbose:
                    print(f'Loading model {verifier_name}: ', end='')
                try:
                    models[verifier_name] = model[verifier_name]
                    if verbose:
                        print('success!')
                except Exception as excpt:
                    if verbose:
                        print(f'failed! Reason: {excpt}\nAttempting to build & save model from scratch: ', end='')
                    try:
                        cur_model = build_verifier(verifier_name)
                        models[verifier_name] = cur_model
                        save_face_verifiers(cur_model, show_prog_bar=False, overwrite=True)
                        if verbose:
                            print('success!')
                    except Exception as excpt:
                        if verbose:
                            print(f'failed! Reason: {excpt}')
                                
        else:
            # Otherwise, tries to build model from scratch & save it
            if verbose:
                print(f'Model {verifier_name} does not exist.',
                      'Attempting to build & save model from scratch: ', sep='\n', end='')
            try:
                cur_model = build_verifier(verifier_name)
                models[verifier_name] = cur_model
                save_face_verifiers(cur_model, show_prog_bar=False, overwrite=False)
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
            db = pickle.load(open(file_path, 'wb'))
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
            threshold value used (key: threshold) and the distances calculated
            using the specified metric (key: distances). Note that if no match
            is found, then the 'indexes' will have a length of zero (i.e. will
            be empty).
    
    Signature:
        match_index_obj = calc_similarity(tgt_embd, embds, metric='cosine',
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
    
    # Makes a decision
    decision = (distances <= threshold).squeeze()
    return {'idxs': np.where(decision)[0],
            'threshold': threshold,
            'distances': distances}

# ------------------------------------------------------------------------------

# def calc_representations(img_paths, model_name='VGG-Face', model=None,
#                          grayscale=False, align=True, normalization='base',
#                          unique_id_start=0):
    
#     # Currently, this function does not support multiple models or Ensemble
#     if model_name == 'Ensemble':
#         print('[WARNING] Ensemble and multiple models currently unsupported! Returning empty representation.')
#         return []
    
#     # Builds the face verifier model - uses input variables to store result from this
#     # function, maybe using a different name is clearer?
#     model_names, junk, models = build_face_verifier(model_name=model_name, model=model)
    
#     representations = []
#     unique_id       = unique_id_start
    
#     # Loops through each image
#     for img_path in img_paths:
#         embeddings = {} # initialize embeddings dictionary
#         img_name   = img_path.split('/')[-1] # gets image name
        
#         # For the current image, loops through each model and create its representation
#         for name, model in zip(model_names, models.values()):
#             # Find current model's input shape and process the face image
#             input_shape_x, input_shape_y = functions.find_input_shape(model)
#             face = process_single_face(img_path, target_size = (input_shape_x, input_shape_y),
#                                        grayscale=grayscale)
            
#             # Normalizes input and finds the embedding
#             face = functions.normalize_input(img=face, normalization=normalization)
#             embeddings[name] = model.predict(face)[0].tolist()
            
#         # Create representation and increment unique id
#         representations.append(Representation(unique_id, img_name,
#                                               image_fp=img_path,
#                                               embeddings=embeddings))
#         unique_id += 1
    
#     return representations, unique_id

# ------------------------------------------------------------------------------

def calc_embedding(img_path, verifier_models, detector_name='VGG-Face',
                    align=True, verifier_names='opencv', normalization='base'):
    """
    TODO: ADD DESCRIPTION
    """
    # Converts verifier names into a list if it is a single entry
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]

    # Detect faces & align:
    output = detect_faces(img_path, detector_backend=detector_name,
                          align=align, return_type='both', face_detector=None)

    # TODO: MAKE THE FUNCTION ACCEPT MULTIPLE FACES IN ONE IMAGE
    # Since we assume there is only 1 face (and region):
    face   = output['faces'][0]
    region = output['regions'][0]

    # For each verifier model provided
    embeddings={}
    for verifier_name in verifier_names:
        # Gets the current verifier model
        model = verifier_models[verifier_name]

        # Determine target size
        input_x, input_y = functions.find_input_shape(model)

        # Process face
        processed_face = process_face(face, target_size=(input_x, input_y),
                                normalization=normalization, grayscale=False)

        # Calculate embeddings
        embeddings[verifier_name] = model.predict(processed_face)[0]

    return (region, embeddings)

# ------------------------------------------------------------------------------

def create_new_representation(img_path, region, embeddings, tag='', uid='',
                                ignore_taglist=['--', '---']):

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



# ------------------------------------------------------------------------------


