# ==============================================================================
#                             API FUNCTIONS
# ==============================================================================

# Module / package imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle

import numpy                 as np
import matplotlib.image      as mpimg
import matplotlib.pyplot     as plt
import api.global_variables  as glb

from io                      import BytesIO
from tqdm                    import tqdm
from filecmp                 import cmp
from zipfile                 import ZipFile
from tempfile                import TemporaryDirectory
from sqlalchemy              import create_engine, inspect, MetaData, select,\
                                    insert, update, delete, text, update
from sqlalchemy.orm          import sessionmaker
from IFR.classes             import FaceRep, Person, VerificationMatch,\
                                    ProcessedFilesTemp, ProcessedFiles, Base,\
                                    tempClustering
from IFR.functions           import get_image_paths, do_face_detection,\
                                    calc_embeddings, ensure_detectors_exists,\
                                    ensure_verifiers_exists,\
                                    discard_small_regions,\
                                    rename_file_w_hex_token,\
                                    flatten_dir_structure, image_is_uncorrupted
from sklearn.cluster         import DBSCAN

from shutil                          import move           as sh_move
from deepface.DeepFace               import build_model    as build_verifier
from deepface.detectors.FaceDetector import build_model    as build_detector

# ______________________________________________________________________________
#                       UTILITY & GENERAL USE FUNCTIONS
# ------------------------------------------------------------------------------

def show_cluster_results(group_no, db, ncols=4, figsize=(15, 15), color='black',
                         add_separator=False):
    """
    Creates a figure with all images belonging to a group with 'group_no' and
    returns the handle to its figure. Each image belonging to the group is
    displayed in a 'ncols' by nrows grid. Nrows is calculated by:

        nrows = (np.ceil(np.sum(labels == group_no) / ncols)).astype(int)

    Example: 10 matches with ncols=4 results in nrows=3. The figure will have 4
    images on the first row, another 4 images on the second row and 2 images
    (and 2 'blank' images) on the third row.

    Inputs:
        1. group_no      - a group number [integer].

        2. db            - representation database [RepDatabase].

        3. ncols         - number of columns (or images per row)
                            [integer, default=4].

        4. figsize       - size of figure (in X and Y directions) in inches
                            [tuple of floats, default=(15, 15)].

        5. color         - font color (must be a valid matplotlib color string)
                            [string, default='black'].

        6. add_separator - adds a 'textual' separator at the bottom of the
                            figure. This separator is simply a sequence of '='
                            characters [boolean, default=False].

    Output:
        1. image file handle (handle to a file-like object).

    Signature:
        img_file = show_cluster_results(group_no, db, ncols=4, figsize=(15, 15),
                                        color='black', add_separator=False)
    """
    # Gets the group number of all Representations in the database
    labels = []
    for rep in db.reps:
        labels.append(rep.group_no)
    labels = np.array(labels)

    # Determines number of rows based on number of columns
    nrows = (np.ceil(np.sum(labels == group_no) / ncols)).astype(int)

    # Creates figure and suplot axes. Also flattens the axes into a single list
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs      = [ax for ax in axs.flat]

    # Initializes the current axis and loops over each label & representation
    cur_ax   = 0
    for rep in db.reps:
        # If the current label matches the chosen cluster number
        if rep.group_no == group_no:
            # Plot the representation with title
            axs[cur_ax].imshow(mpimg.imread(rep.image_fp), aspect="auto")
            axs[cur_ax].set_title(rep.image_name\
                                + f' (cluster: {rep.group_no})', color=color)
            cur_ax += 1

    # Adds a divider / separator at the end of plot
    if add_separator:
        txt = "=" * 100
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center',
                    fontsize=12, color=color)

    # Creates a BytesIO object to save the figure in
    img_file = BytesIO()
    plt.savefig(img_file, format='png')

    return img_file

# ______________________________________________________________________________
#                DETECTORS & VERIFIERS BUILDING, SAVING & LOADING
# ------------------------------------------------------------------------------

def saved_model_exists(model_name, save_dir):
    """
    Checks if a saved model exists with name 'model_name'. Does that by
    comparing the 'model_name' against the name of all files in 'save_dir'.
    
    Inputs:
        1. model_name - model name [string].

        2. save_dir   - full path of the save directory [string].
        
    Output:
        1. Boolean value indicating if the saved model exists or not.
    
    Signature:
        model_exists = saved_model_exists(model_name, save_dir)
    """
    # Save directory provided is not a directory
    if not os.path.isdir(save_dir):
        return False
    
    # Otherwise, check if model name is in the names of files in the
    # 'save_dir' directory
    is_in = False
    for file_name in os.listdir(save_dir):
        is_in = is_in or model_name in file_name
    
    return is_in

# ------------------------------------------------------------------------------

def save_built_model(model_name, model_obj, save_dir, overwrite=False,
                        verbose=False):
    """
    Saves a built face verifier or detector model ('model_obj') with name
    'model_name' as a pickled object. The model is saved in the 'save_dir'
    directory. If a model already exists with the same name, this function does
    not overwrite it unless the 'overwrite' flag is set to True. All errors are
    suppressed unless verbose is True, in which case they are printed to the
    console.
    
    Inputs:
        1. model_name - model's name [string].

        2. model_obj  - built model object [model object].

        3. save_dir   - full path of the save directory [string].

        4. overwrite  - toggles if the function should overwrite any existing
                        model with the new model if both have the same name
                        [boolean, default=False].

        5. verbose    - toggles the amount of text output [boolean,
                        default=False].
                        
    Output:
        1. returns a status flag of True on error (otherwise returns False)
    
    Signature:
        status = save_built_model(model_name, model_obj, save_dir,
                                  overwrite=False, verbose=False)
    """
    # Prints message
    if verbose:
        print(f'[save_built_model] Saving {model_name}: ', end='')

    # Checks if the save directory provided is a directory
    if not os.path.isdir(save_dir):
        if verbose:
            print(f'failed! Reason: {save_dir} is not a directory',
                   'or does not exist.')
        return True
    
    # Saves model if it does not exist or if 'overwrite' is set to True
    if not saved_model_exists(model_name, save_dir=save_dir) or overwrite:
        try:
            # Creates the file's full path
            file_fp  = os.path.join(save_dir, model_name) + '.pickle'
        
            # Saves the built model as a pickled object
            with open(file_fp, 'wb') as handle:
                pickle.dump(model_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Prints success message
            if verbose:
                print('success!')
            
        except Exception as excpt:
            # Prints exception, deletes pickle file and returns True
            try:
                os.remove(file_fp)
            except:
                pass
            
            if verbose:
                print(f'failed! Reason: {excpt}')
            
            return True
    
    # Otherwise, skips saving because the model exists
    else:
        # Prints skip message
        if verbose:
            print(f'skipped! Reason: {model_name} already exists.')
            
    return False

# ------------------------------------------------------------------------------

def load_built_model(model_name, save_dir, verbose=False):
    """
    Loads a face detector or verifier model named 'model_name'. The model is
    loaded from the 'save_dir' directory. Errors are suppressed if 'verbose' is
    set to False, otherwise they are printed to the console. A 'None' object is
    returned on error, instead of the built model object.
    
    Inputs:
        1. model_name - model's name [string].

        2. save_dir   - full path of the save directory [string].

        3. verbose    - toggles the amount of text output [boolean,
                        default=False].
                        
    Output:
        1. returns the built model object (or 'None' object on error)
            
    Signature:
        model = load_built_model(model_name, save_dir, verbose=False)
    """
    # Initializes output
    model = None

    # Checks if the save directory provided is a directory
    if not os.path.isdir(save_dir):
        raise OSError(f'Save directory does not exist ({save_dir})!')
    
    # Prints message
    if verbose:
        print('[load_built_model] Loading model',
             f'{model_name}: ', end='')

    # Checks if the saved model exists
    if saved_model_exists(model_name, save_dir=save_dir):
        # Loads the model
        file_fp = os.path.join(save_dir, model_name)
        try:
            with open(file_fp, 'rb') as handle:
                model = pickle.load(handle)
            if verbose:
                print('success!')
        except Exception as excpt:
            if verbose:
                print(f'failed! Reason: {excpt}')

    # Model does not exist at specified path
    else:
        if verbose:
            print(f'failed! Reason: {model_name}'
                  f' does not exist in {save_dir}', sep='')
    
    return model

# ------------------------------------------------------------------------------

def init_load_detectors(detector_names, saved_models_dir, models={}):
    """
    Loads all face detectors with names 'detector_names' which are saved in the
    'saved_models_dir' directory. This function is used at server start up to
    avoid the need (if possible) to build models from scatch everytime.

    If a saved face detector does not exist, this function tries to build it
    from scratch.

    Note: this function ensures that 'detector_names' is a list so it can handle
    cases where 'detector_names' is just a single string.

    Inputs:
        1. detector_names   - names of saved face detectors [string or list of
                               strings].

        2. saved_models_dir - full path of the directory containing the saved
                               face detector models [string].

        3. models           - contains model names (key) and built model objects
                               (pair). This is mainly used if one dictionary is
                               used to store different types of models, such as
                               face detectors and verifiers for example
                               [dictionary, default={}].

    Output:
        1. dictionary containing model names (key) and built model objects
            (pair) [dictionary].

    Signature:
        models = init_load_detectors(detector_names, saved_models_dir,
                                     models={})
    """
    # Ensures that 'detector_names' is a list
    if not isinstance(detector_names, list):
        detector_names = [detector_names]

    # Loops through each detector name in 'detector_names'
    for detector_name in detector_names:
        # First, try loading (opening) the model
        model = load_built_model(detector_name + '.pickle', saved_models_dir,
                                   verbose=True)

        # If successful, save the model in a dictionary
        if model is not None:
            models[detector_name] = model

        # Otherwise, build the model from scratch
        else:
            print(f'[build_detector] Building {detector_name}: ', end='')
            try:
                models[detector_name] = build_detector(detector_name)
                print('success!\n')

            except Exception as excpt:
                print(f'failed! Reason: {excpt}\n')

    return models

# ------------------------------------------------------------------------------

def init_load_verifiers(verifier_names, saved_models_dir, models={}):
    """
    Loads all face verifiers with names 'verifier_names' which are saved in the
    'saved_models_dir' directory. This function is used at server start up to
    avoid the need (if possible) to build models from scatch everytime.

    If a saved face verifier does not exist, this function tries to build it
    from scratch.

    Note: this function ensures that 'verifier_names' is a list so it can handle
    cases where 'verifier_names' is just a single string.

    Inputs:
        1. verifier_names   - names of saved face verifiers [string or list of
                               strings].

        2. saved_models_dir - full path of the directory containing the saved
                               face verifier models [string].

        3. models           - contains model names (key) and built model objects
                               (pair). This is mainly used if one dictionary is
                               used to store different types of models, such as
                               face detectors and verifiers for example
                               [dictionary, default={}].

    Output:
        1. dictionary containing model names (key) and built model objects
            (pair) [dictionary].

    Signature:
        models = init_load_verifiers(verifier_names, saved_models_dir,
                                     models={})
    """
    # Ensures that 'verifier_names' is a list
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]

    # Loops through each verifier name in 'verifier_names'
    for verifier_name in verifier_names:
        # First, try loading (opening) the model
        model = load_built_model(verifier_name + '.pickle', saved_models_dir,
                                   verbose=True)

        # If successful, save the model in a dictionary
        if model is not None:
            models[verifier_name] = model

        # Otherwise, build the model from scratch
        else:
            print(f'[build_verifier] Building {verifier_name}: ', end='')
            try:
                models[verifier_name] = build_verifier(verifier_name)
                print('success!\n')

            except Exception as excpt:
                print(f'failed! Reason: {excpt}\n')

    return models

# ------------------------------------------------------------------------------

def save_built_detectors(detector_names, saved_models_dir, overwrite=False,
                         verbose=False):
    """
    Saves built face detectors with names in 'detector_names' at the directory
    'saved_models_dir'. If a face detector with a given name already exists,
    this function does not save (skips) it UNLESS overwrite is True, in which
    case it overwrites it with the new detector.

    Inputs:
        1. verifier_names   - names of saved face detectors [string or list of
                               strings].

        2. saved_models_dir - full path of the directory containing the saved
                               face detector models [string].

        3. overwrite        - toggles between overwriting existing face detector
                               models [boolean, default=False].

        4. verbose          - toggles if the function should print useful
                               information to the console [boolean,
                               default=False].

    Output:
        1. None

    Signature:
        save_built_detectors(detector_names, saved_models_dir, overwrite=False,
                             verbose=False)
    """
    # Ensures that 'verifier_names' is a list
    if not isinstance(detector_names, list):
        detector_names = [detector_names]

    # Loops through each verifier name in 'verifier_names'
    for detector_name in detector_names:
        # Saving face detectors
        if glb.models[detector_name] is not None:
            save_built_model(detector_name, glb.models[detector_name],
                                saved_models_dir, overwrite=overwrite,
                                verbose=verbose)

# ------------------------------------------------------------------------------

def save_built_verifiers(verifier_names, saved_models_dir, overwrite=False,
                         verbose=False):
    """
    Saves built face verifiers with names in 'verifier_names' at the directory
    'saved_models_dir'. If a face verifier with a given name already exists,
    this function does not save (skips) it UNLESS overwrite is True, in which
    case it overwrites it with the new verifier.

    Inputs:
        1. verifier_names   - names of saved face verifiers [string or list of
                               strings].

        2. saved_models_dir - full path of the directory containing the saved
                               face verifier models [string].

        3. overwrite        - toggles between overwriting existing face verifier
                               models [boolean, default=False].

        4. verbose          - toggles if the function should print useful
                               information to the console [boolean,
                               default=False].

    Output:
        1. None

    Signature:
        save_built_verifiers(verifier_names, saved_models_dir, overwrite=False,
                             verbose=False)
    """
    # Ensures that 'verifier_names' is a list
    if not isinstance(verifier_names, list):
        verifier_names = [verifier_names]

    # Loops through each verifier name in 'verifier_names'
    for verifier_name in verifier_names:
        # Saving face verifiers
        if glb.models[verifier_name] is not None:
            save_built_model(verifier_name, glb.models[verifier_name],
                                saved_models_dir, overwrite=overwrite,
                                verbose=verbose)

    return None

# ______________________________________________________________________________
#                   REPRESENTATION DATABASE RELATED FUNCTIONS
# ------------------------------------------------------------------------------

def get_embeddings_as_array(verifier_name):
    """
    Gets all of the embeddings for a given 'verifier_name' from the session
    object stored in the global variable 'sqla_session' in
    'global_variables.py', and returns it as a N x M numpy array where N is the
    number of embeddings and M is the number of elements of each embeddings.

    Inputs:
        1. verifier_name - name of verifier [string].

    Output:
        1. N x M numpy array where each row corresponds to a face image's
            embedding [numpy array].

    Signature:
        embeddings = get_embeddings_as_array(verifier_name)
    """
    # Get all face embeddings
    query = glb.sqla_session.query(FaceRep.embeddings)

    # Loops through each query result, appending the appropriate embedding to
    # the list
    embeddings = [] # initializes empty list
    for row in query.all():
        # print(f'Iter {i}:', type(row[0]), row[0][verifier_name], sep=' | ')
        embeddings.append(row[0][verifier_name])

    return np.array(embeddings)

# ------------------------------------------------------------------------------

def process_faces_from_dir(img_dir, detector_models, verifier_models,
                        detector_name='retinaface', verifier_names=['ArcFace'],
                        normalization='base', align=True, auto_grouping=True, 
                        eps=0.5, min_samples=2, metric='cosine', pct=0.02,
                        check_models=True, verbose=False):
    """
    Processes face images contained in the directory 'img_dir'. If there are no
    images in the directory, an assertion error is raised. The 'processing'
    includes the following steps, performed per image:
        1. Faces are detected in the image using the 'detector_name' face
            detector.

        2. If a detected face (region) is too small, it is discarded. This
            filtering is determined by 'pct'. If a region's area is smaller than
            the original image's area multiplied by this percentage factor
            'pct', then it is discarded. This helps with detection of tiny faces
            which are not useful for recognition.

        3. For each filtered face, the deep neural embeddings (which is just a
            vector representation of the face) is calculated.

        4. A face representation object (see help(FaceRep) for more details) is
            created for each face and added (but not committed!) to the current
            session.

    An optional 'fifth' step is performed if 'auto_grouping' is True. The
    function tries to group similar face representations using the DBSCAN
    algorithm on the embeddings, such that each group corresponds to faces of
    (ideally) the same person. If multiple face verifiers were passed to this
    function, the grouping is performed using the embeddings obtained from the
    FIRST face verifier in the list.

    If 'check_models' is True, then the function ensures that:
        1. the 'detector_name' face detector is in the 'detector_models'
            dictionary.

        2. the 'verifier_names' face verifier is in the 'verifier_models'
            dictionary.

    In both cases, if a detector or verifier is not in the respective
    dictionary, the function attempts to build them from scratch. If the
    building process fails, then an assertion error is raised as either a face
    detector and/or verifier will be missing.

    IMPORTANT: This function uses the 'sqla_session' global variable from the
    'global_variables.py' module to add changes (but not commit) to the SQL
    alchemy session.

    Inputs:
         1. img_dir         - full path to the directory containing the images
                                [string].

         2. detector_models - dictionary of face detector model names (keys) and
                                objects (values) [dictionary].

         3. verifier_models - dictionary of face verifier model names (keys) and
                                objects (values) [dictionary].

         4. detector_name   - chosen face detector's name. Options: opencv, ssd,
                                mtcnn or retinaface [string,
                                default='retinaface'].

         5. verifier_names  - chosen face verifiers' name(s). Options: VGG-Face,
                                OpenFace, Facenet, Facenet512, DeepFace, DeepID
                                and ArcFace. Can be either a string (with a
                                single name) or a list of string (with several
                                names) [string or list of strings,
                                default=['ArcFace']].

         6. normalization   - normalizes the face image and may increase face
                                recognition performance depending on the
                                normalization type and the face verifier model.
                                Options: base, raw, Facenet, Facenet2018,
                                VGGFace, VGGFace2 and ArcFace [string,
                                default='base'].
        
         7. align           - toggles if face images should be aligned. This
                                improves face recognition performance at the
                                cost of some speed [boolean, default=True].

         8. auto_grouping   - toggles whether the faces should be grouped
                                automatically using the DBSCAN algorithm. If
                                multiple verifier names are passed, uses the
                                embeddings of the first verifier during the
                                clustering procedure [boolean, default=True].

         9. eps             - the maximum distance between two samples for one
                                to be considered as in the neighborhood of the
                                other. This is the most important DBSCAN
                                parameter to choose appropriately for the
                                specific data set and distance function
                                [float, default=0.5].

        10. min_samples     - the number of samples (or total weight) in a
                                neighborhood for a point to be considered as a
                                core point. This includes the point itself
                                [integer, min_samples=2].

        11. metric          - the metric used when calculating distance between
                                instances in a feature array. It must be one of
                                the options allowed by
                                sklearn.metrics.pairwise_distances
                                [string, default='cosine'].

        12. pct             - percentage of image area as a decimal. This will
                                be used to filter out 'small' detections [float,
                                default=0.02].

        12. check_models    - toggles if the function should ensure the face 
                                detectors and verifiers are contained in the
                                respective dictionaries [boolean, default=True].
            
        14. verbose         - toggles the function's warnings and other messages
                                [boolean, default=True].

    Output:
        1. returns a list of the FaceRep objects created [list of FaceRep
            objects].

    Signature:
        records = process_faces_from_dir(img_dir, detector_models,
                        verifier_models, detector_name='retinaface',
                        verifier_names=['ArcFace'], normalization='base',
                        align=True, auto_grouping=True, eps=0.5, min_samples=2,
                        metric='cosine', check_models=True, verbose=False)
    """
    # Initializes records (which will be a list of FaceReps)
    records = []

    # Assuming img_dir is a directory containing images
    img_paths = get_image_paths(img_dir)
    img_paths.sort()

    # No images found, do something about it
    if len(img_paths) == 0:
        # Does something about the fact that there are no images in the
        # directory - for now just raise an assertion error
        raise AssertionError('No images in the directory specified')

    # Ensures that the face detector and verifiers exist
    if check_models:
        # Ensures face detectors exist
        ret1, detector_models = ensure_detectors_exists(models=detector_models,
                                                detector_names=[detector_name],
                                                verbose=verbose)

        # Ensures face verifiers exist
        ret2, verifier_models = ensure_verifiers_exists(models=verifier_models,
                                                verifier_names=verifier_names,
                                                verbose=verbose)

        # Asserts that the face detectors and verifiers exist
        assert ret1 and ret2, f'Could not ensure existence of '\
                            + f'face detectors ({ret1}) or verifiers ({ret2})!'

    # If auto grouping is True, then initialize the embeddings list
    if auto_grouping:
        embds = []
    
    # Obtains the processed files names from the ProcessedFiles table to skip
    # already processed files
    proc_fnames = glb.sqla_session.query(ProcessedFiles.filename)
    proc_fnames = [item[0] for item in proc_fnames.all()]

    # Creates the progress bar
    n_imgs = len(img_paths)
    pbar   = tqdm(range(0, n_imgs), desc='Processing face images',
                    disable=False)

    # Loops through each image in the 'img_dir' directory
    for index, i, img_path in zip(pbar, range(0, n_imgs), img_paths):
        # Skips the current file if it has already been processed
        if img_path[img_path.rindex('/')+1:] in proc_fnames:
            if glb.DEBUG:
                print(f'Skipping: {img_path}'.ljust(40), '(already processed)')
            continue

        # Detects faces
        output = do_face_detection(img_path, detector_models=detector_models,
                                    detector_name=detector_name, align=align,
                                    verbose=verbose)

        # Filter regions & faces which are too small
        image_size             = mpimg.imread(img_path).shape
        filtered_regions, idxs = discard_small_regions(output['regions'],
                                                        image_size, pct=pct)
        filtered_faces         = [output['faces'][i] for i in idxs]

        # Calculates the deep neural embeddings for each face image in outputs
        embeddings = calc_embeddings(filtered_faces, verifier_models,
                                     verifier_names=verifier_names,
                                     normalization=normalization)

        # Loops through each (region, embedding) pair and create a record
        # (FaceRep object)
        for region, cur_embds in zip(filtered_regions, embeddings):
            # Create a FaceRep record
            record = FaceRep(image_name_orig=img_path.split('/')[-1],
                        image_name='', image_fp_orig=img_path,
                        image_fp='', group_no=-1, region=region,
                        embeddings=cur_embds)
            
            # Appends each record to the records list
            records.append(record)

            # If auto grouping is True, then store each calculated embedding
            if auto_grouping:
                embds.append(cur_embds[verifier_names[0]])

        # After file has been processed, add it to the ProcessedFiles table
        glb.sqla_session.add(ProcessedFiles(filename=img_path.split('/')[-1],
                                            # filepath=img_path,
                                            filesize=os.path.getsize(img_path)))
    
    if glb.DEBUG:
        print('Commits processed files')
    glb.sqla_session.commit()

    # Loops through each record and add them to the global session
    if glb.DEBUG:
        print('add representation to FaceRep table')
    for record in records:
        glb.sqla_session.add(record)
    glb.sqla_session.commit()

    # Add how many person to Person table as the detected clusters
    if glb.DEBUG:
        print('add person to Person table')
    subquery = select(FaceRep.group_no).where(FaceRep.group_no > -1).group_by(
                    FaceRep.group_no).order_by(FaceRep.group_no)
    query = insert(Person).from_select(["group_no"], subquery)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    # Populate the person_id field in FaceRep with the corresponding ID in Person table
    if glb.DEBUG:
        print('Create joins between Person and FaceRep tables')
    subquery = select(Person.id).where(FaceRep.group_no == 
                Person.group_no).where(FaceRep.group_no > -1)
    query = update(FaceRep).values(person_id =
                subquery.scalar_subquery()).where(FaceRep.group_no > -1)
    if glb.DEBUG:
        print(query)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()
    
    # Set group_no to -2 for the representation that have been linked with person
    if glb.DEBUG:
        print('Set group_no to -2 for FaceRep and Person that',
              'have been already linked together')
    query = update(FaceRep).values(group_no = -2).where(FaceRep.group_no > -1)
    glb.sqla_session.execute(query)
    query = update(Person).values(group_no = -2).where(Person.group_no > -1)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    # Clusters Representations together using the DBSCAN algorithm
    if auto_grouping and len(embds) > 0:
        group_facereps(verifier_names[0], eps=eps, min_samples=min_samples,
                        metric=metric, verbose=verbose)

    # Return representation database
    return records

# ------------------------------------------------------------------------------

def process_image_zip_file(myfile, image_dir, t_check=True, n_token=2,
                            valid_exts=['.jpg', '.png', '.npy']):
    """
    Processes a zip file containing image files. The zip file ('myfile') is
    assumed to have only valid image files (i.e. '.jpg', '.png', etc).
    
    The contents of the zip file are extracted to a named temporary directory.
    Then each file is checked to see if they have already been processed (exists
    with the same file name and size in the ProcessedFiles table of the
    database) OR if they are duplicate files. A file is considered a duplicate
    if there is at least one file in the 'image_dir' directory that:
        
        1. has the same file size (checked via filecmp.cmp(..., shallow=False))
        2. has the same contents (checked via filecmp.cmp(..., shallow=False))
        3. has the same image width and height (checked via imagesize)
    
    An existing file or duplicate file is ignored during the extraction process.
    If 'auto_rename' is True, then each unique file with the same name as a file
    in 'image_dir' directory gets renamed to a unique identifier using uuid4()
    from the uuid library. If, however, 'auto_rename' is False then the file is
    also skipped despite being a unique file.

    Finally, all unique (possibly renamed) files are moved from the temporary
    directory to the 'image_dir' directory, and the temporary directory is
    deleted.

    Effectively, this function attempts to extract only unique (non-existing)
    image files from the zip file provided and rename them if necessary.

    Inputs:
        1. myfile      - zip file obtained through FastAPI [zip file].

        2. image_dir   - path to directory in which the extracted images will be
                            saved to [string].

        3. auto_rename - toggles between automatic renaming of image files with
                            a non-unique name [boolean, default=True].

    Output:
        1. list with the paths of each image file that was skipped [list of
            strings].

    Signature:
        skipped_files = process_image_zip_file(myfile, image_dir,
                                                auto_rename=True)
    """
    # Create temporary directory and extract all files to it
    with TemporaryDirectory(prefix="create_database_from_zip-") as tempdir:
        with ZipFile(BytesIO(myfile.file.read()), 'r') as myzip:
            # Extracts all files in the zip file to a temporary directory,
            # flattens the directory structure and filters the files by valid
            # extensions
            myzip.extractall(path=tempdir)
            flatten_dir_structure(tempdir, valid_exts=valid_exts,
                                    n_token=n_token)

            # Obtains the files' paths, removes corrupted images and creates a
            # new list with only the uncorrupted files
            all_tpaths = [os.path.join(tempdir, file) for file\
                        in os.listdir(tempdir)]
            tpaths     = []
            for pth in all_tpaths:
                if not image_is_uncorrupted(pth, transpose_check=t_check):
                    os.remove(pth)     # deletes corrupted images
                else:
                    tpaths.append(pth) # appends valid path to tpaths

            # Repopulates the 'proc_files_temp' table
            if repopulate_temp_file_table(tpaths):
                raise AssertionError("Could not repopulate"\
                                   + "'proc_files_temp' table.")

            # Queries the database to figure out which files have the SAME size
            query  = select(ProcessedFiles.filename,
                            ProcessedFilesTemp.filename).join(\
                            ProcessedFilesTemp, ProcessedFiles.filesize ==\
                            ProcessedFilesTemp.filesize)
            result = glb.sqla_session.execute(query)

            # Initializes the skipped_files list then loops through each matched
            # & temporary file pairs in the query's result
            skipped_files = []
            for fname, tname in result:
                # Obtains the full path of the matched & temporary files
                fname_fullpath = os.path.join(image_dir, fname)
                tname_fullpath = os.path.join(tempdir, tname)
                
                # Checks if the files are different or not
                if not cmp(fname_fullpath, tname_fullpath, shallow=False):
                    # Files are different, so check if they have the same name
                    if fname == tname:
                        # Names are the same, so rename them
                        tname = rename_file_w_hex_token(tname)
                    
                    # Determines the new full path for tname and moves the file
                    tname_fullpath_dest = os.path.join(image_dir, tname)
                    sh_move(tname_fullpath, tname_fullpath_dest)

                else:
                    # Files are the same, so remove it from tempdir
                    os.remove(tname_fullpath)
                    skipped_files.append(tname)

            # Queries for files that have SAME name and DIFFERENT size from the
            # existing ones that have to be renamed
            query = select(ProcessedFilesTemp.filename).join(ProcessedFiles,
                    (ProcessedFilesTemp.filename == ProcessedFiles.filename)\
                    & (ProcessedFilesTemp.filesize != ProcessedFiles.filesize))
            result = glb.sqla_session.execute(query)

            # Loops through each row in result
            for row in result:
                # Obtains the file name and renames it
                filename         = row.filename
                filename_renamed = rename_file_w_hex_token(filename)

                # Moves the file to the appropriate location
                sh_move(os.path.join(tempdir, filename),
                        os.path.join(tempdir, filename_renamed))

            # Now it's safe to move the remaining files in tempdir directly to
            # img_dir
            for file in os.listdir(tempdir):
                # Moves each file to the appropriate location (ensuring it is
                # not a directory)
                if not os.path.isdir(os.path.join(tempdir, file)):
                    sh_move(os.path.join(tempdir,file),
                            os.path.join(image_dir, file))

    return skipped_files

# ______________________________________________________________________________
#                               OTHER FUNCTIONS
# ------------------------------------------------------------------------------

def get_matches_from_similarity(similarity_obj):
    """
    Gets all matches from the session object stored in the global variable
    'sqla_session' from global_variable.py based on the current similairty
    object 'similairty_obj'. This object is obtained from 'calc_similarity()'
    function (see help(calc_similarity) for more information).

    Inputs:
        1. similarity_obj - dictionary containing the indexes of matches (key:
            idxs), the threshold value used (key: threshold) and the distances
            of the matches (key: distances) [dictionary].

    Output:
        1. list containing each matched as a VerificationMatch object. Note
            that the length of this list is equal to the number of matches (so
            an empty list means no matches).

    Signature:
        matches = get_matches_from_similarity(similarity_obj)
    """
    # Initializes the matches list
    matches = []

    # Get all face embeddings
    query = glb.sqla_session.query(FaceRep)

    # Loops through each index in the similarity object and stores the
    # corresponding Representation
    for i, dst in zip(similarity_obj['idxs'], similarity_obj['distances']):
        rep = query.all()[i]
        matches.append(VerificationMatch(unique_id=rep.id,
                            image_name=rep.image_name_orig,
                            person_id=rep.person_id,
                            image_fp=rep.image_fp_orig,
                            region=rep.region,
                            distance=dst,
                            embeddings=[name for name in rep.embeddings.keys()],
                            threshold=similarity_obj['threshold']))
    
    return matches

# ------------------------------------------------------------------------------

def group_facereps(verifier_name, eps=0.5, min_samples=2, metric='cosine',
                    verbose=False):
    """
    Groups face representations. Each group is considered to be a unique person.
    This functions raises an assertion error if it fails to get the embeddings
    corresponding to the face verifier 'verifier_name' from the database.

    Note that this functions uses the global session (glb.sqla_session) to read
    and modify tables in the database.

    Inputs:
        1. verifier_name - face verifier name [string].
        
        2. eps           - the maximum distance between two samples for one to
                           be considered as in the neighborhood of the other.
                           This is the most important DBSCAN parameter to
                           choose appropriately for the specific data set and
                           distance function [float, default=0.5].

        3. min_samples   - the number of samples (or total weight) in a
                           neighborhood for a point to be considered as a core
                           point. This includes the point itself [integer,
                           min_samples=2].

        4. metric        - the metric used when calculating distance between
                           instances in a feature array. It must be one of the
                           options allowed by sklearn.metrics.pairwise_distances
                           [string, default='cosine'].

        5. verbose       - controls if the function should output information to
                           the console [boolean, default=False]

    Output:
        1. None

    Signature:
        group_facereps(verifier_name, eps=0.5, min_samples=2, metric='cosine')
    """
    # Attempts to get all embeddings stored in the database, given the chosen
    # verifier model's name
    try:
        embds = get_embeddings_as_array(verifier_name)
    except Exception as excpt:
        raise AssertionError(f'Could not get embeddings (reason: {excpt})')

    # Clusters embeddings using DBSCAN algorithm
    result = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(embds)

    # Deletes 'temp_clustering' table and repopulates it
    glb.sqla_session.execute(delete(tempClustering))
    glb.sqla_session.commit()

    group_no = [{'group_no': int(no)} for no in result.labels_]
    query    = insert(tempClustering).values(group_no)
    
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    # Updates the group_no in FaceRep with the values taken from TempClustering
    # table
    query = text("UPDATE representation "                                +\
                 "SET group_no = (SELECT group_no FROM temp_clustering " +\
                 "WHERE representation.id = temp_clustering.id)")
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    # Gets the list of ids of the new clusters
    query  = select(tempClustering.group_no).group_by(tempClustering.group_no)
    result = glb.sqla_session.execute(query)
    new_clusters = [item.group_no for item in result.all()]

    # Gets the person id for each cluster
    for cluster in new_clusters:
        query  = select(FaceRep.person_id).where((FaceRep.group_no == cluster)\
                        & (FaceRep.person_id != None)).limit(1)
        result = glb.sqla_session.execute(query).first()

        # Checks if a match exists for 'person_id' in this cluster
        if result:
            # Obtains the person id and prints the cluster-person association if
            # verbose = True
            person_id = result.person_id
            if verbose:
                print(f"Cluster {cluster} associated with person {person_id}")

            # Sets the 'person_id' field for all the records of this cluster
            query = update(FaceRep).values(person_id = person_id).where(
                    (FaceRep.group_no == cluster) & (FaceRep.person_id == None))
            glb.sqla_session.execute(query)

            # Sets the 'group_no' for all records in this cluster to -2, meaning
            # that they have been done
            query = update(FaceRep).values(group_no = -2).where(
                        FaceRep.group_no == cluster)
            glb.sqla_session.execute(query)
            glb.sqla_session.commit()

        # Otherwise, the cluster is not associated with any person and new
        # person needs to be created
        else:
            if verbose:
                print(f"Cluster {cluster} is not associated with any person.")
                print("A new person will be created and associated with all",
                      "the relevant representations")

            # Inserts a new record (person) into the Person table
            query  = text("INSERT INTO person(name, group_no) VALUES(Null, -2)")
            result = glb.sqla_session.execute(query)
            glb.sqla_session.commit()

            # Gets the id of the new inserted record / person
            person_id = result.lastrowid

            # Updates the 'person_id' value for all the records in
            # Representation for this cluster
            query = text("UPDATE representation SET person_id = "           +\
                        str(person_id) + ", group_no = -2 WHERE "           +\
                        "(representation.group_no == " + str(cluster)       +\
                        ") & ((SELECT rep2.id FROM representation AS rep2 " +\
                        "WHERE (rep2.group_no == " + str(cluster)           +\
                        ") & (rep2.person_id IS NOT NULL)) IS NULL)")
            glb.sqla_session.execute(query)
            glb.sqla_session.commit()

    return None

# ______________________________________________________________________________
#                           DATABASE RELATED FUNCTIONS 
# ------------------------------------------------------------------------------

def database_exists(db_full_path):
    """
    Checks if a database file exists in the path provided 'db_full_path',
    returning True if so and False otherwise. Remember to include the file's
    extension in the full path along with its name.

    Input:
        1. db_full_path - full path to the database file [string].

    Output:
        1. boolean indicating if the database file exists [boolean].
        
    Signature:
        ret = database_exists(db_full_path)
    """
    return True if os.path.isfile(db_full_path) else False

# ------------------------------------------------------------------------------

def database_is_empty(engine):
    """
    Checks if the database is empty, returning True if so and False otherwise.
    This function handles exceptions (e.g. if engine == None), returning False
    in these cases (as they are not a database).

    Input:
        1. engine - engine object (see help(load_database) for more information
                    on what this object is) [engine object].

    Output:
        1. boolean indicating if the database provided is empty or not
            [boolean]. 

    Signature:
        ret = database_is_empty(engine)
    """
    try:
        ret = inspect(engine).get_table_names() == []
    except:
        ret = False

    return ret

# ------------------------------------------------------------------------------

def all_tables_exist(engine, check_names):
    """
    Checks if the database accessed by the engine object 'engine' contains all
    table names provided in the list 'check_names', returning True if that is
    the case and False otherwise.

    Note: this function ensures that 'check_names' is a list so it can handle
    cases where 'check_names' is just a single string.

    Inputs:
        1. engine      - engine object (see help(load_database) for more
                            information on what this object is) [engine object].

        2. check_names - table names [string or list of strings].

    Output:
        1. boolean indicating if all tables exist in the database provided
            [boolean].
        
    Signature:
        ret = all_tables_exist(engine, check_names)
    """
    # Ensures 'check_names' is a list
    if not isinstance(check_names, list):
        check_names = [check_names]

    # Initializes the return value 'ret' as True, then loops over each name in
    # 'check_names' list
    ret = True
    for name in check_names:
        ret = ret and inspect(engine).has_table(name)

    return ret

# ------------------------------------------------------------------------------

def load_database(relative_path, create_new=True, force_create=False):
    """
    Loads a SQLite database specified by its full path 'db_full_path'.

    Inputs:
        1. relative_path - full path to the database file [string].

        2. create_new    - toggles if a new database should be created IF one
                             could not be loaded from the full path provided
                             [boolean, default=True].

        3. force_create  - toggles if a new database should be created REGARDLESS
                             of a database existing in the full path provided
                             (this effectively disregards 'db_full_path')
                             [boolean, default=False].

    Output:
        1. returns a engine object (if successful) or a None object
            [engine object].

        2. returns a base object (if successful) or a None object [base object].

    Signature:
        engine, base = load_database(db_full_path, create_new=True,
                                     force_create=False)
    """
    db_full_path = os.path.join(glb.API_DIR   , relative_path)
    engine       = create_engine("sqlite:///" + relative_path)

    # If database exists, opens it
    if   database_exists(db_full_path) and not force_create:
        # Binds the metadata to the engine
        metadata_obj = MetaData()
        metadata_obj.reflect(bind=engine)

    # If 'create_new' or 'force_create' are True, creates a new database
    elif create_new or force_create:
        # create the SQLAlchemy tables' definitions
        Base.metadata.create_all(engine)

    else:
        # Otherwise, returns a None object with a warning
        print('[load_database] WARNING: Database loading failed',
              '(and a new was NOT created).')
        engine = None

    return engine

# ------------------------------------------------------------------------------

def start_session(engine):
    """
    Creates a Session object from the database connected by the engine object
    'engine'.
    
    The primary usage interface for persistence operations is the Session
    object, from now on referred to as simply 'session'. The session establishes
    all conversations with the database and represents a “holding zone” for all
    the objects which have been loaded or associated with it during its
    lifespan.
    
    It provides the interface where SELECT and other queries are made that will
    return and modify ORM-mapped objects. The ORM objects themselves are
    maintained inside the session, inside a structure called the identity map -
    a data structure that maintains unique copies of each object, where “unique”
    means “only one object with a particular primary key”.

    Input:
        1. engine - engine object (see help(load_database) for more
                      information on what this object is) [engine object].

    Output:
        1. the session as a Session object (if successful) or None object
            (on failure) [session object or None].

    Signature:
        session = start_session(engine)
    """
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
    except:
        session = None

    return session

# ------------------------------------------------------------------------------

def facerep_set_groupno_done(session, face_id):
    """
    TODO: Update documentation
    """
    # Loops through each file
    query = update(FaceRep).values(group_no = -1, person_id=None).where(FaceRep.id == face_id)
    if session:
        session.execute(query)
        session.commit()

# ------------------------------------------------------------------------------

def people_clean_without_repps(session):
    """
    Remove records from Person table that don't have any corresponding records in
    FaceRep table.
    """

    textual_sql = text("DELETE FROM person WHERE person.id IN (SELECT person.id \
                        FROM person LEFT JOIN representation \
                        ON person.id == representation.person_id \
                        GROUP BY person.id \
                        HAVING COUNT(representation.id) == 0)")
    if(session):
        session.execute(textual_sql)
        session.commit()

# ------------------------------------------------------------------------------

def repopulate_temp_file_table(tpaths):
    """
    TODO: Documentation
    """

    # First, tries to clear everything in the 'proc_files_temp' table
    try:
        stmt = delete(ProcessedFilesTemp)
        glb.sqla_session.execute(stmt)
        glb.sqla_session.commit()
    except Exception as excpt:
        glb.sqla_session.rollback()
        print("Error when clearing 'proc_files_temp' table",
             f'(reason: {excpt})')
        return True

    if glb.DEBUG:
        print("Populating 'proc_files_temp' table")

    # Loops through each temporary path in 'tpaths'
    for tpath in tpaths:
        # Adds each file name and size to the 'proc_files_temp' table
        glb.sqla_session.add(ProcessedFilesTemp(
                                        filename=tpath[tpath.rindex('/')+1:],
                                        filesize=os.path.getsize(tpath))
                            )

    # Commits the changes
    if glb.DEBUG:
        print('Committing newly added temporary files')
    glb.sqla_session.commit()

    return False

# ------------------------------------------------------------------------------


