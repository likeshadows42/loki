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
from uuid                    import uuid4, UUID
from zipfile                 import ZipFile
from tempfile                import TemporaryDirectory
from IFR.classes             import RepDatabase, Representation,\
                                    VerificationMatch
from IFR.functions           import get_image_paths, do_face_detection,\
                                    calc_embeddings
from sklearn.cluster         import DBSCAN

from shutil                  import move             as sh_move


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
            db = RepDatabase()

    # If path does not point to a file, open an 'empty' database
    else:
        print(f'failed! Reason: database does not exist.')
        db = RepDatabase()

    return db

# ______________________________________________________________________________
#                   REPRESENTATION DATABASE RELATED FUNCTIONS
# ------------------------------------------------------------------------------

def create_new_rep(original_fp, img_fp, region, embeddings, uid='',
                   group_no=-1, tag='', ignore_taglist=['--', '---']):
    """
    Creates a new representation object. For more information see
    help(Representation).

    Inputs:
        1. original_fp    - original image's full path [string].

        2. img_fp         - image's full path [string].

        3. region         - face region on the original image [list of 4
            integers].

        4. embeddings     - face verifier name (key) and embedding
            [1-D numpy array] (item). Can have multiple verifier, embedding
            pairs (key, value pairs) [dictionary].

        5. uid            - valid unique object identifier. If left empty ('') a
            unique object identifier is created using uuid4 from the uuid
            library [string, default=''].
        
            Note: a valid uid string representation obeys the following (case
            insensitive) regular expression:
                        '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
                        '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

        6. group_no       - group number. If group_no == -1 then this means
            'no group' / 'groupless' [integer, default=-1].

        7. tag            - name tag or nickname [string, default=''].

        8. ignore_taglist - list of strings that are treated as equivalent to ''
            (i.e. no tag) [list of strings, default=['--', '---']].
    
    Output:
        1. Representation object

    Signature:
        new_rep = create_new_rep(original_fp, img_fp, region, embeddings,
                    uid='', group_no=-1, tag='', ignore_taglist=['--', '---'])
    """

    # If Unique IDentifier (UID) is not provided, generate one
    if len(uid) == 0 or uid == '':
        uid = uuid4()

    # If tag is in the ignore taglist, then it is considered as a "ignore" tag
    if tag in ignore_taglist:
        tag = ''

    # If group number is less than -1 (i.e. invalid) set it to -1 (i.e. no
    # group)
    if group_no < -1:
        group_no = -1

    # Returns the new representation
    return Representation(uid, orig_name=original_fp.split('/')[-1],
                          image_name=img_fp.split('/')[-1], orig_fp=original_fp,
                          image_fp=img_fp, group_no=group_no, name_tag=tag,
                          region=region, embeddings=embeddings)

# ------------------------------------------------------------------------------

def get_embeddings_as_array(db, verifier_name):
    """
    Gets all of the embeddings for a given 'verifier_name' from the database
    'db' and returns it as a N x M numpy array where N is the number of
    embeddings and M is the number of elements of each embeddings.

    Inputs:
        1. db            - Representation database [RepDatabase object].
        2. verifier_name - name of verifier [string].

    Output:
        1. N x M numpy array where each row corresponds to a face image's
            embedding

    Signature:
        embeddings = get_embeddings_as_array(db, verifier_name)
    """
    embeddings = [] # initializes empty list
    for rep in db.reps:
        embeddings.append(rep.embeddings[verifier_name])

    return np.array(embeddings)

# ------------------------------------------------------------------------------

def create_reps_from_dir(img_dir, detector_models, verifier_models,
            detector_name='retinaface', align=True, verifier_names=['ArcFace'],
            show_prog_bar=True, normalization='base', tags=[], uids=[],
            auto_grouping=True, eps=0.5, min_samples=2, metric='cosine',
            verbose=False):
    """
    Creates a representations from images in a directory 'img_dir'. The
    representations are returned in a list, and the list of representations. If
    tags and/or unique identifiers (uids) are provided, make sure that they 
    correspond to the sorted (ascending) image names contained in 'img_dir'.

    Inputs:
        1. img_dir - full path to the directory containing the images [string].

        2. verifier_models - dictionary of model names (keys) and model objects
             (values) [dictionary].

        3. detector_name - chosen face detector's name. Options: opencv, ssd,
             dlib, mtcnn, retinaface (default=retinaface) [string].
        
        4. align - toggles if face images should be aligned. This improves face
             recognition performance at the cost of some speed (default=True)
             [boolean].

        5. verifier_names - chosen face verifier's name. Options: VGG-Face,
             OpenFace, Facenet, Facenet512, DeepFace, DeepID and ArcFace
             (default=ArcFace) [string].

        6. show_prog_bar - toggles the progress bar on or off (default=True)
             [boolean].

        7. normalization - normalizes the face image and may increase face
             recognition performance depending on the normalization type and the
             face verifier model. Options: base, raw, Facenet, Facenet2018,
             VGGFace, VGGFace2 and ArcFace (default='base') [string].

        8. tags - list of strings where each string corresponds to a tag for the
             i-th image, i.e. tags[0] is the tag for the first image in the
             sorted list of image names obtained from 'img_dir' directory. If an
             empty list is provided, this is skipped during the Representation
             creation process (default='') [string or list of strings].

        9. uids - list of strings where each string corresponds to a unique
             identifier (UID) for the i-th image, i.e. uids[0] is the UID for
             the first image in the sorted list of image name obtain from
             'img_dir' directory. If an empty list is provided, a UID is created
             for each image during the representation creation process
             (default='') [string or list of strings].

        10. auto_grouping - toggles whether Representations should be grouped /
             clusted automatically using the DBSCAN algorithm. If multiple
             verifier names are passed, uses the embeddings of the first
             verifier during the clustering procedure (default=True) [boolean].

        11. eps - the maximum distance between two samples for one to be
             considered as in the neighborhood of the other. This is the most
             important DBSCAN parameter to choose appropriately for the
             specific data set and distance function (default=0.5) [float].

        12. min_samples - the number of samples (or total weight) in a
             neighborhood for a point to be considered as a core point. This
             includes the point itself (min_samples=2) [integer].

        13. metric - the metric used when calculating distance between instances
              in a feature array. It must be one of the options allowed by
              sklearn.metrics.pairwise_distances (default='cosine') [string].
            
        14. verbose - toggles the function's warnings and other messages
            (default=True) [boolean].

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
    rep_list = []
    
    # Assuming img_dir is a directory containing images
    img_paths = get_image_paths(img_dir)
    img_paths.sort()

    # No images found, return empty database
    if len(img_paths) == 0:
        return RepDatabase()

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

    # If auto grouping is True, then initialize the embeddings list
    if auto_grouping:
        embds = []

    # Loops through each image in the 'img_dir' directory
    for pb_idx, i, img_path in zip(pbar, range(0, n_imgs), img_paths):
        # Detects faces
        output = do_face_detection(img_path, detector_models=detector_models,
                                    detector_name=detector_name, align=align,
                                    verbose=verbose)

        # Assumes only 1 face is detect (even if multiple are present). If no
        # face are detected, skips this image file
        if output is not None:
            face   = [output['faces'][0]]
            region = output['regions'][0]
        else:
            continue

        # Calculate the face image embedding
        embeddings = calc_embeddings(face, verifier_models,
                                     verifier_names=verifier_names,
                                     normalization=normalization)
        embeddings = embeddings[0]

        # If auto grouping is True, then store each calculated embedding
        if auto_grouping:
            embds.append(embeddings[verifier_names[0]])

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
        rep_list.append(create_new_rep(img_path, img_path, region, embeddings,
                                        tag=tag, uid=uid))

    # Clusters Representations together using the DBSCAN algorithm
    if auto_grouping:
        # Clusters embeddings using DBSCAN algorithm
        results = DBSCAN(eps=eps, min_samples=min_samples,
                         metric=metric).fit(embds)

        # Loops through each label and updates the 'group_no' attribute of each
        # Representation IF group_no != -1 (because -1 is already the default
        # value and means "no group")
        for i, lbl in enumerate(results.labels_):
            if lbl == -1:
                continue
            else:
                rep_list[i].group_no = lbl

    # Return representation database
    return RepDatabase(*rep_list)

# ------------------------------------------------------------------------------

def process_image_zip_file(myfile, image_dir, auto_rename=True):
    """
    Processes a zip file containing image files. The zip file ('myfile') is
    assumed to have only valid image files (i.e. '.jpg', '.png', etc).
    
    First, the contents of the zip file are extracted to a named temporary
    directory. All file names in the save directory 'image_dir' is obtained. If
    'auto_rename' is True, then each image with the same name as a file in
    'image_dir' directory gets renamed. Each image is then moved from the
    temporary directory to the 'image_dir' directory, and the temporary
    directory is deleted. If 'auto_rename' is False then images with the same
    name are simply ignored.

    Inputs:
        1. myfile      - zip file obtained through FastAPI [zip file].

        2. image_dir   - path to directory in which the extracted images will be
                            saved to [string].

        3. auto_rename - toggles between automatic renaming of image files with
                            a non-unique name [boolean, default=True].

    Output:
        1. list with the names of each image file saved [list of strings].

    Signature:
        new_names = process_image_zip_file(myfile, image_dir, auto_rename=True)
    """
    # Create temporary directory and extract all files to it
    with TemporaryDirectory(prefix="create_database_from_zip-") as tempdir:
        with ZipFile(BytesIO(myfile.file.read()), 'r') as myzip:
            # Extracts all files in the zip folder
            myzip.extractall(tempdir)
            
            # Obtais all file names and temporary file names
            all_fnames = [name.split('/')[-1] for name in os.listdir(image_dir)]
            all_tnames = [name.split('/')[-1] for name in os.listdir(tempdir)]

            # Initializes new names list
            new_names = []

            # Loops through each temporary file name
            for tname in all_tnames:
                # If they match any of the current file names, rename them using
                # a unique id if 'auto_rename' is True. If 'auto_rename' is 
                # False (and file requires renaming) skip this file.
                if tname in all_fnames:
                    if auto_rename:
                        uid        = uuid4().hex
                        new_name   = uid + '.' + tname.split('.')[-1] # uid.extension
                        new_names.append(uid[0:8]   + '-' +\
                                         uid[8:12]  + '-' +\
                                         uid[12:16] + '-' +\
                                         uid[16::])
                    else:
                        continue

                # Otherwise, dont rename it
                else:
                    new_name = tname

                # Move (and rename if needed) file to appropriate directory
                new_fp = os.path.join(image_dir, new_name)
                old_fp = os.path.join(tempdir, tname)
                sh_move(old_fp, new_fp)

    return new_names

# ------------------------------------------------------------------------------

def fix_uid_of_renamed_imgs(new_names):
    """
    Ensures that the unique identifier of the renamed images matches the name of
    the image (names of renamed images are valid unique identifiers). This
    function modifies entries in the database contained in the 'rep_db' global
    variable (for more information see global_variables.py).

    Input:
        1. new_names - new names (of renamed files) [list of strings].

    Output:
        1. None

    Signature:
        fix_uid_of_renamed_imgs(new_names)
    """
    # Loops through each representation
    for i, rep in enumerate(glb.rep_db.reps):
        # Determines the current image name and, if it is one of the
        # files that was renamed, use its name (which is a unique id) as
        # its unique id
        cur_img_name = rep.image_name.split('.')[0]
        cur_img_name = cur_img_name[0:8]   + '-' +\
                       cur_img_name[8:12]  + '-' +\
                       cur_img_name[12:16] + '-' +\
                       cur_img_name[16::]
                
        if cur_img_name in new_names:
            glb.rep_db.reps[i].unique_id = UUID(cur_img_name)
            glb.db_changed               = True

    return None

# ______________________________________________________________________________
#                               OTHER FUNCTIONS
# ------------------------------------------------------------------------------

def get_matches_from_similarity(similarity_obj, db):
    """
    Gets all matches from the database 'db' based on the current similairty
    object 'similairty_obj'. This object is obtained from 'calc_similarity()'
    function (see help(calc_similarity) for more information).

    Inputs:
        1. similarity_obj - dictionary containing the indexes of matches (key:
            idxs), the threshold value used (key: threshold) and the distances
            of the matches (key: distances) [dictionary].
        
        2. db             - representation database [RepDatabase object].

    Output:
        1. List containing each matched Representation. Note that the length of
            this list is equal to the number of matches.

    Signature:
        match_obj = get_matches_from_similarity(similarity_obj, db)
    """

    # Initializes the matches list
    matches = []

    # Loops through each index in the similarity object and stores the
    # corresponding Representation
    for i, dst in zip(similarity_obj['idxs'], similarity_obj['distances']):
        rep = db.reps[i]
        matches.append(VerificationMatch(unique_id=rep.unique_id,
                            image_name=rep.image_name, group_no=rep.group_no,
                            name_tag=rep.name_tag, image_fp=rep.image_fp,
                            region=rep.region, distance=dst,
                            embeddings=[name for name in rep.embeddings.keys()],
                            threshold=similarity_obj['threshold']))
    
    return matches

# ------------------------------------------------------------------------------

