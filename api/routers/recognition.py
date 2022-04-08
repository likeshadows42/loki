# ==============================================================================
#                        RECOGNITION-RELATED API METHODS
# ==============================================================================
import os
import cv2
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy                  as np
import matplotlib.pyplot      as plt
import api.global_variables   as glb

from io                       import BytesIO
from uuid                     import UUID, uuid4
from pydoc                    import describe
from pandas                   import describe_option
from typing                   import List, Optional
from zipfile                  import ZipFile
from fastapi                  import APIRouter, UploadFile, Depends, Query
from tempfile                 import mkdtemp
from IFR.classes              import *
from IFR.functions            import create_reps_from_dir, calc_embedding,\
                        get_embeddings_as_array, calc_similarity, create_dir,\
                        load_representation_db, load_face_verifier,\
                        create_new_representation, get_matches_from_similarity,\
                        string_is_valid_uuid4, show_cluster_results,\
                        get_property_from_database    as get_prop_from_db
from fastapi.responses        import Response

from shutil                   import rmtree, move     as sh_move
from matplotlib               import image            as mpimg
from deepface.DeepFace        import build_model      as build_verifier

data_dir = glb.DATA_DIR
img_dir  = glb.IMG_DIR
rdb_dir  = glb.RDB_DIR

# ______________________________________________________________________________
#                             ROUTER INITIALIZATION
# ------------------------------------------------------------------------------

fr_router                    = APIRouter()
fr_router.face_verifier      = None
fr_router.face_verifier_name = None

# ______________________________________________________________________________
#                                  API METHODS
# ------------------------------------------------------------------------------

@fr_router.get("/debug/inspect_globals")
async def inspect_globals():
    """
    API endpoint: inspect_globals.

    Inspects the values of all global variables, printing them to the server's
    console and returning the following dictionary (key:value):
        directories: list of all directory paths
        dir_names  : list with names of each directory path (i.e. directories[0]
                        is the path for dir_names[0])
        models     : list of names of all loaded face verifiers
        num_reps   : number of Representations in the currently loaded database
        db_changed : boolean flag indicating if the database has been modified
                        (and will be saved on shutdown)

    Parameters:
        None

    Output:\n
        JSON-encoded dictionary with the following key/value pairs is returned:
            1. directories: list of directories' full paths
            2. dir_names  : list of directories' names
            3. models     : list of loaded face verifier models
            4. num_reps   : number of Representations in the currently loaded
                            database
            5. db_changed : flag indicating if the database has been modified
                            (and will be saved on shutdown)
    """

    # Printing all path variables (name and full path)
    print('[inspect_globals] Path variables:')
    directory_list = [glb.API_DIR, glb.DATA_DIR   , glb.IMG_DIR,
                      glb.RDB_DIR, glb.SVD_MDL_DIR, glb.SVD_VRF_DIR]
    for name, fp in zip(glb.directory_list_names, directory_list):
        print(f'   -> Directory {name}'.ljust(30), f': {fp}', sep='')
    print('')

    # Printing all loaded face verifier models
    print('[inspect_globals] models:')
    for key, value in glb.models.items():
        print(f'   -> {key}'.ljust(16) + f': {value}')
    print('')

    # Printing the number of Representations in the database
    print('[inspect_globals] database:')
    print('  - there are {} representations'.format(len(glb.rep_db)))
    for i, rep in zip(range(len(glb.rep_db)), glb.rep_db):
        print(f' Entry: {i}  |  ', end='')
        rep.show_summary()
    print('')

    # Printing the 'database has been modified' flag
    print(f'[inspect_globals] database change status: {glb.db_changed}\n')

    return {'directories': directory_list,
            'dir_names'  : glb.directory_list_names,
            'models'     : list(glb.models.keys()),
            'num_reps'   : len(glb.rep_db),
            'db_changed' : glb.db_changed}

# ------------------------------------------------------------------------------

@fr_router.post("/debug/default_values_change")
async def verify_with_upload(value1: Optional[int] = Query(10),
                             value2: Optional[int] = Query(-33)):
    return {'value1':value1, 'value2':value2}

# ------------------------------------------------------------------------------

@fr_router.post("/debug/reset_server")
async def reset_server(no_database: bool = Query(False, description="Toggles if database should be empty or loaded [boolean]")):
    """
    API endpoint: reset_server()

    Allows the user to restart the server without manually requiring to shut it
    down and start it up.

    Parameter:
        1. no_database - toggles if the database should be empty or loaded on
             server reset (default=False) [boolean]

    Output:\n
        JSON-encoded dictionary with the following attributes:
            1. message: message stating the server has been restarted (string)
    """
    print('\n!!!!!!!!!!!!!!!! RESTARTING THE SERVER !!!!!!!!!!!!!!!!')
    print('======== Starting initialization process ======== \n')

    # Directories & paths initialization
    print('  -> Resetting global variables:')
    API_DIR     = os.path.join(os.path.dirname(os.path.realpath("__file__")),
                                'api')
    print('[PATH]'.ljust(12), 'API_DIR'.ljust(12), f': reset! ({API_DIR})')
    DATA_DIR    = os.path.join(API_DIR     , 'data')
    print('[PATH]'.ljust(12), 'DATA_DIR'.ljust(12), f': reset! ({DATA_DIR})')
    IMG_DIR     = os.path.join(DATA_DIR    , 'img')
    print('[PATH]'.ljust(12), 'IMG_DIR'.ljust(12), f': reset! ({IMG_DIR})')
    RDB_DIR     = os.path.join(DATA_DIR    , 'database')
    print('[PATH]'.ljust(12), 'RDB_DIR'.ljust(12), f': reset! ({RDB_DIR})')
    SVD_MDL_DIR = os.path.join(API_DIR     , 'saved_models')
    print('[PATH]'.ljust(12), 'SVD_MDL_DIR'.ljust(12), f': reset! ({SVD_MDL_DIR})')
    SVD_VRF_DIR = os.path.join(SVD_MDL_DIR , 'verifiers')
    print('[PATH]'.ljust(12), 'SVD_VRF_DIR'.ljust(12), f': reset! ({SVD_VRF_DIR})')
    print('')

    models      = {}    # stores all face verifier models
    print('[MODELS]'.ljust(12), 'models'.ljust(12), f': reset! ({models})')
    rep_db      = []    # stores representation database
    print('[DATABASE]'.ljust(12), 'rep_db'.ljust(12), f': reset! ({rep_db})')
    db_changed  = False # indicates whether database has been modified (and should be saved)
    print('[FLAG]'.ljust(12), 'db_changed'.ljust(12), f': reset! ({db_changed})')
    print('')

    # Directories & paths initialization
    print('  -> Directory creation:')
    directory_list = [glb.API_DIR, glb.DATA_DIR, glb.IMG_DIR, glb.RDB_DIR,
                      glb.SVD_MDL_DIR, glb.SVD_VRF_DIR]
    
    for directory in directory_list:
        print(f'Creating {directory} directory: ', end='')
    
        if create_dir(directory):
            print('directory exists. Continuing...')
        else:
            print('success.')
    print('')

    # Tries to load a database if it exists and no_database == False. If not,
    # either creates a new, empty database.
    print('  -> Loading / creating database:')
    if no_database:
        print('Empty database loaded (no_database=True)')
        glb.rep_db = []
    else:
        if not os.path.isfile(os.path.join(glb.RDB_DIR, 'rep_database.pickle')):
            glb.db_changed = True
        glb.rep_db = load_representation_db(os.path.join(glb.RDB_DIR,
                                        'rep_database.pickle'), verbose=True)
    
    print('')
    
    # Loads (or creates) all face verifiers
    print('  -> Loading / creating face verifiers:')
    for verifier_name in glb.verifier_names:
        # Quick fix to avoid problems when len(glb.verifier_names) == 1. In this
        # case, FOR loops over each letter in the string instead of considering
        # the entire string as one thing.
        if verifier_name == '':
            continue

        # First, try loading (opening) the model
        model = load_face_verifier(verifier_name + '.pickle', glb.SVD_VRF_DIR,
                                   verbose=True)

        # If successful, save the model in a dictionary
        if not isinstance(model, list):
            glb.models[verifier_name] = model

        # Otherwise, build the model from scratch
        else:
            print(f'[build_verifier] Building {verifier_name}: ', end='')
            try:
                glb.models[verifier_name] = build_verifier(verifier_name)
                print('success!\n')

            except Exception as excpt:
                print(f'failed! Reason: {excpt}\n')

    print('\n -------- End of initialization process -------- \n')
    return {"message": "Server has been restarted"}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/edit_default_directories")
async def edit_default_directories(img_dir: str = Query(glb.IMG_DIR, description="Full path to image directory (string)"),
                                   rdb_dir: str = Query(glb.RDB_DIR, description="Full path to Representation database directory (string)")):
    """
    API endpoint: edit_default_directories()

    Allows the user to edit the image and Representation database directories.
    These are determined automatically on server startup. These edit will NOT be
    persistent across server restarts.

    Parameters:
    - img_dir: full path to image directory (string)

    - rdb_dir: full path to Representation database directory (string)

    Output:\n
        JSON-encoded dictionary with the following key/value pairs is returned:
            1. status: boolean flag indicating if something went wrong
                       (status=True) or if everything went ok (status=False)
            2. message: informative message stating if the function executed
                        without errors or if there were errors
    """
    # Intitializes output message & status flag
    output_msg = ''
    status     = False
    
    # Sets the IMG_DIR path to the one provided IF it is a valid directory
    if os.path.isdir(img_dir):
        glb.IMG_DIR = img_dir
        output_msg += f'IMG_DIR set to {img_dir}\n'
    else:
        output_msg += f'Path provided is not a valid directory ({img_dir})\n'
        status      = True

    # Sets the RDB_DIR path to the one provided IF it is a valid directory
    if os.path.isdir(rdb_dir):
        glb.RDB_DIR = rdb_dir
        output_msg += f'RDB_DIR set to {rdb_dir}\n'
    else:
        output_msg += f'Path provided is not a valid directory ({rdb_dir})\n'
        status      = True

    return {'status':status, 'message':output_msg}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/get_property_from_database")
async def get_property_from_database(
    propty        : AvailableRepProperties = Query(default_property, description="Representations' property (string)"),
    do_sort       : bool                   = Query(False, description="Flag to perform sorting of the results (boolean)"),
    suppress_error: bool                   = Query(True , description="Flag to suppress error, returning an empty list instead of raising an exception (boolean")):
    """
    API endpoint: get_property_from_database()

    Gets a specific property from all Representations in the database. In the
    specific case of the 'name_tag' property, only unique name tags are
    returned.

    Parameters:
    - propty: Representations' property (string)

    - do_sort: flag to perform sorting of the results (boolean, default: False)

    - suppress_error: flag to suppress error, returning an empty list instead
                        of raising an exception (boolean, default: True)

    Output:\n
        JSON-encoded list containing the chosen property from each
        Representation. The list is sorted if 'do_sort' is set to True. The list
        will be empty if the Representation database has a length of zero or if
        'suppress_error' is True and a non-existant property 'param' is chosen.
    """
    return get_prop_from_db(glb.rep_db, propty, do_sort=do_sort,
                            suppress_error=suppress_error)  

# ------------------------------------------------------------------------------

@fr_router.post("/utility/view_database")
async def view_database(
    amt_detail : MessageDetailOptions = Query(default_msg_detail, description="Amount of detail in the output (string)"),
    output_type: MessageOutputOptions = Query(default_msg_output, description="Type of message format (string or structure) of the output (string)")):
    """
    API endpoint: view_database()

    Returns information about all Representations in the database. Either all
    information or a summary is returned depending on the value of 'amt_detail'.
    Similarly, the output can be either a JSON-encoded structure or a
    string-like message depending on the value of 'output_type'.

    Parameters:
    - amt_detail : amount of detail in the output (string)

    - output_type: type of message format (string or structure) of the output
                    (string)

    Output:\n
        - If output_type='structure':
            JSON-encoded structure containing all (amt_detail='complete') or
            some (amt_detail='summary') of the attributes of each Representation
            in the database.
        
        - If output_type='message':
            String like message containing all (amt_detail='complete') or some
            (amt_detail='summary') of the attributes of each Representation
            in the database.
    """
    if output_type == MessageOutputOptions.STRUCTURE:
        # Initialize output object
        output_obj = []

        # Case 1: STRUCTURE & SUMMARY
        if   amt_detail == MessageDetailOptions.SUMMARY:
            if len(glb.rep_db) > 0:
                for rep in glb.rep_db:
                    output_obj.append(RepsSummaryOutput(unique_id=rep.unique_id,
                                                    image_name=rep.image_name,
                                                    group_no=rep.group_no,
                                                    name_tag=rep.name_tag,
                                                    region=rep.region))

        # Case 2: STRUCTURE & COMPLETE
        elif amt_detail == MessageDetailOptions.COMPLETE:
            if len(glb.rep_db) > 0:
                for rep in glb.rep_db:
                    output_obj.append(RepsInfoOutput(unique_id=rep.unique_id,
                        image_name=rep.image_name, group_no=rep.group_no,
                        name_tag=rep.name_tag, image_fp=rep.image_fp,
                        region=rep.region,
                        embeddings=[name for name in rep.embeddings.keys()]))

        # Case 3: [Exception] STRUCTURE & ???
        else:
            raise AssertionError('Amout of detail should be '\
                               + 'either SUMMARY or COMPLETE.')


    elif output_type == MessageOutputOptions.MESSAGE:
        # Initialize output object
        output_obj = ''

        # Case 4: MESSAGE & SUMMARY
        if   amt_detail == MessageDetailOptions.SUMMARY:
            if len(glb.rep_db) > 0:
                for rep in glb.rep_db:
                    output_obj += f'UID: {rep.unique_id}'.ljust(25)\
                           +  f' | Image name: {rep.image_name}'.ljust(25)\
                           +  f' | Group: {rep.group_no}'.ljust(15)\
                           +  f' | Tag: {rep.name_tag}'.ljust(25)\
                           +  f' | Region: {rep.region}'.ljust(15) + '\n'
            else:
                output_obj = 'Database is empty.'

        # Case 5: MESSAGE & COMPLETE
        elif amt_detail == MessageDetailOptions.COMPLETE:
            if len(glb.rep_db) > 0:
                for rep in glb.rep_db:
                    embd_names = [name for name in rep.embeddings.keys()]

                    output_obj += f'UID: {rep.unique_id}'.ljust(25)\
                           +  f' | Image name: {rep.image_name}'.ljust(25)\
                           +  f' | Group: {rep.group_no}'.ljust(15)\
                           +  f' | Tag: {rep.name_tag}'.ljust(15)\
                           +  f' | Region: {rep.region}'.ljust(30)\
                           +  f' | Image path: {rep.image_fp}'.ljust(30)\
                           + ' | embeddings: {}'.format(', '.join(embd_names))\
                           + '\n'
            else:
                output_obj = 'Database is empty.'

        # Case 6: [Exception] MESSAGE & ???
        else:
            raise AssertionError('Amout of detail should be '\
                               + 'either SUMMARY or COMPLETE.')

    else:
        raise AssertionError('Output type should be '\
                           + 'either STRUCTURE or MESSAGE.')

    return output_obj
        
# ------------------------------------------------------------------------------

@fr_router.post("/utility/clear_database")
async def clear_database():
    """
    API endpoint: clear_database()

    Clears the database. This is equivalent to setting the database to an empty
    one.

    Parameters:
    - None

    Output:\n
        JSON-encoded dictionary with the following attributes:
            1. message: message stating the database has been cleard [string]
    """
    # Clears the database
    glb.rep_db = []

    return {"message": "Database has been cleared."}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/edit_tag_by_uid/")
async def edit_tag_by_uid(
    target_uid  : str = Query(None, description="Unique identifier (string)"),
    new_name_tag: str = Query(None, description="New name tag (string)")):
    """
    API endpoint: edit_tag_by_uid()

    Allows the user to edit the name tag of a specific image (Representation) by
    providing its unique identifier.

    Parameters:
    - target_uid: unique identifier (string)

    - new_name_tag: new name tag (string)

    Output:\n
        JSON-encoded Representation structure with the following attributes:
            1. unique_id : unique identifier (UUID)
            2. name_tag  : name tag (string)
            3. image_name: image name (string)
            4. image_fp  : image full path (string)
            5. region    : the region is a 4 element list (list of integers)
            6. embedding : list of face verifier names for which this
                            Representation has embeddings for
    """
    # Gets the unique ids in the database depending on the database's size
    if len(glb.rep_db) == 0:   # no representations
        all_uids = []

    elif len(glb.rep_db) == 1: # single representation
        all_uids = [glb.rep_db[0].unique_id]

    elif len(glb.rep_db) > 1:  # many representations
        # Loops through each representation in the database and gets the unique
        # id tag
        all_uids = []
        for rep in glb.rep_db:
            all_uids.append(rep.unique_id)
    
    else: # this should never happen (negative size for a database? preposterous!)
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    # Search the list for the chosen unique id
    try:
        index = all_uids.index(UUID(target_uid))
    except:
        index = -1

    # If a match if found, update the name tag and return the updated
    # representation
    if index >= 0:
        glb.rep_db[index].name_tag = new_name_tag
        glb.db_changed = True
        rep            = glb.rep_db[index]
        output_obj     = RepsInfoOutput(unique_id  = rep.unique_id,
                                        name_tag   = rep.name_tag,
                                        image_name = rep.image_name,
                                        image_fp   = rep.image_fp,
                                        region     = rep.region,
                                        embeddings = [name for name in\
                                                    rep.embeddings.keys()])
        
    else:
        output_obj = []

    return output_obj

# ------------------------------------------------------------------------------

@fr_router.post("/utility/search_database_by_tag/")
async def search_database_by_tag(target_tag: str = Query(None, description="Name tag to be searched (string)")):
    """
    API endpoint: search_database_by_tag()

    Allows the user to search the database using a name tag. Returns all
    database entries containing the name tag provided. This search is CASE
    SENSITIVE.

    Parameters:
    - target_tag: name tag to be searched (string)

    Output:\n
        JSON-encoded structure with the following attributes:
            1. unique_ids : list of unique identifiers (list of UUIDs)
            2. name_tags  : list of name tags (list of strings)
            3. image_names: list of image names (list of strings)
            4. image_fps  : list of image full paths (list of strings)
            5. regions    : list of regions. Each region is a 4 element list
                            (list of list of integers)
            6. embeddings : list of face verifier names for which this
                            Representation has embeddings for
    """
    # Initialize output object
    output_obj = []

    # Gets the names in the database depending on the database's size
    if len(glb.rep_db) == 0:   # no representations
        return output_obj

    elif len(glb.rep_db) >= 1: # one or many representations
        for rep in glb.rep_db:
            if target_tag == rep.name_tag:
                output_obj.append(RepsInfoOutput(unique_id = rep.unique_id,
                                            name_tag   = rep.name_tag,
                                            image_name = rep.image_name,
                                            image_fp   = rep.image_fp,
                                            region     = rep.region,
                                            embeddings = [name for name in\
                                                    rep.embeddings.keys()]))

    else: # this should never happen (negative size for a database? preposterous!)
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    return output_obj

# ------------------------------------------------------------------------------

@fr_router.post("/utility/rename_entries_by_tag/")
async def rename_entries_by_tag(
    old_tag: str = Query(None, description="Old name tag (used in search) (string)"),
    new_tag: str = Query(None, description="New name tag (string)")):
    """
    API endpoint: rename_entries_by_tag()

    Allows the user to rename all database entries related to a specific tag
    with a new one. This operation is similar to a search_database_by_tag()
    followed by updating the name tag of each result with the new tag.

    Parameters:
    - old_tag: old name tag - this will be used to search for the database
                entries (string)

    - new_tag: new name tag (string)

    Output:\n
        JSON-encoded structure with the following attributes:
            1. unique_ids : list of unique identifiers (list of UUIDs)
            2. name_tags  : list of name tags (list of strings)
            3. image_names: list of image names (list of strings)
            4. image_fps  : list of image full paths (list of strings)
            5. regions    : list of regions. Each region is a 4 element list
                            (list of list of integers)
            6. embeddings : list of face verifier names for which this
                            Representation has embeddings for
    """
    # Initialize output object
    output_obj = []

    # Gets the names in the database depending on the database's size
    if len(glb.rep_db) == 0:   # no representations
        return output_obj

    elif len(glb.rep_db) >= 1: # one or many representations
        for i, rep in enumerate(glb.rep_db):
            if old_tag == rep.name_tag:
                glb.rep_db[i].name_tag = new_tag
                output_obj.append(RepsInfoOutput(unique_id = rep.unique_id,
                                            name_tag   = new_tag,
                                            image_name = rep.image_name,
                                            image_fp   = rep.image_fp,
                                            region     = rep.region,
                                            embeddings = [name for name in\
                                                    rep.embeddings.keys()]))
                glb.db_changed = True

    else: # this should never happen (negative size for a database? preposterous!)
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    return output_obj

# ------------------------------------------------------------------------------

@fr_router.post("/utility/update_rep_by_uid/")
async def update_rep_by_uid(
    target_uid  : str = Query(None, description="Unique identifier (string)"),
    new_name_tag: str = Query(None, description="New name tag (string)"),
    new_img_name: str = Query(None, description="New image name (string)"),
    new_img_fp  : str = Query(None, description="New image full path (string)"),
    new_region  : List[int] = Query(None, description="New face region in image (list, len=4)")):
    """
    API endpoint: update_rep_by_uid()

    Allows the user to update many parameters of a specific image
    (Representation) by providing its unique identifier.

    Parameters:
    - target_uid: unique identifier (string)

    - new_name_tag: new name tag (string)

    - new_img_name: new image name (string)

    - new_img_fp  : new image full path (string)

    - new_region  : new face region in image (list, len=4)

    Output:\n
        JSON-encoded Representation structure with the following attributes:
            1. unique_id : unique identifier (UUID)
            2. name_tag  : name tag (string)
            3. image_name: image name (string)
            4. image_fp  : image full path (string)
            5. region    : the region is a 4 element list (list of integers)
            6. embedding : list of face verifier names for which this
                            Representation has embeddings for
    """
    # Gets the unique ids in the database depending on the database's size
    if len(glb.rep_db) == 0:   # no representations
        all_uids = []

    elif len(glb.rep_db) == 1: # single representation
        all_uids = [glb.rep_db[0].unique_id]

    elif len(glb.rep_db) > 1:  # many representations
        # Loops through each representation in the database and gets the unique
        # id tag
        all_uids = []
        for rep in glb.rep_db:
            all_uids.append(rep.unique_id)
    
    else: # this should never happen (negative size for a database? preposterous!)
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    # Search the list for the chosen unique id
    try:
        index = all_uids.index(UUID(target_uid))
    except:
        index = -1

    # If a match if found, update the attributes and return the updated
    # representation
    if index >= 0:
        glb.rep_db[index].name_tag   = new_name_tag
        glb.rep_db[index].image_name = new_img_name
        glb.rep_db[index].image_fp   = new_img_fp
        glb.rep_db[index].region     = new_region

        glb.db_changed = True
        rep            = glb.rep_db[index]
        output_obj     = RepsInfoOutput(unique_id  = rep.unique_id,
                                        name_tag   = rep.name_tag,
                                        image_name = rep.image_name,
                                        image_fp   = rep.image_fp,
                                        region     = rep.region,
                                        embeddings = [name for name in\
                                                    rep.embeddings.keys()])
        
    else:
        output_obj = []

    return output_obj

# ------------------------------------------------------------------------------

@fr_router.post("/utility/save_database")
async def save_database(rdb_dir: str = Query(glb.RDB_DIR, description="Full path to Representation database directory (string)")):
    """
    API endpoint: save_database()

    Allows the user to save the database. Sets the global 'database has been
    modified' flag to False (i.e. database has not been modified).

    Parameters:
    - rdb_dir: full path to Representation database directory (string)

    Output:\n
        JSON-encoded dictionary with the following attributes:
            1. message: message stating the database has been saved
            2. status : indicates if an error occurred (status=1) or not
                        (status=0) (boolean)
    """
    # Obtains the database's full path
    try:
        db_fp = os.path.join(rdb_dir, 'rep_database.pickle')

        # Opens the database file (or creates one if necessary) and stores the
        # database object
        with open(db_fp, 'wb') as handle:
            pickle.dump(glb.rep_db, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Updates the database changed flag
        glb.db_changed = False

        # Creates output message and sets the status as 0
        output_msg = "Databased saved!"
        status     = False
    except Exception as excpt:
        # On exception, create output message (with exception) and set the
        # status as 1 (= something went wrong)
        output_msg = f"Unable to save database. Reason: {excpt}"
        status     = True

    return {"message":output_msg, "status":status}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/reload_database")
async def reload_database(
    rdb_dir: str  = Query(glb.RDB_DIR, description="Full path to Representation database directory (string)"),
    verbose: bool = Query(False, description="Controls the amount of text that is printed to the server's console (boolean)")):
    """
    API endpoint: reload_database()

    Allows the user to reload the database. Sets the global 'database has been
    modified' flag to True (i.e. database has been modified).

    Parameters:
    - rdb_dir: full path to Representation database directory (string)

    - verbose: controls the amount of text that is printed to the server's
                console (boolean)

    Output:\n
        JSON-encoded dictionary with the following attributes:
            1. message: message stating the database has been reloaded (string)
            2. status : indicates if an error occurred (status=1) or not
                        (status=0) (boolean)
    """
    # 
    try:
        # Attempts to load the database
        glb.rep_db = load_representation_db(os.path.join(rdb_dir,
                                        'rep_database.pickle'), verbose=verbose)

        # Sets the 'database has been modified' flag to True, creates output
        # message and sets the status as 0
        glb.db_changed = True
        output_msg     = "Databased reloaded!"
        status         = False
    except Exception as excpt:
        # On exception, create output message (with exception) and set the
        # status as 1 (= something went wrong)
        output_msg = f"Unable to save database. Reason: {excpt}"
        status     = True

    return {"message":output_msg, "status":status}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/view_by_group_no")
async def view_by_group_no(target_group_no: int = Query(None, description="Target group number (>= -1) [integer]"),
    return_type  : str = Query('reps', description="Desired return type. Options: 'reps' or 'image' [str]"),
    ncols        : int = Query(4, description="Number of columns in the output image [integer]"),
    figsize      : Tuple[int, int] = Query((15, 15), description="Figure size in inches [tuple of 2 integers]"),
    color        : str = Query('black', description="Text color [str]"),
    add_separator: bool = Query(False, description="Adds a caption acting as a separator [boolean]")):
    """
    API endpoint: view_by_group_no()

    Views all information of all Representations belonging to a group number 
    specified by 'target_group_no'.

    Parameters:
    - target_group_no : desired group / cluster number [integer]

    - return_type : determines if the output of this endpoint will be
         JSON-encoded structures of the Representations ('reps') OR an image
         ('image') (default='reps') [string]

    - ncols : number of columns in the plot image. Ignored if return_type='reps'
         (default=4) [integer]
    
    - figsize : output figure size in inches. Ignored if return_type='reps'
         (default=(15, 15)) [tuple of 2 integers]

    - color : text color (default='black') [string]

    - add_separator : toggles between adding a caption acting as a separator or
         not (default=False) [boolean]

    Output:\n
        - If return_type='reps':
            JSON-encoded structure containing all attributes of each
            Representation in the database belonging to the desired group.

        - If return_type='image':
            Plot of all images in the database belonging to the desired group.
    """
    # Initialize output object
    reps_found = []

    # Gets the Representations in the database which correspond to the desired
    # group number
    if len(glb.rep_db) == 0:   # no representations
        return reps_found

    elif len(glb.rep_db) >= 1: # one or many representations
        for rep in glb.rep_db:
            if target_group_no == rep.group_no:
                reps_found.append(RepsInfoOutput(unique_id = rep.unique_id,
                                                image_name = rep.image_name,
                                                group_no   = rep.group_no,
                                                name_tag   = rep.name_tag,
                                                image_fp   = rep.image_fp,
                                                region     = rep.region,
                                                embeddings = [name for name in\
                                                    rep.embeddings.keys()]))

    else: # this should never happen
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    # Returns the output as either an image or as Representations
    if return_type == 'image':
        # Obtains the subplot figure, seeks to the beginning (just to be safe)
        # and returns the response as a png image
        img_file = show_cluster_results(target_group_no, glb.rep_db,
                                    ncols=ncols, figsize=figsize, color=color,
                                    add_separator=add_separator)
        img_file.seek(0)

        return Response(content=img_file.read(), media_type="image/png")
    
    elif return_type == 'reps':
        # Alternatively, returns the Representations
        return reps_found

    else:
        raise ValueError("Return type should be either 'image' or 'reps'!")

# ------------------------------------------------------------------------------

@fr_router.post("/utility/remove_from_group")
async def remove_from_group(files: List[str]):
    """
    API endpoint: remove_from_group()

    Allows the user to remove files from a particular group. Effectively, this
    endpoint sets the group of all files (provided they are valid files) to -1
    (i.e. group-less).

    Images can be specified by either their unique identifier or their name. If
    an image file can not be found (because the unique identifier or image name
    does not match any present in the database) it will be skipped.

    Note: one can mix and match image names and unique identifiers in the same
    list.

    Parameters:
    - files: list of image names and/or unique identifiers [list of strings].

    Output:\n
        JSON-encoded dictionary containing the following key/value pairs:
            1. removed : number of files removed
            2. skipped : number of files skipped
    """
    # Initializes removed and skipped file / Representation counters
    removed_count = 0
    skipped_count = 0

    # Loops through each file
    for f in files:
        # Tries to load (or find) the file and obtain its embedding
        if string_is_valid_uuid4(f): # string is valid uuid
            for i in range(0, len(glb.rep_db)):
                # If the unique identifier of the current Representation
                # matches the target identifier, remove its group
                if glb.rep_db[i].unique_id == UUID(f):
                    glb.rep_db[i].group_no = -1
                    removed_count += 1 # increments removed counter
                else:
                    pass # do nothing

        elif os.path.isfile(os.path.join(img_dir, f)): # string is a valid file
            for i in range(0, len(glb.rep_db)):
                # If this Representation's full path matches the target
                # file's full path, remove its group
                if glb.rep_db[i].image_fp == f:
                    glb.rep_db[i].group_no = -1
                    removed_count += 1 # increments removed counter
                else:
                    pass # do nothing

        else: # string is not a valid uuid nor file
            skipped_count += 1

    return {'removed':removed_count, 'skipped':skipped_count}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/edit_tag_by_group_no")
async def edit_tag_by_group_no(target_group_no: int = Query(None, description="Target group number (>= -1) [integer]"),
    new_name_tag: str = Query(None, description="New name tag [string]")):
    """
    API endpoint: edit_tag_by_group_no()

    Allows the user to edit the name tag of all images (Representations)
    belonging to a same group.

    Parameters:
    - target_group_no: desired group number [integer]

    - new_name_tag: new name tag [string]

    Output:\n
        JSON-encoded Representation structure with the following attributes:
            1. unique_id : unique identifier [UUID]
            2. image_name: image name [string]
            3. group_no  : group number [integer]
            4. name_tag  : name tag [string]
            5. image_fp  : image full path [string]
            6. region    : the region is a 4 element list [list of integers]
            7. embedding : list of face verifier names for which this
                            Representation has embeddings for [list of strings]
    """
    # Initializes the output object
    output_obj = []

    # Updates the database depending on its size
    if len(glb.rep_db) == 0:   # no representations
        return output_obj

    elif len(glb.rep_db) > 0: # many representations
        # Loops through each Representation in the database
        for i in range(0, len(glb.rep_db)):
            # If the Representation belongs to the group, update the name tag
            if glb.rep_db[i].group_no == target_group_no:
                # Updates name tag and database changed flag
                glb.rep_db[i].name_tag = new_name_tag
                glb.db_changed         = True

                #
                rep = glb.rep_db[i]
                output_obj.append(RepsInfoOutput(unique_id  = rep.unique_id,
                                                 image_name = rep.image_name,
                                                 group_no   = rep.group_no,
                                                 name_tag   = rep.name_tag,
                                                 image_fp   = rep.image_fp,
                                                 region     = rep.region,
                                                 embeddings = [name for name in\
                                                    rep.embeddings.keys()]))

            # Otherwise, do nothing
            else:
                pass # do nothing
    
    else: # this should never happen
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    return output_obj

# ------------------------------------------------------------------------------

@fr_router.post("/utility/remove_images")
async def remove_images(files: List[str],
    remove_from: str = Query('database', description="Indicates if files should be removed from database ('database') or server ('server') [string]"),
    img_dir: str = Query(glb.IMG_DIR, description="Full path to image directory (string)")):
    """
    API endpoint: remove_images()

    Allows the user to remove:
        - image Representations from the database (remove_from='database')
                                        OR
        - images from the server and their Representations from the database
            (remove_from='server')

    Images can be specified by either their unique identifier or their name (as
    it is expected that each image in the server has exactly 1 Representation in
    the database). If an image file can not be found (because the unique
    identifier or image name does not match any present in the database) it will
    be skipped.

    Note: one can mix and match image names and unique identifiers in the same
    list.

    Parameters:
    - files: list of image names and/or unique identifiers [list of strings].

    - remove_from: specifies if images should be removed from the database or
        from both the database AND the server. Options: database or server
        (default='database') [string].
    
    - img_dir: full path of directory containing images [string].

    Output:\n
        JSON-encoded dictionary containing the following key/value pairs:
            1. removed : number of files removed
            2. skipped : number of files skipped
    """
    # Initializes removed and skipped file / Representation counters
    removed_count = 0
    skipped_count = 0

    # Loops through each file
    for f in files:
        # Tries to load (or find) the file and obtain its embedding
        if string_is_valid_uuid4(f): # string is valid uuid
            if remove_from == 'database':
                for rep in glb.rep_db:
                    # If the unique identifier of the current Representation
                    # matches the target identifier, remove this Representation
                    # from the database
                    if rep.unique_id == UUID(f):
                        glb.rep_db.remove(rep)
                        removed_count += 1 # increments removed counter
                    else:
                        pass # do nothing

            elif remove_from == 'server':
                for rep in glb.rep_db:
                    # If the unique identifier of the current Representation
                    # matches the target identifier, remove this file from the
                    # server and its Representation from the database
                    if rep.unique_id == UUID(f):
                        os.remove(rep.image_fp)
                        glb.rep_db.remove(rep)
                        removed_count += 1 # increments removed counter
                    else:
                        pass # do nothing

            else:
                raise ValueError("remove_from should be either 'database' or 'server'")

        elif os.path.isfile(os.path.join(img_dir, f)): # string is a valid file
            if remove_from == 'database':
                for rep in glb.rep_db:
                    # If this Representation's full path matches the target
                    # file's full path, remove this Representation from the
                    # database
                    if rep.image_fp == f:
                        glb.rep_db.remove(rep)
                        removed_count += 1 # increments removed counter
                    else:
                        pass # do nothing

            elif remove_from == 'server':
                for rep in glb.rep_db:
                    # If this Representation's full path matches the target
                    # file's full path, remove the file from the server and its
                    # Representation from the database
                    if rep.image_fp == f:
                        os.remove(rep.image_fp)
                        glb.rep_db.remove(rep)
                        removed_count += 1 # increments removed counter
                    else:
                        pass # do nothing

            else:
                raise ValueError("remove_from should be either 'database' or 'server'")

        else: # string is not a valid uuid nor file
            skipped_count += 1

    return {'removed':removed_count, 'skipped':skipped_count}

# ------------------------------------------------------------------------------

@fr_router.post("/create_database/from_directory")
async def create_database_from_directory(cdb_params: CreateDatabaseParams,
    image_dir   : Optional[str]  = Query(glb.IMG_DIR, description="Full path to directory containing images (string)"),
    db_dir      : Optional[str]  = Query(glb.RDB_DIR, description="Full path to directory containing saved database (string)"),
    force_create: Optional[bool] = Query(False      , description="Flag to force database creation even if one already exists, overwritting the old one (boolean)")):
    """
    API endpoint: create_database_from_directory()

    Creates a database from a directory. The directory is expected to have image
    files in any of the following formats: .jpg, .png, .npy. The database is a
    list of Representation objects.

    Parameters:
    - cdb_params: a structure with the following parameters:
        1. detector_name  - name of face detector model [string]
        2. verifier_names - list of names of face verifier models [list of
                            strings]
        3. align          - perform face alignment flag (default=True) [boolean]
        4. normalization  - name of image normalization [string]
        5. tags           - list of name tags [list of strings]
        6. uids           - list of uids [list of strings]
        7. auto_grouping  - toggles whether Representations should be grouped /
                            clusted automatically using the DBSCAN algorithm
                            (default=True) [boolean]
        8. eps            - maximum distance between two samples for one to be
                            considered as in the neighborhood of the other. This
                            is the most important DBSCAN parameter to choose
                            appropriately for the specific data set and distance
                            function (default=0.5) [float]
        9. min_samples    - the number of samples (or total weight) in a
                            neighborhood for a point to be considered as a core
                            point. This includes the point itself
                            (min_samples=2) [integer]
       10. metric         - the metric used when calculating distance between
                            instances in a feature array. It must be an option
                            allowed by sklearn.metrics.pairwise_distances
                            (default='cosine') [string]
       11. verbose        - output messages to server's console (boolean,
                            default: False)

        [Example] JSON schema:
        {
            "detector_name": "retinaface",
            "verifier_names": ["ArcFace"],
            "align": true,
            "normalization": "base",
            "tags": [],
            "uids": [],
            "auto_grouping": true,
            "eps": 0.5,
            "min_samples": 2,
            "metric": "cosine",
            "verbose": false
        }

    - image_dir   : full path to directory containing images (string,
                     default: <glb.IMG_DIR>)

    - db_dir      : full path to directory containing saved database (string,
                     default: <glb.RDB_DIR>)

    - force_create: flag to force database creation even if one already exists,
                     overwritting the old one (boolean, default: True)

    Output:\n
        JSON-encoded dictionary with the following key/value pairs is returned:
            1. length: length of the newly created database OR of the currently
                loaded one if this process is skipped (i.e. force_create=False
                with existing database loaded)
            
            2. message: informative message string
    """
    # Initialize output message
    output_msg = ''

    # If image directory provided is None or is not a directory, use default
    # directory
    if not image_dir or not os.path.isdir(image_dir):
        global img_dir
        output_msg += 'Image dir is None, does not exist or is not a '\
                   +  'directory. Using default directory instead.\n'
        image_dir  = img_dir

    # If database path provided is None or is not a directory, use default
    # directory
    if not db_dir or not os.path.isdir(db_dir):
        global rdb_dir
        output_msg += 'Database dir is None, does not exist or is not a '\
                   +  'directory. Using default directory instead.\n'
        db_dir = os.path.join(rdb_dir, 'rep_database.pickle')

    # Database exists (and has at least one element) and force create is False
    if len(glb.rep_db) > 0 and not force_create:
        # Do nothing, but set message
        output_msg += 'Database exists (and force create is False). '\
                   +  'Skipping database creation.\n'

    elif len(glb.rep_db) == 0 or force_create:
        output_msg += 'Creating database: '
        glb.rep_db  = create_reps_from_dir(image_dir, glb.models,
                                detector_name=cdb_params.detector_name,
                                align=cdb_params.align, show_prog_bar=True,
                                verifier_names=cdb_params.verifier_names,
                                normalization=cdb_params.normalization,
                                tags=cdb_params.tags, uids=cdb_params.uids,
                                auto_grouping=cdb_params.auto_grouping,
                                min_samples=cdb_params.min_samples,
                                eps=cdb_params.eps, metric=cdb_params.metric,
                                verbose=cdb_params.verbose)
        output_msg += 'success!\n'

    else:
        raise AssertionError('[create_database] Database should have a ' +  
                             'length of 0 or more - this should not happen!')

    # Modifies the database change flag (and sort the database if there is at
    # least 1 element)
    n_reps = len(glb.rep_db)
    if n_reps == 1:
        glb.db_changed = True

    elif n_reps > 1:
        # Sorts database and sets changed flag to True
        glb.rep_db.sort(key=lambda x: x.image_name)
        glb.db_changed = True

    else:
        glb.db_changed = False

    return {'length':len(glb.rep_db), 'message':output_msg}

# ------------------------------------------------------------------------------

@fr_router.post("/create_database/from_zip")
async def create_database_from_zip(myfile: UploadFile,
    params      : CreateDatabaseParams = Depends(),
    image_dir   : Optional[str]  = Query(glb.IMG_DIR, description="Full path to directory containing images (string)"),
    db_dir      : Optional[str]  = Query(glb.RDB_DIR, description="Full path to directory containing saved database (string)"),
    auto_rename : Optional[bool] = Query(True       , description="Flag to force auto renaming of images in the zip file with names that match images already in the image directory (boolean)"),
    force_create: Optional[bool] = Query(False      , description="Flag to force database creation even if one already exists, overwritting the old one (boolean)")):
    """
    API endpoint: create_database_from_zip()

    Creates a database from a zip file. The zip file is expected to contain
    image files in any of the following formats: .jpg, .png, .npy. The database
    is a list of Representation objects.

    The images in the zip file are extracted to a temporary directory. Any image
    with the same name of another image in the 'image directory' is either
    renamed (auto_rename=True) or skipped (auto_rename=False). Renamed images
    are renamed using its unique object identifier.

    Parameters:
    - myfile: a zip file
    - params: a structure with the following parameters:
        1. detector_name  - name of face detector model [string]
        2. verifier_names - list of names of face verifier models [list of
                            strings]
        3. align          - perform face alignment flag (default=True) [boolean]
        4. normalization  - name of image normalization [string]
        5. tags           - list of name tags [list of strings]
        6. uids           - list of uids [list of strings]
        7. auto_grouping  - toggles whether Representations should be grouped /
                            clusted automatically using the DBSCAN algorithm
                            (default=True) [boolean]
        8. eps            - maximum distance between two samples for one to be
                            considered as in the neighborhood of the other. This
                            is the most important DBSCAN parameter to choose
                            appropriately for the specific data set and distance
                            function (default=0.5) [float]
        9. min_samples    - the number of samples (or total weight) in a
                            neighborhood for a point to be considered as a core
                            point. This includes the point itself
                            (min_samples=2) [integer]
       10. metric         - the metric used when calculating distance between
                            instances in a feature array. It must be an option
                            allowed by sklearn.metrics.pairwise_distances
                            (default='cosine') [string]
       11. verbose        - output messages to server's console (boolean,
                            default: False)

    - image_dir   : full path to directory containing images (string,
                     default: <glb.IMG_DIR>)

    - db_dir      : full path to directory containing saved database (string,
                     default: <glb.RDB_DIR>)

    - auto_rename : flag to force auto renaming of images in the zip file with
                     names that match images already in the image directory
                     (boolean, default: True)

    - force_create: flag to force database creation even if one already exists,
                     overwritting the old one (boolean, default: True)

    Output:\n
        JSON-encoded dictionary with the following key/value pairs is returned:
            1. length: length of the newly created database OR of the currently
                loaded one if this process is skipped (i.e. force_create=False
                with existing database loaded)
            
            2. message: informative message string
    """
    # Initialize output message
    output_msg = ''

    # If image directory provided is None or is not a directory, use default
    # directory
    if not image_dir or not os.path.isdir(image_dir):
        global img_dir
        output_msg += 'Image dir is None, does not exist or is not a '\
                   +  'directory. Using default directory instead.\n'
        image_dir = img_dir

    # If database path provided is None or is not a directory, use default
    # directory
    if not db_dir or not os.path.isdir(db_dir):
        global rdb_dir
        output_msg += 'Database dir is None, does not exist or is not a '\
                   +  'directory. Using default directory instead.\n'
        db_dir     = os.path.join(rdb_dir, 'rep_database.pickle')

    # Database exists (and has at least one element) and force create is False
    if len(glb.rep_db) > 0 and not force_create:
        # Do nothing, but set message
        output_msg += 'Database exists (and force create is False). '\
                   +  'Skipping database creation.\n'

    elif len(glb.rep_db) == 0 or force_create:
        # Initialize dont_skip flag as True
        dont_skip = True

        # Extract zip files
        output_msg += 'Extracting images in zip:'
        try:
            # Create temporary directory and extract all files to it
            tempdir = mkdtemp(prefix="create_database_from_zip-")
            with ZipFile(BytesIO(myfile.file.read()), 'r') as myzip:
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

            output_msg += ' success! '
        except Exception as excpt:
            dont_skip   = False
            output_msg += f' failed (reason: {excpt}).'
        finally:
            # Remove the temporary directory
            rmtree(tempdir)

        # Create database
        if dont_skip:
            output_msg += 'Creating database: '
            glb.rep_db  = create_reps_from_dir(image_dir, glb.models,
                                        detector_name=params.detector_name,
                                        align=params.align, show_prog_bar=True,
                                        verifier_names=params.verifier_names,
                                        normalization=params.normalization,
                                        tags=params.tags, uids=params.uids,
                                        auto_grouping=params.auto_grouping,
                                        min_samples=params.min_samples,
                                        eps=params.eps, metric=params.metric,
                                        verbose=params.verbose)

            # Loops through each representation
            for i, rep in enumerate(glb.rep_db):
                # Determines the current image name and, if it is one of the
                # files that was renamed, use its name (which is a unique id) as
                # its unique id
                cur_img_name = rep.image_name.split('.')[0]
                cur_img_name = cur_img_name[0:8]   + '-' +\
                               cur_img_name[8:12]  + '-' +\
                               cur_img_name[12:16] + '-' +\
                               cur_img_name[16::]
                
                if cur_img_name in new_names:
                    glb.rep_db[i].unique_id = UUID(cur_img_name)
                    glb.db_changed          = True

            # TODO: Refactor the unzipping + renaming process into a better
            # function
            # if len(new_names) > 0: # need to fix these files
            output_msg += 'success!\n'
        else:
            output_msg += 'Skipping database creation.\n'

    else:
        raise AssertionError('[create_database] Database should have a ' +  
                             'length of 0 or more - this should not happen!')

    # Modifies the database change flag (and sort the database if there is at
    # least 1 element)
    n_reps = len(glb.rep_db)
    if n_reps == 1:
        glb.db_changed = True

    elif n_reps > 1:
        # Sorts database and sets changed flag to True
        glb.rep_db.sort(key=lambda x: x.image_name)
        glb.db_changed = True

    else:
        glb.db_changed = False

    return {'length':len(glb.rep_db), 'message':output_msg}

# ------------------------------------------------------------------------------

@fr_router.post("/verify/no_upload", response_model=List[VerificationMatchesItem])
async def verify_no_upload(files: List[UploadFile],
                          params: VerificationParams = Depends()):
    """
    API endpoint: verify_no_upload()

    Executes the face verification process on one or more images without
    uploading these images to the server and adding them to the database.

    Parameters:
    - files: list of image files

    - params: a structure with the following parameters:
        1. detector_name - name of face detector model (string)
        2. verifier_name - name of face verifier model (string)
        3. align         - perform face alignment flag (boolean, default: True)
        4. normalization - name of image normalization (string)
        5. metric        - name of similarity / distance metric (string)
        6. threshold     - cutoff value for positive (match) decision (float)
        7. verbose       - output messages to server's console (boolean,
                            default: False)

    Output:\n
        JSON-encoded structure with the following attributes:
            1. unique_ids : list of unique identifiers (list of UUIDs)
            2. name_tags  : list of name tags (list of strings)
            3. image_names: list of image names (list of strings)
            4. image_fps  : list of image full paths (list of strings)
            5. regions    : list of regions. Each region is a 4 element list
                            (list of list of integers)
            6. embeddings : list of face verifier names for which this
                            Representation has embeddings for
            7. distances  : list of distances (similarity) (list of floats)
            8. threshold  : decision (match) cutoff threshold (float)
    """
    # Initializes results list and obtains the relevant embeddings from the 
    # representation database
    verification_results = []
    dtb_embs = get_embeddings_as_array(glb.rep_db, params.verifier_name)

    # Loops through each file
    for f in files:
        # Obtains contents of the file & transforms it into an image
        data  = np.fromfile(f.file, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        image = image[:, :, ::-1]

        # Calculate the face image embedding
        region, embeddings = calc_embedding(image, glb.models,
                                            align=params.align,
                                            detector_name=params.detector_name, 
                                            verifier_names=params.verifier_name,
                                            normalization=params.normalization)

        # Calculates the embedding of the current image
        cur_emb  = embeddings[params.verifier_name]

        # Calculates the similarity between the current embedding and all
        # embeddings from the database
        similarity_obj = calc_similarity(cur_emb, dtb_embs,
                                         metric=params.metric,
                                         model_name=params.verifier_name,
                                         threshold=params.threshold)

        # Gets all matches based on the similarity object and append the result
        # to the results list
        result = get_matches_from_similarity(similarity_obj, glb.rep_db,
                                             params.verifier_name)
        # verification_results.append(result)
        result_df = pd.DataFrame(result)
        verification_results = result_df.to_dict('records')
        # TODO: check for multiple file, not working at the moment!

    return verification_results
    
# ------------------------------------------------------------------------------

@fr_router.post("/verify/with_upload", response_model=List[VerificationMatches])
async def verify_with_upload(files: List[UploadFile],
    params     : VerificationParams = Depends(),
    save_as    : ImageSaveTypes     = Query(default_image_save_type, description="File type which uploaded images should be saved as (string)"),
    overwrite  : bool               = Query(False, description="Flag to indicate if an uploaded image with the same name as an existing one in the server should be saved and replace it (boolean)"),
    auto_rename: bool               = Query(True, description="Flag to force auto renaming of images in the zip file with (boolean)")):
    """
    API endpoint: verify_with_upload()

    Executes the face verification process on one or more images. These images
    are uploaded to the server and added to the database if they do not exist
    already. If overwrite=True then newly uploaded images with the same name 
    replace existing images in the server with the same name.

    Parameters:
    - files: list of image files

    - params: a structure with the following parameters:
        1. detector_name - name of face detector model (string)
        2. verifier_name - name of face verifier model (string)
        3. align         - perform face alignment flag (boolean, default: True)
        4. normalization - name of image normalization (string)
        5. metric        - name of similarity / distance metric (string)
        6. threshold     - cutoff value for positive (match) decision (float)
        7. verbose       - output messages to server's console (boolean,
                            default: False)

    - save_as     : file type which uploaded images should be saved as (string)

    - overwrite   : flag to indicate if an uploaded image with the same name as
                    an existing one in the server should be saved and replace it
                    (boolean, default: False)

    - auto_rename : flag to force auto renaming of images in the zip file with
                    names that match images already in the image directory
                    (boolean, default: True)

    Output:\n
        JSON-encoded structure with the following attributes:
            1. unique_ids : list of unique identifiers (list of UUIDs)
            2. name_tags  : list of name tags (list of strings)
            3. image_names: list of image names (list of strings)
            4. image_fps  : list of image full paths (list of strings)
            5. regions    : list of regions. Each region is a 4 element list
                            (list of list of integers)
            6. embeddings : list of face verifier names for which this
                            Representation has embeddings for
            7. distances  : list of distances (similarity) (list of floats)
            8. threshold  : decision (match) cutoff threshold (float)
    """
    # Initializes results list, gets all files in the image directory and
    # obtains the relevant embeddings from the representation database
    verification_results = []
    all_files            = [name.split('.')[0] for name in os.listdir(img_dir)]
    dtb_embs   = get_embeddings_as_array(glb.rep_db, params.verifier_name)

    # Loops through each file
    for f in files:
        # Initializes empty unique id
        uid = ''

        # Obtains contents of the file & transforms it into an image
        data  = np.fromfile(f.file, dtype=np.uint8)
        img   = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        img   = img[:, :, ::-1]

        # Obtains the file's image name and creates the full path
        img_name = f.filename.split('.')[0]
        img_fp   = os.path.join(img_dir, img_name + '.' + save_as)

        # Saves the image if it does not exist or if overwrite is True.
        # Alternatively, if auto_rename is True, then automatically renames the
        # file (if the file name already exists) using its unique id and saves
        # it. Otherwise skips verification in this file.
        if not (img_name in all_files) or overwrite:
            if save_as == ImageSaveTypes.NPY:
                np.save(img_fp, img, allow_pickle=False, fix_imports=False)
            else:
                mpimg.imsave(img_fp, img)
        elif auto_rename:
            # Creates a new unique object identifier using uuid4, converts it
            # into a string, sets it as the file name, creates the file's full
            # path and saves it
            uid      = uuid4().hex 
            img_name = uid
            img_fp   = os.path.join(img_dir, img_name + '.' + save_as)

            if save_as == ImageSaveTypes.NPY:
                np.save(img_fp, img, allow_pickle=False, fix_imports=False)
            else:
                mpimg.imsave(img_fp, img)
        else:
            continue  # skips verification using this file

        # Calculate the face image embedding
        region, embeddings = calc_embedding(img, glb.models, align=params.align,
                                            detector_name=params.detector_name, 
                                            verifier_names=params.verifier_name,
                                            normalization=params.normalization)

        # Calculates the embedding of the current image
        cur_emb  = embeddings[params.verifier_name]

        # Calculates the similarity between the current embedding and all
        # embeddings from the database
        similarity_obj = calc_similarity(cur_emb, dtb_embs,
                                         metric=params.metric,
                                         model_name=params.verifier_name,
                                         threshold=params.threshold)

        # Gets all matches based on the similarity object and append the result
        # to the results list
        result = get_matches_from_similarity(similarity_obj, glb.rep_db,
                                             params.verifier_name)

        # Chooses a tag automatically from the best match if it has a distance / 
        # similarity of <=0.75*threshold
        if len(result) > 0:
            if similarity_obj['distances'][0]\
                <= 0.75 * similarity_obj['threshold']:
                tag = result['name_tags'][0]
            else:
                tag = ''
        else:
            tag = ''

        # Creates a representation object for this image and adds it to the
        # database
        new_rep = create_new_representation(img_fp, region, embeddings, tag=tag,
                                            uid=uid)
        glb.rep_db.append(new_rep)
        glb.db_changed = True

        # Stores the verification result
        verification_results.append(result)

    return verification_results

# ------------------------------------------------------------------------------

@fr_router.post("/verify/existing_file",
                response_model=List[VerificationMatches])
async def verify_existing_file(files: List[str],
            params: VerificationParams = Depends(),
            img_dir: str = Query(glb.IMG_DIR, description="Full path to image directory (string)")):
    """
    API endpoint: verify_existing_file()

    Executes the face verification process on one or more images already stored
    in the server and/or database.
    
    For images stored in the server (but not necessarily in the database), the
    files are specified as a list of image names. The image directory + image
    name must match the full path of the image stored in the server (so remember
    to include the extensions in the image name).

    For images stored in the database, the files are specified as a list of
    unique identifiers (also as strings). The images are searched in the
    database using the unique identifiers.

    If an image exists in both the server and database, then it is faster to
    provide the unique identifier as images in the database have their
    embeddings precalculated while an image stored in the server will have its
    embedding calculated.

    Note: one can mix and match image names and unique identifiers in the same
    list.

    Parameters:
    - files: list of image names and/or unique identifiers [list of strings].
        NOTE: CURRENTLY BUGGED FOR MULTIPLE FILES - PASS A SINGLE FILE FOR NOW!

    - params: a structure with the following parameters:
        1. detector_name - name of face detector model [string]
        2. verifier_name - name of face verifier model [string]
        3. align         - perform face alignment flag (default=True) [boolean]
        4. normalization - name of image normalization [string]
        5. metric        - name of similarity / distance metric [string]
        6. threshold     - cutoff value for positive (match) decision [float]
        7. verbose       - output messages to server's console (default=False)
                            [boolean]

    - img_dir: full path of directory containing images [string].

    Output:\n
        JSON-encoded structure with the following attributes:
            1. unique_ids : list of unique identifiers (list of UUIDs)
            2. name_tags  : list of name tags (list of strings)
            3. image_names: list of image names (list of strings)
            4. image_fps  : list of image full paths (list of strings)
            5. regions    : list of regions. Each region is a 4 element list
                            (list of list of integers)
            6. embeddings : list of face verifier names for which this
                            Representation has embeddings for
            7. distances  : list of distances (similarity) (list of floats)
            8. threshold  : decision (match) cutoff threshold (float)
    """
    # Initializes results list and obtains the relevant embeddings from the 
    # Representation database
    verification_results = []
    dtb_embs = get_embeddings_as_array(glb.rep_db, params.verifier_name)

    # Gets the unique ids in the database depending on the database's size
    if len(glb.rep_db) == 0:   # no representations
        all_uids = []

    elif len(glb.rep_db) == 1: # single representation
        all_uids = [glb.rep_db[0].unique_id]

    elif len(glb.rep_db) > 1:  # many representations
        # Loops through each Representation in the database and gets the unique
        # id tag
        all_uids = []
        for rep in glb.rep_db:
            all_uids.append(rep.unique_id)
    
    else: # this should never happen (negative size for a database? preposterous!)
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    # Loops through each file
    for f in files:
        # Tries to load (or find) the file and obtain its embedding
        if string_is_valid_uuid4(f): # string is valid uuid
            # Finds the Representation from the unique identifier. If this
            # fails, skips the file and continue
            try:
                index = all_uids.index(UUID(f))
            except:
                continue

            # Obtain the embeddings
            cur_emb = glb.rep_db[index].embeddings[params.verifier_name]

        elif os.path.isfile(os.path.join(img_dir, f)): # string is a valid file
            # Opens the image file
            image = cv2.imread(os.path.join(img_dir, f))
            image = image[:, :, ::-1]

            # Calculate the face image embedding
            region, embeddings = calc_embedding(image, glb.models,
                                        align=params.align,
                                        detector_name=params.detector_name, 
                                        verifier_names=params.verifier_name,
                                        normalization=params.normalization)

            # Calculates the embedding of the current image
            cur_emb = embeddings[params.verifier_name]

        else: # string is not a valid uuid nor file
            continue

        # Calculates the similarity between the current embedding and all
        # embeddings from the database
        similarity_obj = calc_similarity(cur_emb, dtb_embs,
                                        metric=params.metric,
                                        model_name=params.verifier_name,
                                        threshold=params.threshold)

        # Gets all matches based on the similarity object and append the result
        # to the results list
        result = get_matches_from_similarity(similarity_obj, glb.rep_db,
                                                 params.verifier_name)
        verification_results.append(result)

    print(verification_results)

    return verification_results
    
# ------------------------------------------------------------------------------

