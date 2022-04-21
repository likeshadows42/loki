# ==============================================================================
#                        TEST API METHODS
# ==============================================================================
import os
import cv2
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy                 as np
import api.global_variables  as glb

from uuid                    import UUID, uuid4
from pydoc                   import describe
from pandas                  import describe_option
from typing                  import List, Optional
from fastapi                 import APIRouter, UploadFile, Depends, Query, Body
from IFR.api                 import load_representation_db, load_face_verifier,\
                            create_reps_from_dir, get_embeddings_as_array,\
                            fix_uid_of_renamed_imgs, process_image_zip_file,\
                            get_matches_from_similarity, show_cluster_results,\
                            create_new_representation
from IFR.classes             import *
from IFR.functions           import create_dir, string_is_valid_uuid4,\
                               calc_embedding, calc_similarity
from fastapi.responses       import Response

# from shutil                   import rmtree, move     as sh_move
from matplotlib               import image            as mpimg
from deepface.DeepFace        import build_model      as build_verifier
import pandas as pd


glb_data_dir = glb.DATA_DIR
glb_img_dir  = glb.IMG_DIR
glb_rdb_dir  = glb.RDB_DIR

# ______________________________________________________________________________
#                             ROUTER INITIALIZATION
# ------------------------------------------------------------------------------

fr_router = APIRouter()

# ______________________________________________________________________________
#                                  API METHODS
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
    glb.API_DIR = os.path.join(os.path.dirname(os.path.realpath("__file__")),
                               'api')
    print('[PATH]'.ljust(12), 'API_DIR'.ljust(12),
         f': reset! ({glb.API_DIR})')

    glb.DATA_DIR    = os.path.join(glb.API_DIR    , 'data')
    print('[PATH]'.ljust(12), 'DATA_DIR'.ljust(12),
         f': reset! ({glb.DATA_DIR})')
    
    glb.IMG_DIR     = os.path.join(glb.DATA_DIR  , 'img')
    print('[PATH]'.ljust(12), 'IMG_DIR'.ljust(12),
         f': reset! ({glb.IMG_DIR})')
    
    glb.RDB_DIR     = os.path.join(glb.DATA_DIR  , 'database')
    print('[PATH]'.ljust(12), 'RDB_DIR'.ljust(12),
         f': reset! ({glb.RDB_DIR})')
    
    glb.SVD_MDL_DIR = os.path.join(glb.API_DIR       , 'saved_models')
    print('[PATH]'.ljust(12), 'SVD_MDL_DIR'.ljust(12),
         f': reset! ({glb.SVD_MDL_DIR})')
    
    glb.SVD_VRF_DIR = os.path.join(glb.SVD_MDL_DIR   , 'verifiers')
    print('[PATH]'.ljust(12), 'SVD_VRF_DIR'.ljust(12),
         f': reset! ({glb.SVD_VRF_DIR})')
    print('')

    glb.models     = {}
    print('[MODELS]'.ljust(12), 'models'.ljust(12),
         f': reset! ({glb.models})')
    glb.rep_db     = RepDatabase()
    print('[DATABASE]'.ljust(12), 'rep_db'.ljust(12),
         f': reset! ({glb.rep_db})')
    glb.db_changed = False
    print('[FLAG]'.ljust(12), 'db_changed'.ljust(12),
         f': reset! ({glb.db_changed})')
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

@fr_router.post("/utility/view_database")
async def view_database(amt_detail : MessageDetailOptions = Query(default_msg_detail, description="Amount of detail in the output (string)")):
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
    # Initialize output object
    output_obj = []

    # Case 1: SUMMARY
    if   amt_detail == MessageDetailOptions.SUMMARY:
        if glb.rep_db.size > 0:
            for rep in glb.rep_db.reps:
                output_obj.append(RepsSummaryOutput(unique_id=rep.unique_id,
                                                    image_name=rep.image_name,
                                                    group_no=rep.group_no,
                                                    name_tag=rep.name_tag,
                                                    region=rep.region))

    # Case 2: COMPLETE
    elif amt_detail == MessageDetailOptions.COMPLETE:
        if glb.rep_db.size > 0:
            for rep in glb.rep_db.reps:
                output_obj.append(RepsInfoOutput(unique_id=rep.unique_id,
                        image_name=rep.image_name, group_no=rep.group_no,
                        name_tag=rep.name_tag, image_fp=rep.image_fp,
                        region=rep.region,
                        embeddings=[name for name in rep.embeddings.keys()]))

    # Case 3: [Exception] UNKNOWN AMOUNT OF DETAIL
    else:
        raise AssertionError('Amout of detail should be '\
                           + 'either SUMMARY or COMPLETE.')

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
    glb.rep_db.clear()

    return {"message": "Database has been cleared."}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/search_database/", response_model=List[RepsInfoOutput])
async def search_database(search_terms: List[str] = Query(None, description="Name tag to be searched (string)")):
    """
    API endpoint: search_database()

    Allows the user to search records in the database using terms of the 'terms'
    list. The 'terms' list can be composed of:
        1. string(s) representation(s) of a valid unique identifiers (string(s))
        2. image name(s) (string(s))
        
    Note: a valid unique identifier string representation obeys the following
    (case insensitive) regular expression:
                    '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
                    '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

    Each element of the 'terms' list can be one of the two string types
    described above (i.e. one can mix and match string types in the 'terms'
    list).

    This endpoint returns the records (Representations) of each term found.
    If a term is invalid or does not match any record in the database, it is
    ignored / skipped.

    Parameters:
    - search_terms : list of string representation of a valid unique identifiers
                        or image name  (list of strings)

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
    # Searches for the matching records, then converts them to the appropriate
    # output response model
    found_records = glb.rep_db.search(search_terms, get_index=False)
    output_objs   = glb.rep_db.__records2resp_model__(found_records)

    return output_objs

# ------------------------------------------------------------------------------

@fr_router.post("/utility/search_database_by_tag/", response_model=List[RepsInfoOutput])
async def search_database_by_tag(search_tag : str  = Query(None, description="Name tag to be searched (string)"),
                                 ignore_case: bool = Query(False, description="Toggle between case sensitive and case insensitive search (boolean)")):
    """
    API endpoint: search_database_by_tag()

    Allows the user to search the database using a name tag. Returns all
    database entries containing the name tag provided. This search can be either
    case sensitive or case insensitive based on the value of 'ignore_case'.

    Parameters:
    - target_tag : name tag to be searched (string)

    - ignore_case: toggle between case sensitive and case insensitive search
                    (boolean)

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
    # Searches for the matching records, then converts them to the appropriate
    # output response model
    found_records = glb.rep_db.search_by_tag(search_tag,
                                    ignore_case=ignore_case, get_index=False)
    output_objs   = glb.rep_db.__records2resp_model__(found_records)

    return output_objs

# ------------------------------------------------------------------------------

@fr_router.post("/utility/get_attribute_from_database")
async def get_attribute_from_database(
    atr           : AvailableRepProperties = Query(default_property, description="Representations' attribute (string)"),
    verifier_name : str                    = Query(default_verifier, description="Face verifier name (string)"),
    do_sort       : bool                   = Query(False           , description="Flag to perform sorting of the results (boolean)"),
    suppress_error: bool                   = Query(True            , description="Flag to suppress error, returning an empty list instead of raising an exception (boolean")):
    """
    API endpoint: get_attribute_from_database()

    Gets a specific attribute from all Representations in the database.
    
    The embeddings attribute (atr='embeddings') is a special case and works
    differently: what is returned depends on the the 'verifier_name' parameter.
    If a specific face verifier is chosen, then the embeddings for that verifier
    are returned as a list (of lists), and sorting is ignored regardless of
    'do_sort'.

    Alternatively, the string 'names' can be passed as a 'verifier_name'. In
    this case, only the names of ALL embeddings available to each Representation
    are returned (also as a list of lists).
    
    For example, consider a database with 2 Representations with the following
    available embeddings:
        Representation 1: ArcFace, OpenFace, FaceNet 
        Representation 2: ArcFace, VGG-Face

    The output of atr='embeddings' and verifier_name='names' would be:
        output = [[ArcFace, OpenFace, FaceNet], [ArcFace, VGG-Face]]

    Parameters:
    - propty        : Representations' property (string).

    - verifier_name : face verifier name (case sensitive) OR 'names'
                        (string, default: <default_verifier>).

    - do_sort       : flag to perform sorting of the results
                        (boolean, default: False).

    - suppress_error: flag to suppress error, returning an empty list instead
                        of raising an exception (boolean, default: True).

    Output:\n
        JSON-encoded list containing the chosen property from each
        Representation. The list is sorted if 'do_sort' is set to True. The list
        will be empty if the Representation database has a length of zero or if
        'suppress_error' is True and a non-existant property 'param' is chosen.
    """
    # Initializes the skip_sort flag (this is only used when the vector
    # embeddings are returned as it does not make any sense to sort them)
    skip_sort = False

    # Gets the desired attribute of each Representation in the database,
    # supressing errors if suppress_error is True
    attributes = glb.rep_db.get_attribute(atr, suppress_error=suppress_error)

    # If the 'embeddings' property is selected, processes it based on whether
    # the user wants a specific embedding or the embedding names
    if atr == 'embeddings':
        # If the user wants the embedding names
        if verifier_name == 'names':
            all_names = []
            for cur_atrbs in attributes:
                all_names.append([name for name in cur_atrbs.keys()])
            attributes = all_names
        
        # Otherwise, the user wants a specific embedding from the face verifier
        # 'verifier_name'
        else:
            outputs = []
            for cur_atrbs in attributes:
                try:
                    outputs.append([embd.tolist() for embd\
                                    in cur_atrbs[verifier_name]])
                except:
                    outputs.append([])
            attributes = outputs

            # Skips sorting regardless of the the 'do_sort' switch
            skip_sort  = True
    
    # Otherwise, the chosen parameter is not 'embeddings' so ignore this section
    else:
        pass # do nothing

    # Sorts the attributes if 'do_sort' flag is set to True, skipping it if the
    # 'attributes' list is empty. The 'skip_sort' flag is only ever True if a
    # specific embedding is chosen.
    if do_sort and len(attributes) > 0 and not skip_sort:
        attributes.sort()

    return attributes

# ------------------------------------------------------------------------------

@fr_router.post("/utility/get_groups")
async def get_groups():
    attributes = glb.rep_db.get_attribute('group_no')
    attributes = np.unique(attributes)  # only keep unique groups
    attributes = sorted(attributes, key=lambda x: int(x))   # and sort them
    return [str(x) for x in attributes]

# ------------------------------------------------------------------------------

@fr_router.post("/utility/update_record/", response_model=RepsInfoOutput)
async def update_record(
    term          : str = Query(None, description="String Representation of valid unique identifier (including dashes '-') or a valid image name (string)"),
    new_unique_id : str = Query(None, description="String Representation of new unique identifier (including dashes '-') (string)"),
    new_image_name: str = Query(None, description="Image name (including extension) (string)"),
    new_image_fp  : str = Query(None, description="Image full path (string)"),
    new_group_no  : int = Query(None, description="Group number (-1 means 'groupless' / no group) (integer)"),
    new_name_tag  : str = Query(None, description="New name tag (string)"),
    new_region    : List[int] = Query(None, description="Face region in the original image (list of 4 integers)"),
    new_embeddings: dict = Body(None, description="Dictionary containing verifier name and embedding pairs (dictionary)")):
    """
    API endpoint: update_record()

    Allows the user to update / edit a single record (Representation) in the
    database of a specific image (Representation) by providing either a string
    representation of a valid unique identifier OR a valid image name.
        
    Note: a VALID UUID string representation obeys the following (case
    insensitive) regular expression:
                    '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
                    '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

    If the term provided is invalid (or does not have any matching record), then
    nothing will be updated.
        
    All optional parameters are attributes of a record (Representation) that
    can be modified. Only attributes that will be modified should be passed
    as arguments to this function. The rest should be kept at their default
    value of None.

    Important note: this endpoint does not automatically determine the image's
    full path from the image's name. If the image name attribute is modified,
    make sure to modify the image full path attribute with the appropriate path
    as well, or this might lead to other errors!

    Parameters:
    - term      : UUID, string representation of a valid UUID or image name
                    [UUID / string].

    - unique_id : record's (Representation's) unique id [UUID].

    - image_name: record's (Representation's) image name [string].
            
    - image_fp  : record's (Representation's) image full path [string].
            
    - group_no  : record's (Representation's) group number [integer].
            
    - name_tag  : record's (Representation's) name tag [string].
            
    - region    : record's (Representation's) region [list of 4 integers].
            
    - embeddings: record's (Representation's) embeddings [dictionary].

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
    # If 'new_embeddings' is an empty dictionary OR is not a dictionary, set it
    # to None so that it does not update the record by mistake
    if isinstance(new_embeddings, dict):
        if len(new_embeddings) == 0:
            new_embeddings = None
        else:
            pass # do nothing
    else:
        new_embeddings = None

    # Gets the updated record and converts it to the appropriate response model.
    # Since it is converted to a list (with 1 element), just get the first
    # element.
    updated_record = glb.rep_db.update_record(term, unique_id=new_unique_id,
                            image_name=new_image_name, image_fp=new_image_fp,
                            group_no=new_group_no, name_tag=new_name_tag,
                            region=new_region, embeddings=new_embeddings)
    output_obj     = glb.rep_db.__records2resp_model__([updated_record])

    return output_obj[0]

# ------------------------------------------------------------------------------

@fr_router.post("/utility/rename_records_by_tag/", response_model=List[RepsInfoOutput])
async def rename_records_by_tag(
    old_tag      : str  = Query(None, description="Old name tag (used in search) (string)"),
    new_tag      : str  = Query(None, description="New name tag (string)"),
    ignore_case  : bool = Query(False, description="Toggle between case sensitive and case insensitive search (boolean)"),
    blank_strings: List[str] = Query(['""', "''", "--"], description="List of string that will be considered blank / null (list of strings)")):
    """
    API endpoint: rename_records_by_tag()

    Allows the user to rename all database entries related to a specific tag
    with a new one. This operation is similar to a search_database_by_tag()
    followed by updating the name tag of each result with the new tag.

    Parameters:
    - old_tag      : old name tag - this will be used to search for the database
                      entries (string).

    - new_tag      : new name tag (string).

    - ignore_case  : toggle between case sensitive and case insensitive search
                      (boolean).

    - blank_strings: list of strings which are considered blank ones. If
                      'old_tag' and/or 'new_tag' matches any of these strings
                      then they are treated as '' inside the function. This gets
                      around the limitation of trying to send an empty / blank
                      string (default=['""', "''", "--"]) (list of strings).

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
    # Converts the 'old_tag' and/or 'new_tag' to blank / empty ones
    # (respectively) if they match any of the strings in the 'blank_strings'
    # list
    if old_tag in blank_strings:
        old_tag = ''
    else:
        pass # do nothing

    if new_tag in blank_strings:
        new_tag = ''
    else:
        pass # do nothing

    # Renames records and converts them to the appropriate response model
    renamed_records = glb.rep_db.rename_records_by_tag(old_tag, new_tag,
                                                       ignore_case=ignore_case)
    output_objs     = glb.rep_db.__records2resp_model__(renamed_records)

    return output_objs

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
    try:
        # Obtains the database's full path
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
        # On exception, create output message (with exception) and sets the
        # status as 1 (something went wrong)
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
        # status as 1 (something went wrong)
        output_msg = f"Unable to save database. Reason: {excpt}"
        status     = True

    return {"message":output_msg, "status":status}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/view_by_group_no")
async def view_by_group_no(target_group_no: int = Query(None, description="Target group number (>= -1) [integer]"),
    return_type  : str = Query('records', description="Desired return type. Options: 'records' or 'image' [str]"),
    ncols        : int = Query(4, description="Number of columns in the output image [integer]"),
    figsize      : Tuple[int, int] = Query((15, 15), description="Figure size in inches [tuple of 2 integers]"),
    color        : str = Query('black', description="Text color [str]"),
    add_separator: bool = Query(False, description="Adds a caption acting as a separator [boolean]")):
    """
    API endpoint: view_by_group_no()

    Views all information of all Representations belonging to a group number 
    specified by 'target_group_no'. Raises a Value Error if the 'return_type'
    provided is neither 'records' nor 'image'.

    Parameters:
    - target_group_no : desired group / cluster number [integer].

    - return_type : determines if the output of this endpoint will be
         JSON-encoded structures of the Representations ('reps') OR an image
         ('image') (default='records') [string].

    - ncols : number of columns in the plot image. Ignored if return_type='reps'
         (default=4) [integer].
    
    - figsize : output figure size in inches. Ignored if return_type='reps'
         (default=(15, 15)) [tuple of 2 integers].

    - color : text color (default='black') [string].

    - add_separator : toggles between adding a caption acting as a separator or
         not (default=False) [boolean].

    Output:\n
        - If return_type='records':
            JSON-encoded structure containing all attributes of each
            Representation in the database belonging to the desired group.

        - If return_type='image':
            Plot of all images in the database belonging to the desired group.
    """
    # Obtain all records belonging to the specified group number
    recs_found = glb.rep_db.view_by_group_no(target_group_no, print_info=False)

    # Returns the output as either an image or as Representations
    if return_type == 'image':
        # Obtains the subplot figure, seeks to the beginning (just to be safe)
        # and returns the response as a png image
        img_file = show_cluster_results(target_group_no, glb.rep_db,
                                    ncols=ncols, figsize=figsize, color=color,
                                    add_separator=add_separator)
        img_file.seek(0)

        output_obj = Response(content=img_file.read(), media_type="image/png")
    
    elif return_type == 'records':
        # Alternatively, converts the records (Representations) to the
        # appropriate response model
        output_obj = glb.rep_db.__records2resp_model__(recs_found)

    else:
        raise ValueError("Return type should be either 'image' or 'records'!")

    return output_obj

# ------------------------------------------------------------------------------

@fr_router.post("/utility/remove_from_group")
async def remove_from_group(files  : List[str],
                            img_dir: str = Query(glb.IMG_DIR, description="Full path to image directory (string)")):
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
            for i in range(0, glb.rep_db.size):
                # If the unique identifier of the current Representation
                # matches the target identifier, remove its group
                if glb.rep_db.reps[i].unique_id == UUID(f):
                    glb.rep_db.reps[i].group_no = -1
                    removed_count += 1 # increments removed counter
                else:
                    pass # do nothing

        elif os.path.isfile(os.path.join(img_dir, f)): # string is a valid file
            for i in range(0, glb.rep_db.size):
                # If this Representation's full path matches the target
                # file's full path, remove its group
                if glb.rep_db.reps[i].image_name == f:
                    glb.rep_db.reps[i].group_no = -1
                    removed_count += 1 # increments removed counter
                else:
                    pass # do nothing

        else: # string is not a valid uuid nor file
            skipped_count += 1

    return {'removed':removed_count, 'skipped':skipped_count}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/edit_tag_by_group_no")
async def edit_tag_by_group_no(target_group_no: int = Query(None, description="Target group number (>= -1) [integer]"),
                               new_name_tag   : str = Query(None, description="New name tag [string]")):
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
    # Edits the records and returns the number of edited records
    num_records_edited = glb.rep_db.edit_tag_by_group_no(target_group_no,
                                                         new_name_tag)

    return {'num_records_edited':num_records_edited}

# ------------------------------------------------------------------------------

@fr_router.post("/create_database/from_directory")
async def create_database_from_directory(cdb_params: CreateDatabaseParams,
    image_dir   : Optional[str]  = Query(glb_img_dir, description="Full path to directory containing images (string)"),
    force_create: Optional[bool] = Query(False      , description="Flag to force database creation even if one already exists, overwritting the old one (boolean)")):
    """
    API endpoint: create_database_from_directory()

    Creates a database (RepDatabase object) from a directory. The directory is
    expected to have image files in any of the following formats: .jpg, .png,
    .npy.

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
        global glb_img_dir
        output_msg += 'Image dir is None, does not exist or is not a '\
                   +  'directory. Using default directory instead.\n'
        image_dir  = glb_img_dir

    # Database exists (and has at least one element) and force create is False
    if glb.rep_db.size > 0 and not force_create:
        # Do nothing, but set message
        output_msg += 'Database exists (and force create is False). '\
                   +  'Skipping database creation.\n'

    elif glb.rep_db.size == 0 or force_create:
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

    # Modifies the database change flag (and sorts the database if there is at
    # least 1 element)
    if glb.rep_db.size == 1:
        glb.db_changed = True

    elif glb.rep_db.size > 1:
        # Sorts database and sets changed flag to True
        glb.rep_db.sort_database('image_name')
        glb.db_changed = True

    else:
        glb.db_changed = False

    return {'n_records':glb.rep_db.size, 'message':output_msg}

# ------------------------------------------------------------------------------

@fr_router.post("/create_database/from_zip")
async def create_database_from_zip(myfile: UploadFile,
    params      : CreateDatabaseParams = Depends(),
    image_dir   : Optional[str]  = Query(glb.IMG_DIR, description="Full path to directory containing images (string)"),
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

    # Database exists (and has at least one element) and force create is False
    if glb.rep_db.size > 0 and not force_create:
        # Do nothing, but set message
        output_msg += 'Database exists (and force create is False). '\
                   +  'Skipping database creation.\n'

    elif glb.rep_db.size == 0 or force_create:
        # Initialize dont_skip flag as True
        dont_skip = True

        # Extract zip files
        output_msg += 'Extracting images in zip:'

        try:
            # Process the zip file containing the image files
            new_names = process_image_zip_file(myfile, image_dir,
                                               auto_rename=auto_rename)
            output_msg += ' success! '

        except Exception as excpt:
            dont_skip   = False
            output_msg += f' failed (reason: {excpt}).'

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

            # Fixes unique ids of renamed images (ensuring that the unique id in
            # the image name matches the Representation's unique id and
            # vice-versa)
            fix_uid_of_renamed_imgs(new_names)

            output_msg += 'success!\n'
        else:
            output_msg += 'Skipping database creation.\n'

    else:
        raise AssertionError('[create_database] Database should have a ' +  
                             'length of 0 or more - this should not happen!')

    # Modifies the database change flag (and sorts the database if there is at
    # least 1 element)
    if glb.rep_db.size == 1:
        glb.db_changed = True

    elif glb.rep_db.size > 1:
        # Sorts database and sets changed flag to True
        glb.rep_db.sort_database('image_name')
        glb.db_changed = True

    else:
        glb.db_changed = False

    return {'length':glb.rep_db.size, 'message':output_msg}

# ------------------------------------------------------------------------------

@fr_router.post("/verify/no_upload", response_model=List[List[VerificationMatch]])
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
        For each file, returns a list of JSON-encoded structure with the
        following attributes:
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

        Note that the final output for multiple files, each with multiple
        matches, will be a list of list of JSON-encoded structures.
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
        result = get_matches_from_similarity(similarity_obj, glb.rep_db)
        #result_df = pd.DataFrame(result)
        #print(result_df)
        #verification_results = result_df.to_dict('records')
        verification_results.append(result)
        # TODO: check for multiple file, not working at the moment!

    return verification_results
    
# ------------------------------------------------------------------------------

@fr_router.post("/verify/with_upload", response_model=List[List[VerificationMatch]])
async def verify_with_upload(files: List[UploadFile],
    params     : VerificationParams = Depends(),
    img_dir    : str                = Query(glb_img_dir, description="Full path to image directory (string)"),
    save_as    : ImageSaveTypes     = Query(default_image_save_type, description="File type which uploaded images should be saved as (string)"),
    overwrite  : bool               = Query(False, description="Flag to indicate if an uploaded image with the same name as an existing one in the server should be saved and replace it (boolean)"),
    auto_rename: bool               = Query(True, description="Flag to force auto renaming of images in the zip file with (boolean)"),
    auto_tag   : bool               = Query(True, description="Flag to automatically generate name tags based on verification results (boolean)"),
    auto_group : bool               = Query(True, description="Flag to automatically group image based on verification results (boolean)")):
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

    - img_dir     : full path to image directory (string).

    - save_as     : file type which uploaded images should be saved as (string).

    - overwrite   : flag to indicate if an uploaded image with the same name as
                    an existing one in the server should be saved and replace it
                    (boolean, default: False).

    - auto_rename : flag to force auto renaming of images in the zip file with
                    names that match images already in the image directory
                    (boolean, default: True).

    - auto_tag    : toggles automatic name tag generation of the uploaded image
                    based on the face verification results
                    (boolean, default: True).

    - auto_group  : toggles automatic grouping of the uploaded image based on
                    the face verification results (boolean, default: True).

    Output:\n
        For each file, returns a list of JSON-encoded structure with the
        following attributes:
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

        Note that the final output for multiple files, each with multiple
        matches, will be a list of list of JSON-encoded structures.
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
        cur_emb = embeddings[params.verifier_name]

        # Calculates the similarity between the current embedding and all
        # embeddings from the database
        similarity_obj = calc_similarity(cur_emb, dtb_embs,
                                         metric=params.metric,
                                         model_name=params.verifier_name,
                                         threshold=params.threshold)

        # Gets all matches based on the similarity object and append the result
        # to the results list
        result = get_matches_from_similarity(similarity_obj, glb.rep_db)

        # Chooses a tag automatically from the best match if it has a distance / 
        # similarity of <=0.75*threshold and if 'auto_tag' is True
        if auto_tag and len(result) > 0:
            if similarity_obj['distances'][0]\
                <= 0.75 * similarity_obj['threshold']:
                tag = result[0].name_tag
            else:
                tag = ''
        else:
            tag = ''

        # Automatically determines the image's group based on the best match if
        # it has a distance / similarity of <=0.75*threshold and if 'auto_group'
        # is True
        if auto_group and len(result) > 0:
            if similarity_obj['distances'][0]\
                <= 0.75 * similarity_obj['threshold']:
                group_no = result[0].group_no
            else:
                group_no = -1
        else:
            group_no = -1

        # Creates a representation object for this image and adds it to the
        # database
        new_rep = create_new_representation(img_fp, region, embeddings,
                                            group_no=group_no, tag=tag, uid=uid)
        glb.rep_db.reps.append(new_rep)
        glb.db_changed = True

        # Stores the verification result
        verification_results.append(result)

    return verification_results

# ------------------------------------------------------------------------------

@fr_router.post("/verify/existing_file", response_model=List[List[VerificationMatch]])
async def verify_existing_file(files: List[str],
            params : VerificationParams = Depends(),
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
        For each file, returns a list of JSON-encoded structure with the
        following attributes:
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

        Note that the final output for multiple files, each with multiple
        matches, will be a list of list of JSON-encoded structures.
    """
    # Initializes results list and obtains the relevant embeddings from the 
    # Representation database
    verification_results = []
    dtb_embs = get_embeddings_as_array(glb.rep_db, params.verifier_name)

    # Gets the unique ids in the database depending on the database's size
    if glb.rep_db.size == 0:   # no representations
        all_uids = []

    elif glb.rep_db.size > 1:  # one or many representations
        # Loops through each Representation in the database and gets the unique
        # id tag
        all_uids = []
        for rep in glb.rep_db.reps:
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
            cur_emb = glb.rep_db.reps[index].embeddings[params.verifier_name]

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
        result = get_matches_from_similarity(similarity_obj, glb.rep_db)
        verification_results.append(result)

    return verification_results
    
# ------------------------------------------------------------------------------