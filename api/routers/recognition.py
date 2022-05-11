# ==============================================================================
#                              RECOGNITION API METHODS
# ==============================================================================
import os
import cv2
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy                 as np
import api.global_variables  as glb

from uuid                    import uuid4
from typing                  import List, Optional
from fastapi                 import APIRouter, UploadFile, Depends, Query
from IFR.api                 import init_load_verifiers, init_load_detectors,\
                               get_embeddings_as_array, process_image_zip_file,\
                               database_is_empty, get_matches_from_similarity,\
                               all_tables_exist, process_faces_from_dir,\
                               load_database, facerep_set_groupno_done,\
                               start_session, people_clean_without_repps
from IFR.classes             import *
from IFR.functions           import ensure_dirs_exist, calc_embeddings,\
                                    calc_similarity, do_face_detection,\
                                    discard_small_regions
from fastapi.responses       import Response

from matplotlib                      import image          as mpimg
from deepface.DeepFace               import build_model    as build_verifier
from deepface.detectors.FaceDetector import build_model    as build_detector

from sqlalchemy import select, update, insert

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

@fr_router.post("/debug/inspect_globals")
async def inspect_globals(print2console: bool = Query(True, description="Toggles if global variables should be printed to the server's console [boolean]")):
    
    # Obtains all current directory paths
    directories = [glb.API_DIR    , glb.DATA_DIR   , glb.IMG_DIR, glb.RDB_DIR,
                   glb.SVD_MDL_DIR, glb.SVD_VRF_DIR, glb.SVD_DTC_DIR]

    # Prints the path variables along with their names
    if print2console:
        dir_names = ['API root dir', 'Data dir', 'Image dir',
                     'Rep. database dir', 'Saved models dir',
                     'Saved face verifiers dir', 'Saved face detectors dir']

        print("  > Paths:")
        for dir, name in zip(directories, dir_names):
            print(name.ljust(24) + ':', dir)
        print('')
        
    # Prints all face detectors, indicating if they have been loaded (or build
    # from scratch) successfully
    if print2console:
        print("  > All face detectors:")
        for i, name in enumerate(glb.detector_names):
            if name == '':
                continue
            print(f'{i+1}'.ljust(2) + ':', name, end=' ')

            try:
                glb.models[name]
                print('[loaded]')
            except:
                print('[not loaded]')
        print('')

    # Prints all face verifiers, indicating if they have been loaded (or build
    # from scratch) successfully
    if print2console:
        print("  > All face verifiers:")
        for i, name in enumerate(glb.verifier_names):
            if name == '':
                continue
            print(f'{i+1}'.ljust(2) + ':', name, end=' ')

            try:
                glb.models[name]
                print('[loaded]')
            except:
                print('[not loaded]')
        print('')

    # Prints all other global variables
    if print2console:
        print("  > Other variables:")
        print("Models".ljust(21)                + ':', glb.models)
        print("Rep. database".ljust(21)         + ':', glb.rep_db)
        print("Database changed".ljust(21)      + ':', glb.db_changed)
        print("SQLite database:".ljust(21)      + ':', glb.SQLITE_DB)
        print("SQLite database path:".ljust(21) + ':', glb.SQLITE_DB_FP)
        print("SQL alchemy engine:".ljust(21)   + ':', glb.sqla_engine)

    return {'dirs':directories, 'dir_names':dir_names,
            'detector_names':glb.detector_names,
            'verifier_names':glb.verifier_names,
            'model_names':list(glb.models.keys()), 'rep_db':glb.rep_db,
            'db_changed':glb.db_changed, 'sqlite_db_name':glb.SQLITE_DB,
            'sqlite_db_fp':glb.SQLITE_DB_FP, 'sqla_engine':glb.sqla_engine}

# ------------------------------------------------------------------------------

@fr_router.post("/debug/reset_server")
async def reset_server():
    """
    API endpoint: reset_server()

    Allows the user to restart the server without manually requiring to shut it
    down and start it up.

    Output:\n
        JSON-encoded dictionary with the following attributes:
            1. message: message stating the server has been restarted (string)
    """
    print('\n!!!!!!!!!!!!!!!! RESTARTING THE SERVER !!!!!!!!!!!!!!!!')
    print('\n ======== Starting initialization process ======== \n')

    # Directories & paths initialization
    print('  -> Resetting global variables:')
    glb.API_DIR = os.path.join(os.path.dirname(os.path.realpath("__file__")),
                               'api')
    print('[PATH]'.ljust(12), 'API_DIR'.ljust(12),
         f': reset! ({glb.API_DIR})')
    
    glb.DATA_DIR     = os.path.join(glb.API_DIR    , 'data')
    print('[PATH]'.ljust(12), 'DATA_DIR'.ljust(12),
         f': reset! ({glb.DATA_DIR})')
    
    glb.IMG_DIR      = os.path.join(glb.DATA_DIR  , 'img')
    print('[PATH]'.ljust(12), 'IMG_DIR'.ljust(12),
         f': reset! ({glb.IMG_DIR})')
    
    glb.RDB_DIR      = os.path.join(glb.DATA_DIR  , 'database')
    print('[PATH]'.ljust(12), 'RDB_DIR'.ljust(12),
         f': reset! ({glb.RDB_DIR})')
    
    glb.SVD_MDL_DIR  = os.path.join(glb.API_DIR       , 'saved_models')
    print('[PATH]'.ljust(12), 'SVD_MDL_DIR'.ljust(12),
         f': reset! ({glb.SVD_MDL_DIR})')
    
    glb.SVD_VRF_DIR  = os.path.join(glb.SVD_MDL_DIR   , 'verifiers')
    print('[PATH]'.ljust(12), 'SVD_VRF_DIR'.ljust(12),
         f': reset! ({glb.SVD_VRF_DIR})')

    glb.SVD_DTC_DIR  = os.path.join(glb.SVD_MDL_DIR   , 'detectors')
    print('[PATH]'.ljust(12), 'SVD_VRF_DIR'.ljust(12),
         f': reset! ({glb.SVD_DTC_DIR})')
    print('')

    glb.models       = {}
    print('[MODELS]'.ljust(12), 'models'.ljust(12),
         f': reset! ({glb.models})')
    print('')

    glb.SQLITE_DB    = 'loki.sqlite'
    print('[SQLALCH]'.ljust(12), 'SQLITE_DB'.ljust(12),
         f': reset! ({glb.SQLITE_DB})')

    glb.SQLITE_DB_FP = os.path.join('api/data/database', glb.SQLITE_DB)
    print('[SQLALCH]'.ljust(12), 'SQLITE_DB_FP'.ljust(12),
         f': reset! ({glb.SQLITE_DB_FP})')

    glb.sqla_engine  = None
    print('[SQLALCH]'.ljust(12), 'sqla_engine'.ljust(12),
         f': reset! ({glb.sqla_engine})')

    glb.sqla_session = None
    print('[SQLALCH]'.ljust(12), 'sqla_session'.ljust(12),
         f': reset! ({glb.sqla_session})')
    print('')

    # Directories & paths initialization
    print('  -> Directory creation:')
    directory_list = [glb.API_DIR, glb.DATA_DIR, glb.IMG_DIR, glb.RDB_DIR,
                      glb.SVD_MDL_DIR, glb.SVD_VRF_DIR, glb.SVD_DTC_DIR]
    ensure_dirs_exist(directory_list, verbose=True)
    print('')

    # Tries to load a database if it exists. If not, create a new one.
    print('  -> Loading / creating database:', end='')
    glb.sqla_engine = load_database(glb.SQLITE_DB_FP)
    print('success!\n')

    # Loads (or creates) the session. Also commits once to create table
    # definitions if required.
    print('  -> Loading / creating session:', end='')
    glb.sqla_session = start_session(glb.sqla_engine)
    glb.sqla_session.commit()                   # create table definitions
    print('success!\n')
    
    # Loads (or creates) all face verifiers
    print('  -> Loading / creating face verifiers:')
    glb.models = init_load_verifiers(glb.verifier_names, glb.SVD_VRF_DIR)
    print('')

    # Loads (or creates) all face detectors
    print('  -> Loading / creating face detectors:')
    glb.models = init_load_detectors(glb.detector_names, glb.SVD_VRF_DIR,
                                     models=glb.models)
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

@fr_router.post("/utility/view_tables")
async def view_tables(
    table_name : AvailableTables = Query(AvailableTables.FACEREP, description="Name of desired table (string)")):
    """
    API endpoint: view_tables()

    Returns all information corresponding to the selected table. The table must
    exist in the database, otherwise an assertion error is raised.

    Parameter:
    - table_name : name of the desired table [string].

    Output:\n
        List of JSON-encoded structures containing all of the attributes of each
        record in the selected table.
    """
    # Initialize output object
    output_obj = []

    # Prints the appropriate information if the selected table is
    # 'representation'
    if   table_name.lower() == AvailableTables.FACEREP:
        query = glb.sqla_session.query(FaceRep)
        for rep  in query.all():
            output_obj.append(FaceRepOutput(
                id              = rep.id,
                person_id       = rep.person_id,
                image_name_orig = rep.image_name_orig,
                image_fp_orig   = rep.image_fp_orig,
                group_no        = rep.group_no,
                region          = rep.region,
                embeddings      = [name for name in rep.embeddings.keys()]
            ))

    # Prints the appropriate information if the selected table is 'person'
    elif table_name.lower() == AvailableTables.PERSON:
        query = glb.sqla_session.query(Person)
        for prsn in query.all():
            output_obj.append(PersonTableOutput(
                id       = prsn.id,
                name     = prsn.name,
                group_no = prsn.group_no,
                note     = prsn.note
            ))

    # Prints the appropriate information if the selected table is 'proc_files'
    elif table_name.lower() == AvailableTables.PROCFILES:
        query = glb.sqla_session.query(ProcessedFiles)
        for fprc in query.all():
            output_obj.append(ProcessedFilesOutput(
                id       = fprc.id,
                filename = fprc.filename,
                filepath = fprc.filepath,
                filesize = fprc.filesize
            ))

    # Raises an assertion error because the selected table does not exist in the
    # database
    else:
        raise AssertionError("Table name should be either 'person', "
                           + "'representation' or 'proc_files'!")

    return output_obj

# ------------------------------------------------------------------------------

@fr_router.post("/database/clear")
async def database_clear_api():
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
    # Remove SQlite file and recreate again
    os.remove(glb.SQLITE_DB_FP)
    glb.sqla_engine  = load_database(glb.SQLITE_DB_FP)
    glb.sqla_session = start_session(glb.sqla_engine)
    glb.sqla_session.commit()

    return {"message": "Database has been cleared."}

# ------------------------------------------------------------------------------

# @fr_router.post("/utility/search_database/", response_model=List[RepsInfoOutput])
# async def search_database(search_terms: List[str] = Query(None, description="Name tag to be searched (string)")):
#     """
#     API endpoint: search_database()

#     Allows the user to search records in the database using terms of the 'terms'
#     list. The 'terms' list can be composed of:
#         1. string(s) representation(s) of a valid unique identifiers (string(s))
#         2. image name(s) (string(s))
        
#     Note: a valid unique identifier string representation obeys the following
#     (case insensitive) regular expression:
#                     '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
#                     '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

#     Each element of the 'terms' list can be one of the two string types
#     described above (i.e. one can mix and match string types in the 'terms'
#     list).

#     This endpoint returns the records (Representations) of each term found.
#     If a term is invalid or does not match any record in the database, it is
#     ignored / skipped.

#     Parameters:
#     - search_terms : list of string representation of a valid unique identifiers
#                         or image name  (list of strings)

#     Output:\n
#         JSON-encoded Representation structure with the following attributes:
#             1. unique_id : unique identifier [UUID]
#             2. image_name: image name [string]
#             3. group_no  : group number [integer]
#             4. name_tag  : name tag [string]
#             5. image_fp  : image full path [string]
#             6. region    : the region is a 4 element list [list of integers]
#             7. embedding : list of face verifier names for which this
#                             Representation has embeddings for [list of strings]
#     """
#     # Searches for the matching records, then converts them to the appropriate
#     # output response model
#     found_records = glb.rep_db.search(search_terms, get_index=False)
#     output_objs   = glb.rep_db.__records2resp_model__(found_records)

#     return output_objs

# ------------------------------------------------------------------------------

# @fr_router.post("/utility/search_database_by_tag/", response_model=List[RepsInfoOutput])
# async def search_database_by_tag(search_tag : str  = Query(None, description="Name tag to be searched (string)"),
#                                  ignore_case: bool = Query(False, description="Toggle between case sensitive and case insensitive search (boolean)")):
#     """
#     API endpoint: search_database_by_tag()

#     Allows the user to search the database using a name tag. Returns all
#     database entries containing the name tag provided. This search can be either
#     case sensitive or case insensitive based on the value of 'ignore_case'.

#     Parameters:
#     - target_tag : name tag to be searched (string)

#     - ignore_case: toggle between case sensitive and case insensitive search
#                     (boolean)

#     Output:\n
#         JSON-encoded Representation structure with the following attributes:
#             1. unique_id : unique identifier [UUID]
#             2. image_name: image name [string]
#             3. group_no  : group number [integer]
#             4. name_tag  : name tag [string]
#             5. image_fp  : image full path [string]
#             6. region    : the region is a 4 element list [list of integers]
#             7. embedding : list of face verifier names for which this
#                             Representation has embeddings for [list of strings]
#     """
#     # Searches for the matching records, then converts them to the appropriate
#     # output response model
#     found_records = glb.rep_db.search_by_tag(search_tag,
#                                     ignore_case=ignore_case, get_index=False)
#     output_objs   = glb.rep_db.__records2resp_model__(found_records)

#     return output_objs

# ------------------------------------------------------------------------------

# @fr_router.post("/utility/get_attribute_from_database")
# async def get_attribute_from_database(
#     atr           : AvailableRepProperties = Query(default_property, description="Representations' attribute (string)"),
#     verifier_name : str                    = Query(default_verifier, description="Face verifier name (string)"),
#     do_sort       : bool                   = Query(False           , description="Flag to perform sorting of the results (boolean)"),
#     suppress_error: bool                   = Query(True            , description="Flag to suppress error, returning an empty list instead of raising an exception (boolean")):
#     """
#     API endpoint: get_attribute_from_database()

#     Gets a specific attribute from all Representations in the database.
    
#     The embeddings attribute (atr='embeddings') is a special case and works
#     differently: what is returned depends on the the 'verifier_name' parameter.
#     If a specific face verifier is chosen, then the embeddings for that verifier
#     are returned as a list (of lists), and sorting is ignored regardless of
#     'do_sort'.

#     Alternatively, the string 'names' can be passed as a 'verifier_name'. In
#     this case, only the names of ALL embeddings available to each Representation
#     are returned (also as a list of lists).
    
#     For example, consider a database with 2 Representations with the following
#     available embeddings:
#         Representation 1: ArcFace, OpenFace, FaceNet 
#         Representation 2: ArcFace, VGG-Face

#     The output of atr='embeddings' and verifier_name='names' would be:
#         output = [[ArcFace, OpenFace, FaceNet], [ArcFace, VGG-Face]]

#     Parameters:
#     - propty        : Representations' property (string).

#     - verifier_name : face verifier name (case sensitive) OR 'names'
#                         (string, default: <default_verifier>).

#     - do_sort       : flag to perform sorting of the results
#                         (boolean, default: False).

#     - suppress_error: flag to suppress error, returning an empty list instead
#                         of raising an exception (boolean, default: True).

#     Output:\n
#         JSON-encoded list containing the chosen property from each
#         Representation. The list is sorted if 'do_sort' is set to True. The list
#         will be empty if the Representation database has a length of zero or if
#         'suppress_error' is True and a non-existant property 'param' is chosen.
#     """
#     # Initializes the skip_sort flag (this is only used when the vector
#     # embeddings are returned as it does not make any sense to sort them)
#     skip_sort = False

#     # Gets the desired attribute of each Representation in the database,
#     # supressing errors if suppress_error is True
#     attributes = glb.rep_db.get_attribute(atr, suppress_error=suppress_error)

#     # If the 'embeddings' property is selected, processes it based on whether
#     # the user wants a specific embedding or the embedding names
#     if atr == 'embeddings':
#         # If the user wants the embedding names
#         if verifier_name == 'names':
#             all_names = []
#             for cur_atrbs in attributes:
#                 all_names.append([name for name in cur_atrbs.keys()])
#             attributes = all_names
        
#         # Otherwise, the user wants a specific embedding from the face verifier
#         # 'verifier_name'
#         else:
#             outputs = []
#             for cur_atrbs in attributes:
#                 try:
#                     outputs.append([embd.tolist() for embd\
#                                     in cur_atrbs[verifier_name]])
#                 except:
#                     outputs.append([])
#             attributes = outputs

#             # Skips sorting regardless of the the 'do_sort' switch
#             skip_sort  = True
    
#     # Otherwise, the chosen parameter is not 'embeddings' so ignore this section
#     else:
#         pass # do nothing

#     # Sorts the attributes if 'do_sort' flag is set to True, skipping it if the
#     # 'attributes' list is empty. The 'skip_sort' flag is only ever True if a
#     # specific embedding is chosen.
#     if do_sort and len(attributes) > 0 and not skip_sort:
#         attributes.sort()

#     return attributes

# ------------------------------------------------------------------------------

@fr_router.post("/people/list")
async def people_list():
    # attributes = glb.rep_db.get_attribute('group_no')
    # attributes = np.unique(attributes)  # only keep unique groups
    # attributes = sorted(attributes, key=lambda x: int(x))   # and sort them
    query = select(Person.id, Person.name, Person.note)
    result = glb.sqla_session.execute(query)
    return result.fetchall()

# ------------------------------------------------------------------------------

# @fr_router.post("/utility/update_record/", response_model=RepsInfoOutput)
# async def update_record(
#     term          : str = Query(None, description="String Representation of valid unique identifier (including dashes '-') or a valid image name (string)"),
#     new_unique_id : str = Query(None, description="String Representation of new unique identifier (including dashes '-') (string)"),
#     new_image_name: str = Query(None, description="Image name (including extension) (string)"),
#     new_image_fp  : str = Query(None, description="Image full path (string)"),
#     new_group_no  : int = Query(None, description="Group number (-1 means 'groupless' / no group) (integer)"),
#     new_name_tag  : str = Query(None, description="New name tag (string)"),
#     new_region    : List[int] = Query(None, description="Face region in the original image (list of 4 integers)"),
#     new_embeddings: dict = Body(None, description="Dictionary containing verifier name and embedding pairs (dictionary)")):
#     """
#     API endpoint: update_record()

#     Allows the user to update / edit a single record (Representation) in the
#     database of a specific image (Representation) by providing either a string
#     representation of a valid unique identifier OR a valid image name.
        
#     Note: a VALID UUID string representation obeys the following (case
#     insensitive) regular expression:
#                     '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
#                     '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

#     If the term provided is invalid (or does not have any matching record), then
#     nothing will be updated.
        
#     All optional parameters are attributes of a record (Representation) that
#     can be modified. Only attributes that will be modified should be passed
#     as arguments to this function. The rest should be kept at their default
#     value of None.

#     Important note: this endpoint does not automatically determine the image's
#     full path from the image's name. If the image name attribute is modified,
#     make sure to modify the image full path attribute with the appropriate path
#     as well, or this might lead to other errors!

#     Parameters:
#     - term      : UUID, string representation of a valid UUID or image name
#                     [UUID / string].

#     - unique_id : record's (Representation's) unique id [UUID].

#     - image_name: record's (Representation's) image name [string].
            
#     - image_fp  : record's (Representation's) image full path [string].
            
#     - group_no  : record's (Representation's) group number [integer].
            
#     - name_tag  : record's (Representation's) name tag [string].
            
#     - region    : record's (Representation's) region [list of 4 integers].
            
#     - embeddings: record's (Representation's) embeddings [dictionary].

#     Output:\n
#         JSON-encoded Representation structure with the following attributes:
#             1. unique_id : unique identifier [UUID]
#             2. image_name: image name [string]
#             3. group_no  : group number [integer]
#             4. name_tag  : name tag [string]
#             5. image_fp  : image full path [string]
#             6. region    : the region is a 4 element list [list of integers]
#             7. embedding : list of face verifier names for which this
#                             Representation has embeddings for [list of strings]
#     """
#     # If 'new_embeddings' is an empty dictionary OR is not a dictionary, set it
#     # to None so that it does not update the record by mistake
#     if isinstance(new_embeddings, dict):
#         if len(new_embeddings) == 0:
#             new_embeddings = None
#         else:
#             pass # do nothing
#     else:
#         new_embeddings = None

#     # Gets the updated record and converts it to the appropriate response model.
#     # Since it is converted to a list (with 1 element), just get the first
#     # element.
#     updated_record = glb.rep_db.update_record(term, unique_id=new_unique_id,
#                             image_name=new_image_name, image_fp=new_image_fp,
#                             group_no=new_group_no, name_tag=new_name_tag,
#                             region=new_region, embeddings=new_embeddings)
#     output_obj     = glb.rep_db.__records2resp_model__([updated_record])

#     return output_obj[0]

# ------------------------------------------------------------------------------

# @fr_router.post("/utility/rename_records_by_tag/", response_model=List[RepsInfoOutput])
# async def rename_records_by_tag(
#     old_tag      : str  = Query(None, description="Old name tag (used in search) (string)"),
#     new_tag      : str  = Query(None, description="New name tag (string)"),
#     ignore_case  : bool = Query(False, description="Toggle between case sensitive and case insensitive search (boolean)"),
#     blank_strings: List[str] = Query(['""', "''", "--"], description="List of string that will be considered blank / null (list of strings)")):
#     """
#     API endpoint: rename_records_by_tag()

#     Allows the user to rename all database entries related to a specific tag
#     with a new one. This operation is similar to a search_database_by_tag()
#     followed by updating the name tag of each result with the new tag.

#     Parameters:
#     - old_tag      : old name tag - this will be used to search for the database
#                       entries (string).

#     - new_tag      : new name tag (string).

#     - ignore_case  : toggle between case sensitive and case insensitive search
#                       (boolean).

#     - blank_strings: list of strings which are considered blank ones. If
#                       'old_tag' and/or 'new_tag' matches any of these strings
#                       then they are treated as '' inside the function. This gets
#                       around the limitation of trying to send an empty / blank
#                       string (default=['""', "''", "--"]) (list of strings).

#     Output:\n
#         JSON-encoded Representation structure with the following attributes:
#             1. unique_id : unique identifier [UUID]
#             2. image_name: image name [string]
#             3. group_no  : group number [integer]
#             4. name_tag  : name tag [string]
#             5. image_fp  : image full path [string]
#             6. region    : the region is a 4 element list [list of integers]
#             7. embedding : list of face verifier names for which this
#                             Representation has embeddings for [list of strings]
#     """
#     # Converts the 'old_tag' and/or 'new_tag' to blank / empty ones
#     # (respectively) if they match any of the strings in the 'blank_strings'
#     # list
#     if old_tag in blank_strings:
#         old_tag = ''
#     else:
#         pass # do nothing

#     if new_tag in blank_strings:
#         new_tag = ''
#     else:
#         pass # do nothing

#     # Renames records and converts them to the appropriate response model
#     renamed_records = glb.rep_db.rename_records_by_tag(old_tag, new_tag,
#                                                        ignore_case=ignore_case)
#     output_objs     = glb.rep_db.__records2resp_model__(renamed_records)

#     return output_objs

# ------------------------------------------------------------------------------

# @fr_router.post("/utility/save_database")
# async def save_database(rdb_dir: str = Query(glb.RDB_DIR, description="Full path to Representation database directory (string)")):
#     """
#     API endpoint: save_database()

#     Allows the user to save the database. Sets the global 'database has been
#     modified' flag to False (i.e. database has not been modified).

#     Parameters:
#     - rdb_dir: full path to Representation database directory (string)

#     Output:\n
#         JSON-encoded dictionary with the following attributes:
#             1. message: message stating the database has been saved
#             2. status : indicates if an error occurred (status=1) or not
#                         (status=0) (boolean)
#     """
#     try:
#         # Obtains the database's full path
#         db_fp = os.path.join(rdb_dir, 'rep_database.pickle')

#         # Opens the database file (or creates one if necessary) and stores the
#         # database object
#         with open(db_fp, 'wb') as handle:
#             pickle.dump(glb.rep_db, handle, protocol=pickle.HIGHEST_PROTOCOL)

#         # Updates the database changed flag
#         glb.db_changed = False

#         # Creates output message and sets the status as 0
#         output_msg = "Databased saved!"
#         status     = False
#     except Exception as excpt:
#         # On exception, create output message (with exception) and sets the
#         # status as 1 (something went wrong)
#         output_msg = f"Unable to save database. Reason: {excpt}"
#         status     = True

#     return {"message":output_msg, "status":status}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/reload_database")
async def reload_database(
    rdb_dir: str  = Query(glb.RDB_DIR, description="Full path to Representation database directory (string)"),
    verbose: bool = Query(False, description="Controls the amount of text that is printed to the server's console (boolean)")):
    """
    TODO: FIX THIS ENDPOINT!

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
        #glb.rep_db = load_representation_db(os.path.join(rdb_dir,
        #                                'rep_database.pickle'), verbose=verbose)

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

@fr_router.post("/people/get_faces")
async def people_get_faces(person_id: int = Query(None, description="'person_id key' in Person table [integer]")):
    """
    API endpoint: people_get_faces()

    Views all information of all Representations belonging to a group number 
    specified by 'person_id'. Raises a Value Error if the 'return_type'
    provided is neither 'records' nor 'image'.

    Parameters:
        - person_id : desired group / cluster number [integer].

    Output:\n
            JSON-encoded FaceRep result for a specific person_id
    """
    query = select(FaceRep.id, FaceRep.person_id, FaceRep.image_name_orig, FaceRep.image_fp_orig, FaceRep.region).where(FaceRep.person_id == person_id)
    result = glb.sqla_session.execute(query)

    return_value = []
    for item in result:
        return_value.append({'id': item.id, 'person_id': item.person_id, 'image_name_orig': item.image_name_orig, 'image_fp_orig': item.image_fp_orig, 'region': [int(x) for x in item.region] })

    return return_value


# ------------------------------------------------------------------------------

@fr_router.post("/people/add_new")
async def people_add_new(person_name: str = Query(None, description="new person name [string]")):
    """
    API endpoiunt: add a new Person record and return its ID
    """
    query = insert(Person).values(name=person_name)
    result = glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    return result.inserted_primary_key


# ------------------------------------------------------------------------------

@fr_router.post("/people/set_name")
async def people_set_name(person_id: int = Query(None, description="'person_id key' in Person table [integer]"),
                          person_name: str = Query(None, description="new person name [string]")):
    """
    API endpoint: set the name of one person in Person table given its ID
    """
    query = update(Person).values(name = person_name).where(Person.id == person_id)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    return 'ok'


# ------------------------------------------------------------------------------

@fr_router.post("/people/set_note")
async def people_set_note(person_id: int = Query(None, description="'person_id key' in Person table [integer]"),
                          new_note: str = Query(None, description="new person note [string]")):
    """
    API endpoint: set the note of one person in Person table given its ID
    """
    query = update(Person).values(note = new_note).where(Person.id == person_id)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    return 'ok'


# ------------------------------------------------------------------------------

@fr_router.post("/people/assign_facerep")
async def people_assing_facerep(person_id: int = Query(None, description="'ID primary key in Person table [integer]"),
                          facerep_id: int = Query(None, description="ID primary key in FaceRep table [integer")):
    """
    API endpoiunt: join a FaceRep record to one Person record through the primary join Person.id -> FaceRep.person_id
    """
    query = update(FaceRep).values(person_id = person_id, group_no = -2).where(FaceRep.id == facerep_id)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    return 'ok'

# ------------------------------------------------------------------------------

@fr_router.post("/facerep/unjoin")
async def facerep_unjoin(face_id  : int = Query(None, description="ID of FaceRep record")):
    """
    API endpoint: unjoin a FaceRep record from a Person, setting its person_id to None and group_no to -1 

    This API unlinked a FaceRep record from its Person and set its person_id to None and group_no to -1 

    Parameters:
    - ID:  the FaceRep ID of the record to unjoin [integer]

    Output:\n
        JSON-encoded dictionary containing the following key/value pairs:
            1. removed : number of files removed
            2. skipped : number of files skipped
    """

    # Set group_no to -2 and person_id=None for a specific FaceRep record
    facerep_set_groupno_done(glb.sqla_session, face_id)

    # check and remove eventually record from People table that doesn't have any
    #corresponding record in FaceRep table
    people_clean_without_repps(glb.sqla_session)

    return 'OK'

# ------------------------------------------------------------------------------

@fr_router.post("/facerep/get_ungrouped")
async def facerep_get_ungrouped():
    """
    API endpoint: get all records from FaceRep that are not linked wioth a Person
                  I.E. the ones that has group_no field set to -1

    Parameters:
        - None

    Output:\n
        JSON-encoded list of FaceRep.id(s) that match the above condition
    """

    query = select(FaceRep.id, FaceRep.person_id, FaceRep.image_name_orig, FaceRep.region).where(FaceRep.group_no == -1)
    result = glb.sqla_session.execute(query)
    return_value = []
    for item in result:
        print([int(x) for x in item.region])
        return_value.append({'id': item.id, 'person_id': item.person_id, 'image_name_orig': item.image_name_orig, 'region': [int(x) for x in item.region] })

    return return_value

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
async def create_database_from_directory(params: CreateDatabaseParams,
    image_dir   : Optional[str]  = Query(glb_img_dir, description="Full path to directory containing images (string)")):
    """
    API endpoint: create_database_from_directory()

    Creates an SQLite database from a directory containing images. The image
    files are expected to have any of the following formats: .jpg, .png, .npy.

    Parameters:
    - params: a structure with the following parameters:
        1. detector_name  - name of face detector model [string].
        2. verifier_names - list of names of face verifier models [list of
                            strings].
        3. align          - perform face alignment flag (default=True)
                            [boolean].
        4. normalization  - name of image normalization [string].
        5. auto_grouping  - toggles whether Representations should be grouped /
                            clusted automatically using the DBSCAN algorithm
                            (default=True) [boolean].
        6. eps            - maximum distance between two samples for one to be
                            considered as in the neighborhood of the other. This
                            is the most important DBSCAN parameter to choose
                            appropriately for the specific data set and distance
                            function (default=0.5) [float].
        7. min_samples    - the number of samples (or total weight) in a
                            neighborhood for a point to be considered as a core
                            point. This includes the point itself
                            (min_samples=2) [integer].
        8. metric         - the metric used when calculating distance between
                            instances in a feature array. It must be an option
                            allowed by sklearn.metrics.pairwise_distances
                            (default='cosine') [string].
        9. pct            - used to filter faces which are smaller than this
                            percentage of the original image's area (width x
                            height) [float].
       10. check_models   - toggles if the function should check if all desired
                            face detector & verifiers are correctly loaded. If
                            they are not, builds them from scratch, exitting if
                            the building fails [boolean].
       11. verbose        - output messages to server's console [boolean].

        [Example] JSON schema:
        {
          "detector_name": "retinaface",
          "verifier_names": ["ArcFace"],
          "align": true,
          "normalization": "base",
          "auto_grouping": true,
          "eps": 0.5,
          "min_samples": 2,
          "metric": "cosine",
          "pct": 0.02,
          "check_models": true,
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
    # Initialize output message and records
    output_msg = ''
    records    = []

    # If image directory provided is None or is not a directory, use default
    # directory
    if not image_dir or not os.path.isdir(image_dir):
        global glb_img_dir
        output_msg += 'Image dir is None, does not exist or is not a '\
                   +  'directory. Using default directory instead.\n'
        image_dir  = glb_img_dir

    # Database does not exist
    if  database_is_empty(glb.sqla_engine):
        # Do nothing, but set message
        output_msg += 'Database does not exist! '\
                   +  'Please create one before using this endpoint.\n'

    # Face Representation table does not exist
    elif not all_tables_exist(glb.sqla_engine, ['representation']):
        # Do nothing, but set message
        output_msg += "Face representation table ('representation') "\
                   +  'does not exist! Please ensure that this table exists '\
                   +  'before using this endpoint.\n'

    # Otherwise (database is not empty and table exists), 
    else:
        # Processes face images from the image directory provided. Note that
        # this function adds the changes to the global session but does not
        # commit them.
        records = process_faces_from_dir(image_dir, glb.models, glb.models,
                            detector_name  = params.detector_name,
                            verifier_names = params.verifier_names,
                            normalization  = params.normalization,
                            align          = params.align,
                            auto_grouping  = params.auto_grouping,
                            eps            = params.eps,
                            min_samples    = params.min_samples,
                            metric         = params.metric,
                            pct            = params.pct,
                            check_models   = params.check_models,
                            verbose        = params.verbose)
        
        # Commits the records and updates the message
        glb.sqla_session.commit()
        output_msg += ' success!'
    
    return {'n_records':len(records), 'message':output_msg}

# ------------------------------------------------------------------------------

@fr_router.post("/create_database/from_zip")
async def create_database_from_zip(myfile: UploadFile,
    params      : CreateDatabaseParams = Depends(),
    image_dir   : Optional[str]  = Query(glb.IMG_DIR, description="Full path to directory containing images (string)"),
    auto_rename : Optional[bool] = Query(True       , description="Flag to force auto renaming of images in the zip file with names that match images already in the image directory (boolean)")):
    """
    API endpoint: create_database_from_zip()

    Creates an SQLite database from a zip file. The zip file is expected to
    contain image files in any of the following formats: .jpg, .png, .npy.

    The images in the zip file are extracted to a temporary directory. Any image
    with the same name of another image in the 'image directory' is either
    renamed (auto_rename=True) or skipped (auto_rename=False). Renamed images
    are renamed using a random unique object identifier obtained by uuid4() from
    the uuid library.

    Parameters:
    - myfile: a zip file

    - params: a structure with the following parameters:
        1. detector_name  - name of face detector model [string].
        2. verifier_names - list of names of face verifier models [list of
                            strings].
        3. align          - perform face alignment flag (default=True)
                            [boolean].
        4. normalization  - name of image normalization [string].
        5. auto_grouping  - toggles whether Representations should be grouped /
                            clusted automatically using the DBSCAN algorithm
                            (default=True) [boolean].
        6. eps            - maximum distance between two samples for one to be
                            considered as in the neighborhood of the other. This
                            is the most important DBSCAN parameter to choose
                            appropriately for the specific data set and distance
                            function (default=0.5) [float].
        7. min_samples    - the number of samples (or total weight) in a
                            neighborhood for a point to be considered as a core
                            point. This includes the point itself
                            (min_samples=2) [integer].
        8. metric         - the metric used when calculating distance between
                            instances in a feature array. It must be an option
                            allowed by sklearn.metrics.pairwise_distances
                            (default='cosine') [string].
        9. pct            - used to filter faces which are smaller than this
                            percentage of the original image's area (width x
                            height) [float].
       10. check_models   - toggles if the function should check if all desired
                            face detector & verifiers are correctly loaded. If
                            they are not, builds them from scratch, exitting if
                            the building fails [boolean].
       11. verbose        - output messages to server's console [boolean].

        [Example] JSON schema:
        {
          "detector_name": "retinaface",
          "verifier_names": ["ArcFace"],
          "align": true,
          "normalization": "base",
          "auto_grouping": true,
          "eps": 0.5,
          "min_samples": 2,
          "metric": "cosine",
          "pct": 0.02,
          "check_models": true,
          "verbose": false
        }

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

    # Database does not exist
    if  database_is_empty(glb.sqla_engine):
        # Do nothing, but set message
        output_msg += 'Database does not exist! '\
                   +  'Please create one before using this endpoint.\n'

    # Face Representation table does not exist
    elif not all_tables_exist(glb.sqla_engine, ['representation']):
        # Do nothing, but set message
        output_msg += "Face representation table ('representation') "\
                   +  'does not exist! Please ensure that this table exists '\
                   +  'before using this endpoint.\n'

    # Otherwise (database is not empty and table exists), 
    else:
        # Initialize dont_skip flag as True
        dont_skip   = True

        # Extract zip files
        output_msg += 'Extracting images in zip:'
        skipped_files = []

        try:
            # Process the zip file containing the image files
            skipped_files = process_image_zip_file(myfile, image_dir,
                                                    auto_rename=auto_rename)
            output_msg += ' success! '

        except Exception as excpt:
            dont_skip   = False
            output_msg += f' failed (reason: {excpt}).'

        # Processes face images from the image directory provided if 'dont_skip'
        # is True
        if dont_skip:
            output_msg += 'Creating database: '
            records = process_faces_from_dir(image_dir, glb.models, glb.models,
                            detector_name  = params.detector_name,
                            verifier_names = params.verifier_names,
                            normalization  = params.normalization,
                            align          = params.align,
                            auto_grouping  = params.auto_grouping,
                            eps            = params.eps,
                            min_samples    = params.min_samples,
                            metric         = params.metric,
                            pct            = params.pct,
                            check_models   = params.check_models,
                            verbose        = params.verbose)
        
            # Commits the records and updates the message
            glb.sqla_session.commit()
            output_msg += ' success!'
        else:
            records = []

    return {'n_records':len(records), 'n_skipped':len(skipped_files),
            'skipped_files':skipped_files, 'message':output_msg}

# ------------------------------------------------------------------------------

@fr_router.post("/verify/no_upload", response_model=List[List[List[VerificationMatch]]])
async def verify_no_upload(files: List[UploadFile],
                          params: VerificationParams = Depends()):
    """
    API endpoint: verify_no_upload()

    Executes the face verification process on one or more images without
    uploading these images to the server and adding them to the database.

    Parameters:
    - files: list of image files

    - params: a structure with the following parameters:
        1. detector_name - name of face detector model [string]
        2. verifier_name - name of face verifier model [string]
        3. align         - perform face alignment flag [boolean, default=True]
        4. normalization - name of image normalization [string]
        5. metric        - name of similarity / distance metric [string]
        6. threshold     - cutoff value for positive (match) decision [float]
        7. verbose       - output messages to server's console [boolean,
                            default=False]

    Output:\n
        Returns a 4-level (triple nested list), where each level allows for the
        following:
            Level 1: contemplates multiple images in a single function call
                     (list of images)
            Level 2: contemplates (possibly) multiple faces per image
                     (list of faces)
            Level 3: contemplates (possibly) multiple matches per face
                     (list of matches)
            Level 4: the result / match object itself (JSON-encoded structures)
    """
    # Initializes results list and obtains the relevant embeddings from the 
    # representation database
    verification_results = []
    dtb_embs = get_embeddings_as_array(params.verifier_name)

    # Loops through each file
    for f in files:
        # Obtains contents of the file & transforms it into an image
        data  = np.fromfile(f.file, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        image = image[:, :, ::-1]

        # Detects faces
        output = do_face_detection(image, detector_models=glb.models,
                                    detector_name=params.detector_name,
                                    align=params.align, verbose=params.verbose)

        # Filter regions & faces which are too small
        filtered_regions, idxs = discard_small_regions(output['regions'],
                                                    image.shape, pct=params.pct)
        filtered_faces         = [output['faces'][i] for i in idxs]

        # Calculates the deep neural embeddings for each face image in outputs
        embeddings = calc_embeddings(filtered_faces, glb.models,
                                     verifier_names=params.verifier_name,
                                     normalization=params.normalization)

        # Initialize current image's result container
        cur_img_results = []

        # 
        for cur_embd in embeddings:
            # Calculates the similarity between the current embedding and all
            # embeddings from the database
            similarity_obj = calc_similarity(cur_embd[params.verifier_name],
                                             dtb_embs,
                                             metric=params.metric,
                                             face_verifier=params.verifier_name,
                                             threshold=params.threshold)

            # Gets all matches based on the similarity object and append the
            # result to the results list
            result = get_matches_from_similarity(similarity_obj)

            # Stores the result for each face on the current image
            cur_img_results.append(result)

        # Then, stores the set of results (of the current image) to the
        # verification results list
        verification_results.append(cur_img_results)

    return verification_results
    
# ------------------------------------------------------------------------------

@fr_router.post("/verify/with_upload", response_model=List[List[List[VerificationMatch]]])
async def verify_with_upload(files: List[UploadFile],
    params     : VerificationParams = Depends(),
    img_dir    : str                = Query(glb_img_dir, description="Full path to image directory (string)"),
    save_as    : ImageSaveTypes     = Query(default_image_save_type, description="File type which uploaded images should be saved as (string)"),
    overwrite  : bool               = Query(False, description="Flag to indicate if an uploaded image with the same name as an existing one in the server should be saved and replace it (boolean)"),
    auto_rename: bool               = Query(True, description="Flag to force auto renaming of images in the zip file with (boolean)"),
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

    - auto_group  : toggles automatic grouping of the uploaded image based on
                    the face verification results (boolean, default: True).

    Output:\n
        Returns a 4-level (triple nested list), where each level allows for the
        following:
            Level 1: contemplates multiple images in a single function call
                     (list of images)
            Level 2: contemplates (possibly) multiple faces per image
                     (list of faces)
            Level 3: contemplates (possibly) multiple matches per face
                     (list of matches)
            Level 4: the result / match object itself (JSON-encoded structures)
    """
    # Initializes FaceReps & results list, gets all files in the image directory
    # and obtains the relevant embeddings from the representation database
    verification_results = []
    all_files            = [name.split('.')[0] for name in os.listdir(img_dir)]
    dtb_embs             = get_embeddings_as_array(params.verifier_name)

    # Loops through each file
    for f in files:
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
            img_name = str(uuid4())
            img_fp   = os.path.join(img_dir, img_name + '.' + save_as)

            if save_as == ImageSaveTypes.NPY:
                np.save(img_fp, img, allow_pickle=False, fix_imports=False)
            else:
                mpimg.imsave(img_fp, img)
        else:
            continue  # skips verification using this file

        # Detects faces
        output = do_face_detection(img, detector_models=glb.models,
                                    detector_name=params.detector_name,
                                    align=params.align, verbose=params.verbose)

        # Filter regions & faces which are too small
        filtered_regions, idxs = discard_small_regions(output['regions'],
                                                    img.shape, pct=params.pct)
        filtered_faces         = [output['faces'][i] for i in idxs]

        # Calculates the deep neural embeddings for each face image in outputs
        embeddings = calc_embeddings(filtered_faces, glb.models,
                                     verifier_names=params.verifier_name,
                                     normalization=params.normalization)

        # Initialize current image's result container
        cur_img_results = []

        # Loops through each face region and embedding and creates a FaceRep for
        # each face
        for region, cur_embd in zip(filtered_regions, embeddings):
            # Calculates the similarity between the current embedding and all
            # embeddings from the database
            similarity_obj = calc_similarity(cur_embd[params.verifier_name],
                                             dtb_embs, metric=params.metric,
                                             face_verifier=params.verifier_name,
                                             threshold=params.threshold)

            # Gets all matches based on the similarity object and appends the
            # result to the current image results list
            result = get_matches_from_similarity(similarity_obj)
            cur_img_results.append(result)

            # Automatically determines the image's group based on the best match
            # if it has a distance / similarity of <=0.75*threshold and if
            # 'auto_group' is True
            if auto_group and len(result) > 0:
                if similarity_obj['distances'][0]\
                    <= 0.75 * similarity_obj['threshold']:
                    group_no = result[0].group_no
                else:
                    group_no = -1
            else:
                group_no = -1

            # Creates a FaceRep for each detected face
            rep = FaceRep(image_name_orig=img_name, image_name='',
                             image_fp_orig=img_fp, image_fp='',
                             group_no=group_no, region=region,
                             embeddings=cur_embd)

            # Adds each FaceRep to the global session
            glb.sqla_session.add(rep)

        # Stores the verification result and commits the FaceReps
        verification_results.append(cur_img_results)
        glb.sqla_session.commit()

    return verification_results

# ------------------------------------------------------------------------------

@fr_router.post("/verify/existing_file", response_model=List[List[List[VerificationMatch]]])
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
        Returns a 4-level (triple nested list), where each level allows for the
        following:
            Level 1: contemplates multiple images in a single function call
                     (list of images)
            Level 2: contemplates (possibly) multiple faces per image
                     (list of faces)
            Level 3: contemplates (possibly) multiple matches per face
                     (list of matches)
            Level 4: the result / match object itself (JSON-encoded structures)
    """
    # Initializes results list and obtains the relevant embeddings from the 
    # Representation database
    verification_results = []
    dtb_embs             = get_embeddings_as_array(params.verifier_name)

    # Gets the unique ids in the database
    query    = glb.sqla_session.query(FaceRep.id)
    all_uids = [id[0] for id in query.all()]

    # Loops through each file
    for i, f in enumerate(files):
        # Tries to convert file string into a unique identifier
        try:
            is_valid_uid = int(f) in all_uids
        except:
            is_valid_uid = False

        # Tries to load (or find) the file and obtain its embedding
        if is_valid_uid: # string is valid uuid
            # Obtain the embeddings
            query      = glb.sqla_session.query(FaceRep.embeddings).where(FaceRep.id == int(f))
            embeddings = list(query.all()[0])

        elif os.path.isfile(os.path.join(img_dir, f)): # string is a valid file
            # Opens the image file
            image = cv2.imread(os.path.join(img_dir, f))
            image = image[:, :, ::-1]

            # Detects faces
            output = do_face_detection(f, detector_models=glb.models,
                                    detector_name=params.detector_name,
                                    align=params.align, verbose=params.verbose)

            # Filter regions & faces which are too small
            filtered_regions, idxs = discard_small_regions(output['regions'],
                                        mpimg.imread(f).shape, pct=params.pct)
            filtered_faces         = [output['faces'][i] for i in idxs]

            # Calculates the deep neural embeddings for each face image in
            # outputs
            embeddings = calc_embeddings(filtered_faces, glb.models,
                                            verifier_names=params.verifier_name,
                                            normalization=params.normalization)

        else: # string is not a valid unique identifier nor file
            continue

        # Initialize current image's result container
        cur_img_results = []

        # Loops through each face region and embedding and creates a FaceRep for
        # each face
        for cur_embd in embeddings:
            # Calculates the similarity between the current embedding and all
            # embeddings from the database
            similarity_obj = calc_similarity(cur_embd[params.verifier_name],
                                             dtb_embs, metric=params.metric,
                                             face_verifier=params.verifier_name,
                                             threshold=params.threshold)

            # Gets all matches based on the similarity object and appends the
            # result to the current image results list
            result = get_matches_from_similarity(similarity_obj)
            cur_img_results.append(result)

        # 
        verification_results.append(cur_img_results)

    return verification_results
    
# ------------------------------------------------------------------------------