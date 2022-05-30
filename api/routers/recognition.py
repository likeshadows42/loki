# ==============================================================================
#                              RECOGNITION API METHODS
# ==============================================================================
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy                 as np
import api.global_variables  as glb

from typing                  import List, Optional
from filecmp                 import cmp
from fastapi                 import APIRouter, UploadFile, Depends, Query
from IFR.api                 import init_load_verifiers, init_load_detectors,\
                               get_embeddings_as_array, process_image_zip_file,\
                               database_is_empty, get_matches_from_similarity,\
                               all_tables_exist, process_faces_from_dir,\
                               load_database, facerep_set_groupno_done,\
                               start_session, people_clean_without_repps
from tempfile                import TemporaryDirectory
from sqlalchemy              import select, update, insert, delete
from IFR.classes             import *
from IFR.functions           import ensure_dirs_exist, calc_embeddings,\
                                calc_similarity, do_face_detection,\
                                discard_small_regions, image_is_uncorrupted,\
                                rename_file_w_hex_token

from shutil                  import rmtree, move   as sh_move
from matplotlib              import image          as mpimg

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
    """
    API endpoint: inspect_globals()

    Allows the user to inspect the values of all global variables. Mainly used
    for debugging.

    Input:
        1. print2console - toggles if the function should also print the values
                            of the global variable to the console [boolean,
                            default=True].

    Output:\n
        JSON-encoded dictionary with the following attributes:
            1. dirs: set up directories                      [list of strings]
            2. dir_names: names of each set up directory     [list of strings]
            3. detector_names: names of face detectors       [list of strings]
            4. verifier_names: names of face verifiers       [list of strings]
            5. model_names   : names of loaded models        [list of strings]
            6. sqlite_db_name: name of SQLite database       [string]
            7. sqlite_db_fp  : full path of SQLite database  [string]
            8. sqla_engine   : SQLAlchemy engine object      [engine object]
    """
    # Obtains all current directory paths
    directories = [glb.API_DIR    , glb.DATA_DIR   , glb.IMG_DIR, glb.RDB_DIR,
                   glb.SVD_MDL_DIR, glb.SVD_VRF_DIR, glb.SVD_DTC_DIR]

    # Prints the path variables along with their names
    dir_names = ['API root dir', 'Data dir', 'Image dir',
                 'Rep. database dir', 'Saved models dir',
                 'Saved face verifiers dir', 'Saved face detectors dir']

    if print2console:
        print("  > Paths:")
        for dir, name in zip(directories, dir_names):
            print(name.ljust(24) + ':', dir)
        print('')
        
    # Prints all face detectors, indicating if they have been loaded (or build
    # from scratch) successfully
    if print2console:
        print("  > All face detectors:")
        if isinstance(glb.detector_names, list):
            all_detectors = glb.detector_names
        else:
            all_detectors = [glb.detector_names]
        
        for i, name in enumerate(all_detectors):
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
        if isinstance(glb.verifier_names, list):
            all_verifiers = glb.verifier_names
        else:
            all_verifiers = [glb.verifier_names]
        
        for i, name in enumerate(all_verifiers):
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
        print("SQLite database:".ljust(21)      + ':', glb.SQLITE_DB)
        print("SQLite database path:".ljust(21) + ':', glb.SQLITE_DB_FP)
        print("SQL alchemy engine:".ljust(21)   + ':', glb.sqla_engine)

    return {'dirs':directories, 'dir_names':dir_names,
            'detector_names':glb.detector_names,
            'verifier_names':glb.verifier_names,
            'model_names':list(glb.models.keys()),
            'sqlite_db_name':glb.SQLITE_DB,
            'sqlite_db_fp':glb.SQLITE_DB_FP}

# ------------------------------------------------------------------------------

@fr_router.post("/debug/server_reset")
async def server_reset():
    """
    API endpoint: server_reset()

    Allows the user to restart the server without manually requiring to shut it
    down and start it up.

    Output:\n
        JSON-encoded dictionary with the following attributes:
            1. message: message stating the server has been restarted [string]
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
                                   sql_dir: str = Query(glb.RDB_DIR, description="Full path to Representation database directory (string)")):
    """
    API endpoint: edit_default_directories()

    Allows the user to edit the image and Representation database directories.
    These are determined automatically on server startup. These edit will NOT be
    persistent across server restarts.

    Parameters:
    - img_dir: full path to image directory [string]

    - sql_dir: full path to SQLite database (including file name!) [string]

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

    # Sets the SQLITE_DB_FP path to the one provided IF it is a valid directory
    if os.path.isfile(sql_dir):
        glb.SQLITE_DB_FP = sql_dir
        output_msg += f'SQLITE_DB_FP set to {sql_dir}\n'
    else:
        output_msg += f'Full path provided is not a valid file ({sql_dir})\n'
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
                id         = rep.id,
                person_id  = rep.person_id,
                image_name = rep.image_name,
                image_fp   = rep.image_fp,
                group_no   = rep.group_no,
                region     = rep.region,
                embeddings = [name for name in rep.embeddings.keys()]
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
                filesize = fprc.filesize
            ))

    # Raises an assertion error because the selected table does not exist in the
    # database
    else:
        raise AssertionError("Invalid table selected!")

    return output_obj

# ------------------------------------------------------------------------------

@fr_router.post("/utility/image_dir_clear")
async def image_dir_clear():
    """
    API endpoint: image_dir_clear()

    Clears the image directory and the SQLite database. Then, recreates the new
    database with the necessary tables, but each and every table is empty (i.e.
    has no content).

    The image directory is cleared by removing the entire directory and
    recreating an empty directory with the same path.

    Parameters:
        - None

    Output:\n
        JSON-encoded dictionary with the following attributes:
            1. 'message': message stating that the directory was cleared OR with
                the result of any eventual exception [string]

            2. 'status' : flag indicating if every went smoothly without error
                (False) or if there was an error (True) [boolean]
    """
    # Initializes message & status
    msg    = ''
    status = False

    # Removes the image directory (and all of its content) then recreates it
    try:
        rmtree(glb.IMG_DIR)
        ensure_dirs_exist([glb.IMG_DIR], verbose=False)
        msg    = f'Image directory cleared'
    except Exception as excpt:
        msg    = f'Exception occured: {excpt}'
        status = True

    # Remove SQlite file and recreate again
    os.remove(glb.SQLITE_DB_FP)
    glb.sqla_engine  = load_database(glb.SQLITE_DB_FP)
    glb.sqla_session = start_session(glb.sqla_engine)
    glb.sqla_session.commit()

    return {'message':msg, 'status':status}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/database_clear")
async def database_clear():
    """
    API endpoint: database_clear()

    Clears the SQLite database. The new database contains the necessary tables,
    but each and every table is empty (i.e. has no content).

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

@fr_router.post("/utility/remove_image")
async def remove_image(image_name : str  = Query(None, description="Image's name (with extension) [string]"),
                remove_from_server: bool = Query(True, description="Toggles between  [integer]"),
                image_dir         : str  = Query(glb.IMG_DIR, description="Full path to server's image directory [str]")):
    """
    API endpoint: remove_image()
    
    Removes an image from the database, that is, removes all associated face
    representation (FaceReps) records and processed file (ProcessedFiles)
    record. If 'remove_from_server' is True, then also removes the image from
    the server.

    WARNING: Please note that this operation is irreversible!

    Parameters:
        - image_name        : image's name [string].

        - remove_from_server: toggles between removing the chosen image from the
                                server or not [boolean, default=True].

        - image_dir         : directory containing all images in the server
                                [string, default=<glb.IMG_DIR>].

    Output:\n
        JSON-encoded dictionary with the following key/value pairs is returned:
            1. status: flag indicating if the function executed without any
                    errors (False) or if 1 or more errors occurred (True).
            
            2. message: informative message string
    """
    # Initializes failed files list and return flag
    ret_flag     = False
    msg          = 'ok'

    # Tries to ensures the image name is a name (and not a full path)
    try:
        image_name = image_name[image_name.rindex('/')+1:]
    except:
        pass

    print('Image full path:', os.path.join(image_dir, image_name),
          '| decision:', os.path.isfile(os.path.join(image_dir, image_name)))

    # First, checks if the image exists
    if not os.path.isfile(os.path.join(image_dir, image_name)):
        ret_flag = True
        msg      = 'Chosen image does not exist!'
        return {'status':ret_flag, 'message':msg}

    # Deletes the image from the server
    if remove_from_server:
        try:
            os.remove(os.path.join(image_dir, image_name))
        except Exception as excpt:
            ret_flag = True
            print(f'Could not remove image {image_name}.\n',
                  f'(reason: {excpt})')
            msg = f'Could not remove image {image_name}.'
    else:
        msg = 'ok (delete image skipped)'

    # Deletes all FaceReps associated with the chosen image
    dele = delete(FaceRep).where(FaceRep.image_name == image_name)
    glb.sqla_session.execute(dele)
    glb.sqla_session.commit()
    
    # Deletes all ProcessedFiles associated with the chosen image
    dele = delete(ProcessedFiles).where(ProcessedFiles.filename == image_name)
    glb.sqla_session.execute(dele)
    glb.sqla_session.commit()

    return {'status':ret_flag, 'message':msg}

# ------------------------------------------------------------------------------

@fr_router.post("/people/list")
async def people_list():
    """
    API endpoint: people_list()
    
    Returns the id, name and note of ALL people in the Person table.

    Parameters:
    - None

    Output:\n
        JSON-encoded ...
    """
    query = select(Person.id, Person.name, Person.note)
    result = glb.sqla_session.execute(query)
    return result.fetchall()

# ------------------------------------------------------------------------------

@fr_router.post("/people/get_front_image")
async def people_list():
    """
    API endpoint: people_list()
    
    Returns the id, name and note of the front image for each person.

    Parameters:
    - None

    Output:\n
        JSON-encoded ...
    
    IN DEVELOPMENT
    NOT WORKING AT THE MOMENT!!!
    """
    
    query = select(Person.id, Person.name, Person.note).where(front_img = True)
    result = glb.sqla_session.execute(query)
    return result.fetchall()

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
    query = select(FaceRep.id, FaceRep.person_id, FaceRep.image_name,
                    FaceRep.image_fp, FaceRep.region).where(\
                    FaceRep.person_id == person_id)
    result = glb.sqla_session.execute(query)

    return_value = []
    for item in result:
        return_value.append({'id': item.id, 'person_id': item.person_id,
                             'image_name': item.image_name,
                             'image_fp': item.image_fp,
                             'region': [int(x) for x in item.region] })

    return return_value

# ------------------------------------------------------------------------------

@fr_router.post("/people/add_new")
async def people_add_new(person_name: str = Query(None, description="new person name [string]")):
    """
    API endpoint: people_add_new()
    
    Adds a new Person record and returns its id.

    Parameters:
        - person_name: name of new person [string].

    Output:\n
        Id of new record added to Person table
    """
    query = insert(Person).values(name=person_name)
    result = glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    return result.inserted_primary_key

# ------------------------------------------------------------------------------

@fr_router.post("/people/set_name")
async def people_set_name(person_id  : int = Query(None, description="'person_id key' in Person table [integer]"),
                          person_name: str = Query(None, description="new person name [string]")):
    """
    API endpoint: people_set_name()

    Sets the name of one person in Person table given its id.

    Parameters:
        - person_id  : person record's id [integer].

        - person_name: name of new person [string].

    Output:\n
        Returns an 'ok' message [string].
    """
    query = update(Person).values(name = person_name).where(Person.id == person_id)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    return 'ok'

# ------------------------------------------------------------------------------

@fr_router.post("/people/set_note")
async def people_set_note(person_id: int = Query(None, description="'person_id key' in Person table [integer]"),
                          new_note : str = Query(None, description="new person note [string]")):
    """
    API endpoint: people_set_note()
    
    Sets the note of one person in Person table given its id.

    Parameters:
        - person_id: person record's id [integer].

        - new_note : note [string].

    Output:\n
        Returns an 'ok' message [string].
    """
    query = update(Person).values(note = new_note).where(Person.id == person_id)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    return 'ok'

# ------------------------------------------------------------------------------

@fr_router.post("/people/assign_facerep")
async def people_assign_facerep(person_id : int = Query(None, description="'ID primary key in Person table [integer]"),
                                facerep_id: int = Query(None, description="ID primary key in FaceRep table [integer")):
    """
    API endpoint: people_assign_facerep()
    
    Joins a FaceRep record to one Person record through the primary join
    Person.id -> FaceRep.person_id.

    Parameters:
        - person_id : person record's id [integer].

        - facerep_id: face representation record's id [integer].

    Output:\n
        Returns an 'ok' message [string].
    """
    query = update(FaceRep).values(person_id = person_id,
                                  group_no = -2).where(FaceRep.id == facerep_id)
    glb.sqla_session.execute(query)
    glb.sqla_session.commit()

    return 'ok'

# ------------------------------------------------------------------------------

@fr_router.post("/people/remove_person")
async def remove_person(person_id : int  = Query(-1, description="Person ID [integer]"),
                       del_images : bool = Query(True, description="Toggles if the associated images on the server should be deleted [boolean]"),):
    """
    API endpoint: remove_person()
    
    Removes a person from the database, along with all of the associated face
    representation (FaceReps) records. If 'del_images' is True, then all of the
    images on the server associated to this person will also be deleted. For
    each file, if the deletion process fails, a message is printed to the
    console with the exception.

    WARNING: Please note that this operation is irreversible!

    Parameters:
        - person_id : person record's id [integer, default=-1].

        - del_images: toggles between removing the associated images on the
                        server [boolean, default=True].

    Output:\n
        JSON-encoded dictionary with the following key/value pairs is returned:
            1. status: flag indicating if the function executed without any
                    errors (False) or if 1 or more errors occurred (True).

            2. failed_files: list containing the full paths of any file that
                    could not be deleted. This list will always be empty if
                    del_images is False.
            
            3. message: informative message string
    """
    # Initializes failed files list and return flag
    failed_files = []
    ret_flag     = False
    msg          = 'ok'

    # First, checks if a Person with 'person_id' exists
    if glb.sqla_session.query(Person.id).filter_by(id=person_id).first() is None:
        ret_flag = True
        msg      = f'Person {person_id} does not exist!'
        return {'status':ret_flag, 'failed_files':failed_files, 'message':msg}

    # Gets all FaceReps associated with the current person
    query   = select(FaceRep).where(Person.id == person_id)
    results = glb.sqla_session.execute(query)

    # Deletes the images associated with each FaceRep if del_images=True
    if del_images:
        for rep in results:
            try:
                os.remove(rep[0].image_fp)
            except Exception as excpt:
                ret_flag = True
                print(f'Could not remove image {rep[0].image_fp}.\n',
                      f'(reason: {excpt})')
                failed_files.append(rep[0].image_fp)
        
        n = len(failed_files)
        if n > 0:
            msg = f'{n} files failed!'
    else:
        msg = 'ok (delete images skipped)'

    # Deletes all FaceReps associated with the chosen person
    dele = delete(FaceRep).where(FaceRep.person_id==person_id)
    glb.sqla_session.execute(dele)
    glb.sqla_session.commit()
    
    # Deletes the selected Person
    dele = delete(Person).where(Person.id == person_id)
    glb.sqla_session.execute(dele)
    glb.sqla_session.commit()

    return {'status':ret_flag, 'failed_files':failed_files, 'message':msg}

# ------------------------------------------------------------------------------

@fr_router.post("/facerep/unjoin")
async def facerep_unjoin(face_id  : int = Query(None, description="ID of FaceRep record")):
    """
    API endpoint: facerep_unjoin()
    
    Unjoins a FaceRep record from a Person, setting its person_id to None and
    the group_no to -1. This API unlinks a FaceRep record from its corresponding
    Person, sets its person_id to None and group_no to -1. 

    Parameters:
        - face_id: the target FaceRep record's id to be unjoined [integer].

    Output:\n
        Returns an 'ok' message [string].
    """

    # Set group_no to -2 and person_id=None for a specific FaceRep record
    facerep_set_groupno_done(glb.sqla_session, face_id)

    # check and remove eventually record from People table that doesn't have any
    #corresponding record in FaceRep table
    people_clean_without_repps(glb.sqla_session)

    return 'ok'

# ------------------------------------------------------------------------------

@fr_router.post("/facerep/get_ungrouped")
async def facerep_get_ungrouped():
    """
    API endpoint: facerep_get_ungrouped()
    
    Gets all records from FaceRep that are not linked with to a Person (i.e. the
    ones that have a group_no equal to -1).

    Parameters:
        - None

    Output:\n
        JSON-encoded list of FaceRep.id(s) that match the above condition
    """
    # 
    query = select(FaceRep.id, FaceRep.person_id, FaceRep.image_name,
                   FaceRep.region).where(FaceRep.group_no == -1)
    result = glb.sqla_session.execute(query)

    # 
    return_value = []
    for item in result:
        print([int(x) for x in item.region])
        return_value.append({'id': item.id, 'person_id': item.person_id,
                             'image_name': item.image_name,
                             'region': [int(x) for x in item.region] })

    return return_value

# ------------------------------------------------------------------------------

@fr_router.post("/facerep/get_person")
async def facerep_get_person(facerep_id: int = Query(None, description="ID primary key in FaceRep table [integer")):
    """
    API endpoint: facerep_get_person()
    
    Gets the related Person record by using FaceRep.id reference key join.

    Parameters:
        - facerep_id: FaceRep record's id [integer].

    Output:\n
        JSON-encoded Person record that matches the above condition
    """
    query = select(Person).join(FaceRep).where(FaceRep.id == facerep_id)
    result = glb.sqla_session.execute(query)
    # glb.sqla_session.commit()

    return result.fetchall()

# ------------------------------------------------------------------------------

@fr_router.post("/faces/import_from_directory")
async def faces_import_from_directory(params: CreateDatabaseParams,
    image_dir   : Optional[str]  = Query(glb_img_dir, description="Full path to directory containing images (string)")):
    """
    API endpoint: faces_import_from_directory()

    Adds face representation records to the FaceRep table. These records are
    created from images contained in a directory. The image files are expected
    to have any of the following formats: .jpg, .png, .npy.

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
    # Initialize output message, records and duplicate file names
    output_msg     = ''
    records        = []

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

    # Otherwise (database is not empty and table exists)
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

@fr_router.post("/faces/import_from_zip")
async def faces_import_from_zip(myfile: UploadFile,
    params    : CreateDatabaseParams = Depends(),
    image_dir : Optional[str]  = Query(glb.IMG_DIR, description="Full path to directory containing images (string)"),
    t_check   : Optional[bool] = Query(True, description="Toggles transpose check to ensure image is uncorrupted (boolean)"),
    n_token   : Optional[int]  = Query(2, description="Number of hexadecimal tokens to be used during renaming (integer)")):
    """
    API endpoint: faces_import_from_zip()

    Adds face representation records to the FaceRep table. These records are
    created from images contained in the uploaded zip file. The image files are
    expected to have any of the following formats: .jpg, .png, .npy.

    The images in the zip file are extracted to a temporary directory. This
    extraction process flattens and removes any directory structure inside the
    zip file, leaving only a list of files. If files have the same name, they
    are adequately renamed. This process also filters files with unsupported
    extension, leaving only .jpg, .png and .npy files currently. Furthermore,
    the image files are tested to ensure that they are not corrupted. Finally,
    duplicate files (different names, same contents) are removed.

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

    - image_dir   : full path to directory containing images [string,
                    default: <glb.IMG_DIR>].

    - t_check     : toggles the transpose check during the process of ensuring
                    image files are uncorrupted. This makes this check slightly
                    slower, but is more robust again corrupted image files
                    [boolean, default=True].

    - n_token     : number of hexadecimal tokens used during file renaming (if
                    any file is renamed). Each hexadecimal token is composed of
                    2 random hexadecimal numbers [positive integer, default=2].

    - force_create: flag to force database creation even if one already exists,
                     overwritting the old one [boolean, default=True].

    Output:\n
        JSON-encoded dictionary with the following key/value pairs is returned:
            1. length: length of the newly created database OR of the currently
                loaded one if this process is skipped (i.e. force_create=False
                with existing database loaded)
            
            2. message: informative message string
    """
    # Initialize output message and skipped_files list
    output_msg    = ''
    skipped_files = []

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
    elif not all_tables_exist(glb.sqla_engine, glb.sqla_table_names):
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
                                            t_check=t_check, n_token=n_token,
                                            valid_exts=glb.supported_exts)
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
    params          : VerificationParams = Depends(),
    img_dir         : str                = Query(glb_img_dir, description="Full path to image directory (string)"),
    save_as         : ImageSaveTypes     = Query(default_image_save_type, description="File type which uploaded images should be saved as (string)"),
    overwrite       : bool               = Query(False, description="Flag to indicate if an uploaded image with the same name as an existing one in the server should be saved and replace it (boolean)"),
    n_token         : int                = Query(2, description="Number of hexadecimal tokens used during renaming (positive integer)"),
    auto_group      : bool               = Query(True, description="Flag to automatically group image based on verification results (boolean)"),
    threshold_per   : float              = Query(.75, description="Threshold percentage below which the autogroup algorithm automatically assign the images to the best matching person"),
    t_check         : bool               = Query(True, description="Toggles the transpose check during image integrity check (boolean)")):
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
    # Initializes verification results and skipped files lists. Obtains the
    # names of all files in the image directory 'img_dir' and gets all the
    # appropriate embeddings as a 2D array
    verification_results = []
    skipped_files        = []
    all_files            = os.listdir(img_dir)
    dtb_embs             = get_embeddings_as_array(params.verifier_name)

    # Creates a temporary directory
    with TemporaryDirectory(prefix="verify_with_upload-") as tempdir:
        # Loops through each file
        for f in files:
            # Obtains the file's image name and creates the full path
            img_name = f.filename
            img_fp   = os.path.join(tempdir, img_name[:img_name.rindex('.')]\
                        + '.' + save_as)

            # Obtains contents of the file & transforms it into an image
            data  = np.fromfile(f.file, dtype=np.uint8)
            img   = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            img   = img[:, :, ::-1]

            # ----------------------- File save / upload -----------------------
            # Saves the image to the temporary directory if it does not exist or
            # if overwrite is True. Alternatively, if a file exists with the
            # same name and overwrite is False, automatically renames it using
            # hexadecimal tokens and then saves it
            if not (img_name in all_files) or overwrite:
                if save_as == ImageSaveTypes.NPY:
                    np.save(img_fp, img, allow_pickle=False, fix_imports=False)
                else:
                    mpimg.imsave(img_fp, img)
            else:
                # Renames the file using hexadecimal tokens, creates the file's
                # full path and saves it
                img_name = rename_file_w_hex_token(
                            img_name[:img_name.rindex('.')] + '.' + save_as,
                            n_token=n_token)
                img_fp   = os.path.join(tempdir, img_name)

                if save_as == ImageSaveTypes.NPY:
                    np.save(img_fp, img, allow_pickle=False, fix_imports=False)
                else:
                    mpimg.imsave(img_fp, img)

            # --------------------------- File Check ---------------------------
            # Checks if the file extension is supported
            if not (img_name[img_name.rindex('.'):].lower() in
                    glb.supported_exts):
                print(f'File skipped (extension not supported): {img_fp}')
                skipped_files.append(img_fp)
                os.remove(img_fp)
                continue

            # Checks if the current file is uncorrupted, continuing if it is
            # corrupted
            if not image_is_uncorrupted(img_fp, transpose_check=t_check):
                print(f'File skipped (image is corrupted): {img_fp}')
                skipped_files.append(img_fp)
                os.remove(img_fp)
                continue

            # Only performs the file check if overwrite is False. If overwrite
            # is True, then this check is skipped as it does not matter
            if not overwrite:
                # Initializes the 'skip_this_file' flag as False
                skip_this_file = False

                # Queries the database to figure out which files have the SAME
                # size
                query  = select(ProcessedFiles.filename).where(\
                            ProcessedFiles.filesize == os.path.getsize(img_fp))
                result = glb.sqla_session.execute(query)
                fpaths = [os.path.join(img_dir, fname[0]) for fname in result]

                # Loops through each matched file path in the query's result
                for i, fpath in enumerate(fpaths):
                    # Checks if the files are different and if they are, set the
                    # 'skip_this_file' flag to True (and break the loop)
                    if cmp(img_fp, fpath, shallow=False):
                        skip_this_file = True
                
                    if skip_this_file:
                        break

                # Skips the current file if skip_this_file=True
                if skip_this_file:
                    print(f'File skipped (file check failed): {img_fp}')
                    skipped_files.append(img_fp)
                    os.remove(img_fp)
                    continue

            # Finally, after all checks are cleared, move the file from the
            # temporary directory to the image one
            sh_move(img_fp, os.path.join(img_dir, img_name))
            img_fp = os.path.join(img_dir, img_name)
        
            # ----------------- Face detection & verification ------------------
            # Detects faces
            output = do_face_detection(img, detector_models=glb.models,
                                    detector_name=params.detector_name,
                                    align=params.align, verbose=params.verbose)

            # Filter regions & faces which are too small
            filtered_regions, idxs = discard_small_regions(output['regions'],
                                                    img.shape, pct=params.pct)
            filtered_faces         = [output['faces'][i] for i in idxs]

            # Calculates the deep neural embeddings for each face image in
            # outputs
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
                # if it has a distance / similarity of <=threshold_per*threshold and if
                # 'auto_group' is True
                # print('similarity_obj distance', similarity_obj['distances'][0])
                # print("threshold_per * similarity_obj['threshold']", threshold_per * similarity_obj['threshold'])
                # print("result[0].person_id", result[0].person_id)
                if auto_group and len(result) > 0:
                    if similarity_obj['distances'][0]\
                        <= threshold_per * similarity_obj['threshold']:
                        person_id = result[0].person_id
                        group_no = -2
                    else:
                        group_no = -1
                        person_id = None
                else:
                    group_no = -1
                    person_id = None

                # Creates a FaceRep for each detected face
                rep = FaceRep(image_name=img_name, image_fp=img_fp,
                                group_no=group_no, region=region,
                                person_id=person_id, embeddings=cur_embd)

                # Adds each FaceRep to the global session
                glb.sqla_session.add(rep)

            # Stores the verification result and commits the FaceReps
            verification_results.append(cur_img_results)

            # After file has been processed, add it to the ProcessedFiles table
            glb.sqla_session.add(ProcessedFiles(filename=img_name,
                                            filesize=os.path.getsize(img_fp)))
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