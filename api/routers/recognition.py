# ==============================================================================
#                        RECOGNITION-RELATED API METHODS
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy                  as np
import api.global_variables   as glb

from io                       import BytesIO
from uuid                     import UUID
from typing                   import List, Optional
from zipfile                  import ZipFile
from fastapi                  import APIRouter, UploadFile, File, Depends
from IFR.classes              import *
from IFR.functions            import create_reps_from_dir, calc_embedding,\
                        get_embeddings_as_array, calc_similarity,\
                        create_new_representation, get_matches_from_similarity,\
                        get_property_from_database as get_prop_from_db

from matplotlib               import image                       as mpimg

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

@fr_router.post("/debug/input")
async def debug_inputs(params: VerificationParams):
    print('Input parameters:\n',
          ' > detector_name'.ljust(18), f': {params.detector_name}\n',
          ' > verifier_name'.ljust(18), f': {params.verifier_name}\n',
          ' > align'.ljust(18)        , f': {params.align}\n',
          ' > normalization'.ljust(18), f': {params.normalization}\n',
          ' > metric'.ljust(18)       , f': {params.metric}\n',
          ' > threshold'.ljust(18)    , f': {params.threshold}\n',
          ' > verbose'.ljust(18)      , f': {params.verbose}\n', '', sep='')
    
    return params.dict()

# ------------------------------------------------------------------------------

@fr_router.post("/debug/inspect_globals")
async def inspect_globals():

    print('[inspect_globals] Path variables:')
    directory_list = [glb.API_DIR, glb.DATA_DIR   , glb.IMG_DIR,
                      glb.RDB_DIR, glb.SVD_MDL_DIR, glb.SVD_VRF_DIR]

    for name, fp in zip(glb.directory_list_names, directory_list):
        print(f'   -> Directory {name}'.ljust(30), f': {fp}', sep='')
    print('')
    
    print('[inspect_globals] models:')
    for key, value in glb.models.items():
        print(f'   -> {key}'.ljust(16) + f': {value}')
    print('')

    print('[inspect_globals] database:')
    print('  - there are {} representations'.format(len(glb.rep_db)))
    for i, rep in zip(range(len(glb.rep_db)), glb.rep_db):
        print(f' Entry: {i}  |  ', end='')
        rep.show_summary()

    print(f'[inspect_globals] database change status: {glb.db_changed}')

    return {'directories':directory_list,
            'dir_names':glb.directory_list_names,
            'models': list(glb.models.keys()), 'num_reps':len(glb.rep_db),
            'db_changed':glb.db_changed}

# ------------------------------------------------------------------------------

@fr_router.post("/create_database/from_directory")
async def create_database_from_directory(cdb_params: CreateDatabaseParams,
                                    image_dir   : Optional[str]  = glb.IMG_DIR,
                                    db_dir      : Optional[str]  = glb.RDB_DIR,
                                    force_create: Optional[bool] = False):
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
                             image_dir   : Optional[str]        = glb.IMG_DIR,
                             db_dir      : Optional[str]        = glb.RDB_DIR,
                             force_create: Optional[bool]       = False):

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
            with ZipFile(BytesIO(myfile.file.read()), 'r') as myzip:
                myzip.extractall(image_dir)
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
                                        verbose=params.verbose)
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

@fr_router.post("/verify/no_upload", response_model=List[VerificationMatches])
async def verify_no_upload(files: List[UploadFile],
                          params: VerificationParams = Depends()):
    
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
        verification_results.append(result)

    return verification_results
    
# ------------------------------------------------------------------------------

@fr_router.post("/verify/with_upload", response_model=List[VerificationMatches])
async def verify_with_upload(files: List[UploadFile],
                    params     : VerificationParams = Depends(),
                    save_as    : ImageSaveTypes     = default_image_save_type,
                    overwrite  : bool               = False,
                    auto_rename: bool               = True):
    
    # Initializes results list, gets all files in the image directory and
    # obtains the relevant embeddings from the representation database
    verification_results = []
    all_files            = [name.split('.')[0] for name in os.listdir(img_dir)]
    dtb_embs   = get_embeddings_as_array(glb.rep_db, params.verifier_name)
    print('all files:\n\n', all_files, sep='')

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
            pass
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
        similarity_obj = calc_similarity(cur_emb, dtb_embs, metric=params.metric,
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
        print('Automatic tag:', tag)

        # Creates a representation object for this image and adds it to the
        # database
        new_rep = create_new_representation(img_fp, region, embeddings, tag=tag)
        glb.rep_db.append(new_rep)
        glb.db_changed = True

        # Stores the verification result
        verification_results.append(result)

    return verification_results

# ------------------------------------------------------------------------------

@fr_router.post("/utility/edit_default_directories")
async def edit_default_directories(img_dir: str = glb.IMG_DIR,
                                   rdb_dir: str = glb.RDB_DIR):
    # Intitializes output message
    output_msg = ''
    
    # Sets the IMG_DIR path to the one provided IF it is a valid directory
    if os.path.isdir(img_dir):
        glb.IMG_DIR = img_dir
        output_msg += f'IMG_DIR set to {img_dir}'
    else:
        output_msg += f'Path provided is not a valid directory ({img_dir})'

    # Sets the RDB_DIR path to the one provided IF it is a valid directory
    if os.path.isdir(rdb_dir):
        glb.RDB_DIR = rdb_dir
        output_msg += f'RDB_DIR set to {rdb_dir}'
    else:
        output_msg += f'Path provided is not a valid directory ({rdb_dir})'

    return {'message':output_msg}

# ------------------------------------------------------------------------------

@fr_router.post("/utility/get_property_from_database")
async def get_property_from_database(
                            propty : AvailableRepProperties = default_property,
                            do_sort: bool                   = False,
                    suppress_error : bool                   = True):

    return get_prop_from_db(glb.rep_db, propty, do_sort=do_sort,
                            suppress_error=suppress_error)  

# ------------------------------------------------------------------------------

@fr_router.post("/utility/view_database")
async def view_database(amt_detail : MessageDetailOptions = default_msg_detail,
                        output_type: MessageOutputOptions = default_msg_output):
    
    if output_type == MessageOutputOptions.STRUCTURE:
        # Initialize output object
        output_obj = []

        # Case 1: STRUCTURE & SUMMARY
        if   amt_detail == MessageDetailOptions.SUMMARY:
            if len(glb.rep_db) > 0:
                for rep in glb.rep_db:
                    output_obj.append(RepsSummaryOutput(unique_id=rep.unique_id,
                                   name_tag=rep.name_tag, region=rep.region, 
                                   image_fp=rep.image_fp))

        # Case 2: STRUCTURE & COMPLETE
        elif amt_detail == MessageDetailOptions.COMPLETE:
            if len(glb.rep_db) > 0:
                for rep in glb.rep_db:
                    output_obj.append(RepsInfoOutput(unique_id=rep.unique_id,
                        name_tag=rep.name_tag, image_name=rep.image_name,
                        image_fp=rep.image_fp, region=rep.region,
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
                    output_obj += f'UID: {rep.unique_id}'\
                           +  f' | Name Tag: {rep.name_tag}'.ljust(30)\
                           +  f' | Region: {rep.region}'.ljust(30)\
                           +  f' | Image path: {rep.image_fp}'.ljust(30) + '\n'
            else:
                output_obj = 'Database is empty.'

        # Case 5: MESSAGE & COMPLETE
        elif amt_detail == MessageDetailOptions.COMPLETE:
            if len(glb.rep_db) > 0:
                for rep in glb.rep_db:
                    embd_names = [name for name in rep.embeddings.keys()]

                    output_obj += f'UID: {rep.unique_id}'\
                           +  f' | Name Tag: {rep.name_tag}'.ljust(30)\
                           +  f' | Image Name: {rep.image_name}'.ljust(30)\
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

@fr_router.post("/utility/edit_tag_by_uid/{target_uid}")
async def edit_tag_by_uid(target_uid: str, new_name_tag: str):
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

@fr_router.post("/utility/search_database_by_tag/{target_tag}")
async def search_database_by_tag(target_tag: str):

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

@fr_router.post("/utility/rename_entries_by_tag/{target_tag}")
async def rename_entries_by_tag(old_tag: str, new_tag: str):
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

# TODO: fix file overwrite for verify_with_upload AND for create_database_from_zip