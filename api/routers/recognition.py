# ==============================================================================
#                        RECOGNITION-RELATED API METHODS
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy                  as np
import api.global_variables   as glb

from io                       import BytesIO
from typing                   import List, Optional
from zipfile                  import ZipFile
from fastapi                  import APIRouter, UploadFile, File, Depends
from IFR.classes              import *
from IFR.functions            import create_reps_from_dir, calc_embedding,\
                                     get_embeddings_as_array, calc_similarity,\
                                     create_new_representation, \
                                     get_matches_from_similarity

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

    return {'params':params}

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

@fr_router.post("/debug/bug1", response_model=VerificationMatches)
async def bug1(myfile: UploadFile, vf_params_dep: VerificationParams = Depends()):
    # This bug occurs if a UploadFile ('myfile') is required as input
    # If it is removed, then vf_params change depending on user input.
    # If not, then vf_params ignores user's inputs and uses the default values.
    # No idea why this happening. #myfile: UploadFile,
    # 

    vf_params = vf_params_dep.dict()

    print('Input parameters:\n',
          ' > detector_name'.ljust(18), f': {vf_params["detector_name"]}\n',
          ' > verifier_name'.ljust(18), f': {vf_params["verifier_name"]}\n',
          ' > align'.ljust(18)        , f': {vf_params["align"]}\n',
          ' > normalization'.ljust(18), f': {vf_params["normalization"]}\n',
          ' > metric'.ljust(18)       , f': {vf_params["metric"]}\n',
          ' > threshold'.ljust(18)    , f': {vf_params["threshold"]}\n',
          ' > verbose'.ljust(18)      , f': {vf_params["verbose"]}\n', '', sep='')

# ------------------------------------------------------------------------------

# @fr_router.post("/upload_files")
# async def upload_files(files: List[UploadFile], overwrite = False,
#                        save_as: ImageSaveTypes = ImageSaveTypes.NPY):
#     """
#     API ENDPOINT: upload_files
#         Use this endpoint to upload several files at once. The files can be
#         saved as .png, .jpg or .npy files ([save_as='npy']). If the file already
#         exists in the data directory it is skipped unless the 'overwrite' flag
#         is set to True ([overwrite=False]).
#     """    
#     # Gets all files in the raw directory
#     all_files = os.listdir(img_dir)
    
#     for f in files:
#         data     = np.fromfile(f.file, dtype=np.uint8)
#         img      = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
#         img_name = f.filename.split('.')[0]
#         img_fp   = os.path.join(img_dir, img_name + '.' + save_as)

#         if not (img_name in all_files) or overwrite:
#             if save_as == ImageSaveTypes.NPY:
#                 np.save(img_fp, img[:, :, ::-1], allow_pickle=False,
#                                                   fix_imports=False)
#             else:
#                 mpimg.imsave(img_fp, img[:, :, ::-1])

#     return {'message':'Images uploaded successfully.'}

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
async def create_database_from_zip(myfile: UploadFile = File(...),
    detector_name : FaceDetectorOptions       = default_detector,
    verifier_names: List[FaceVerifierOptions] = [default_verifier],
    align         : bool                      = default_align,
    normalization : NormalizationTypes        = default_normalization,
    tags          : Optional[List[str]]       = default_tags,
    uids          : Optional[List[str]]       = default_uids,
    verbose       : bool                      = default_verbose,
    image_dir     : Optional[str]             = glb.IMG_DIR,
    db_dir        : Optional[str]             = glb.RDB_DIR,
    force_create  : Optional[bool]            = False):

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
                                            detector_name=detector_name,
                                            align=align, show_prog_bar=True,
                                            verifier_names=verifier_names,
                                            normalization=normalization,
                                            tags=tags, uids=uids,
                                            verbose=verbose)
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
            detector_name: FaceDetectorOptions = default_detector,
            verifier_name: FaceVerifierOptions = default_verifier,
            align        : bool                = default_align,
            normalization: NormalizationTypes  = default_normalization,
            metric       : DistanceMetrics     = default_metric,
            threshold    : float               = default_threshold):
    
    # Initializes results list and obtains the relevant embeddings from the 
    # representation database
    verification_results = []
    dtb_embs             = get_embeddings_as_array(glb.rep_db, verifier_name)

    # Loops through each file
    for f in files:
        # Obtains contents of the file & transforms it into an image
        data  = np.fromfile(f.file, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        image = image[:, :, ::-1]

        # Calculate the face image embedding
        region, embeddings = calc_embedding(image, glb.models, align=align,
                                            detector_name=detector_name, 
                                            verifier_names=verifier_name,
                                            normalization=normalization)

        # Calculates the embedding of the current image
        cur_emb  = embeddings[verifier_name]

        # Calculates the similarity between the current embedding and all
        # embeddings from the database
        similarity_obj = calc_similarity(cur_emb, dtb_embs, metric=metric,
                                         model_name=verifier_name,
                                         threshold=threshold)

        # Gets all matches based on the similarity object and append the result
        # to the results list
        result = get_matches_from_similarity(similarity_obj, glb.rep_db,
                                             verifier_name)
        verification_results.append(result)

    return verification_results
    
# ------------------------------------------------------------------------------

@fr_router.post("/verify/with_upload", response_model=List[VerificationMatches])
async def verify_with_upload(files: List[UploadFile],
            detector_name: FaceDetectorOptions = default_detector,
            verifier_name: FaceVerifierOptions = default_verifier,
            align        : bool                = default_align,
            normalization: NormalizationTypes  = default_normalization,
            metric       : DistanceMetrics     = default_metric,
            threshold    : float               = default_threshold,
            save_as      : ImageSaveTypes      = default_image_save_type,
            overwrite    : bool                = False):
    
    # Initializes results list, obtains the relevant embeddings from the 
    # representation database and gets all files in the raw directory
    verification_results = []
    all_files            = os.listdir(img_dir)
    dtb_embs             = get_embeddings_as_array(glb.rep_db, verifier_name)

    # Loops through each file
    for f in files:
        # Obtains contents of the file & transforms it into an image
        data  = np.fromfile(f.file, dtype=np.uint8)
        img   = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        img   = img[:, :, ::-1]

        # Obtains the file's image name and creates the full path
        img_name = f.filename.split('.')[0]
        img_fp   = os.path.join(img_dir, img_name + '.' + save_as)

        # Saves the image if it does not exist or if overwrite is True
        if not (img_name in all_files) or overwrite:
            if save_as == ImageSaveTypes.NPY:
                np.save(img_fp, img, allow_pickle=False, fix_imports=False)
            else:
                mpimg.imsave(img_fp, img)

        # Calculate the face image embedding
        region, embeddings = calc_embedding(img, glb.models, align=align,
                                            detector_name=detector_name, 
                                            verifier_names=verifier_name,
                                            normalization=normalization)

        # Calculates the embedding of the current image
        cur_emb  = embeddings[verifier_name]

        # Calculates the similarity between the current embedding and all
        # embeddings from the database
        similarity_obj = calc_similarity(cur_emb, dtb_embs, metric=metric,
                                         model_name=verifier_name,
                                         threshold=threshold)

        # Gets all matches based on the similarity object and append the result
        # to the results list
        result = get_matches_from_similarity(similarity_obj, glb.rep_db,
                                             verifier_name)

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

@fr_router.post("/utility/get_names_in_database")
async def get_names_in_database():

    # Gets the names in the database depending on the database's size
    if len(glb.rep_db) == 0:   # no representations
        all_tags = []

    elif len(glb.rep_db) == 1: # single representation
        all_tags = [glb.rep_db[0].name_tag]

    elif len(glb.rep_db) > 1:  # many representations
        # Loops through each representation in the database and gets the name
        # tag
        all_tags = []
        for rep in glb.rep_db:
            all_tags.append(rep.name_tag)

        # Only keep unique tags and sort the 'all_tags' list
        all_tags = np.unique(all_tags)
        all_tags.sort()
    
    else: # this should never happen (negative size for a database? preposterous!)
        raise AssertionError('Representation database can '
                            +'not have a negative size!')

    return {'names':all_tags}

# ------------------------------------------------------------------------------

