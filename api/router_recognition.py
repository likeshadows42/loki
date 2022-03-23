# ==============================================================================
#                        RECOGNITION-RELATED API METHODS
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy             as np
import global_variables  as glb

from typing              import List, Optional
from fastapi             import APIRouter, UploadFile
from deepface.DeepFace   import find
from api_classes         import FaceVerifierParams, ImageSaveTypes,\
                                CreateDatabaseParams, VerificationParams,\
                                VerificationMatches
from api_functions       import build_face_verifier, create_reps_from_dir,\
                                calc_embedding, get_embeddings_as_array,\
                                calc_similarity, create_new_representation,\
                                get_matches_from_similarity
from router_detection    import fd_router, FaceDetectorOptions
from router_verification import fv_router

from matplotlib          import image                       as mpimg

dst_root_dir = glb.DST_ROOT_DIR
raw_dir      = glb.RAW_DIR
rdb_dir      = glb.RDB_DIR

# ______________________________________________________________________________
#                             ROUTER INITIALIZATION
# ------------------------------------------------------------------------------

fr_router                    = APIRouter()
fr_router.face_verifier      = None
fr_router.face_verifier_name = None

# ______________________________________________________________________________
#                                  API METHODS
# ------------------------------------------------------------------------------

@fr_router.post("/upload_files")
async def upload_files(files: List[UploadFile], overwrite = False,
                       save_as: ImageSaveTypes = ImageSaveTypes.NPY):
    """
    API ENDPOINT: upload_files
        Use this endpoint to upload several files at once. The files can be
        saved as .png, .jpg or .npy files ([save_as='npy']). If the file already
        exists in the data directory it is skipped unless the 'overwrite' flag
        is set to True ([overwrite=False]).
    """    
    # Gets all files in the raw directory
    all_files = os.listdir(raw_dir)
    
    for f in files:
        data     = np.fromfile(f.file, dtype=np.uint8)
        img      = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        img_name = f.filename.split('.')[0]
        img_fp   = os.path.join(raw_dir, img_name + '.' + save_as)

        if not (img_name in all_files) or overwrite:
            if save_as == ImageSaveTypes.NPY:
                np.save(img_fp, img[:, :, ::-1], allow_pickle=False,
                                                  fix_imports=False)
            else:
                mpimg.imsave(img_fp, img[:, :, ::-1])

    return {'message':'Images uploaded successfully.'}

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
    directory_list = [glb.API_DIR    , glb.DST_ROOT_DIR, glb.RAW_DIR,
                      glb.TARGETS_DIR, glb.RDB_DIR     , glb.SVD_MDL_DIR,
                      glb.SVD_VRF_DIR]

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

@fr_router.post("/recognize/single")
async def recognize_single_face(tgt_file: UploadFile,
                                params: FaceVerifierParams):
    """
    API ENDPOINT: recognize_single_face
        Recognizes a single image by comparing it to the images in the directory
        specified by the 'RAW_DIR' path variable.
    """    
    # Obtains contents of the target and reference files & transforms them into
    # images
    tgt_data = np.fromfile(tgt_file.file, dtype=np.uint8)
    tgt_img  = cv2.imdecode(tgt_data, cv2.IMREAD_UNCHANGED)

    # Builds the face verifier model
    fv_router.face_verifier_name, metric_name, fv_router.face_verifier = \
                build_face_verifier(model_name=params.model_name, 
                                    model=fv_router.face_verifier,
                                    distance_metric=params.distance_metric)
    
    # Unnests single element list into element. Does the same for the dictionary
    # of models unless the model name is 'Ensemble' (as Ensemble models consists
    # of 4 models)
    fv_router.face_verifier_name = fv_router.face_verifier_name[0]
    metric_name                  = metric_name[0]

    if fv_router.face_verifier_name != "Ensemble":
        fv_router.face_verifier = \
            fv_router.face_verifier[fv_router.face_verifier_name]

    # Check if a face detection model has been used before, if not use a default
    # option
    if fd_router.face_detector_name == None:
        use_backend = FaceDetectorOptions.OPENCV
    else:
        use_backend = fd_router.face_detector_name

    # Runs face recognition
    response = find(tgt_img, raw_dir, model_name=params.model_name,
                                distance_metric=params.distance_metric,
                                model=fv_router.face_verifier,
                                enforce_detection=params.enforce_detection,
                                detector_backend=use_backend,
                                align=params.align, prog_bar=False,
                                normalization=params.normalization)

    print(response)
    return response

# ------------------------------------------------------------------------------

@fr_router.post("/create_database")
async def create_database(cdb_params: CreateDatabaseParams,
                          img_dir: Optional[str] = None,
                          db_path: Optional[str] = None,
                          force_create: Optional[bool] = False):

    # If image directory provided is None, use default directory
    if not img_dir:
        global raw_dir
        img_dir = raw_dir

    # If database path provided is None, use default path
    if not db_path:
        global rdb_dir
        db_path = os.path.join(rdb_dir, 'rep_database.pickle')

    # Database exists (and has at least one element) and force create is False
    if len(glb.rep_db) > 0 and not force_create:
        # Do nothing, return number of elements in the database?
        pass

    elif len(glb.rep_db) == 0 or force_create:
        glb.rep_db = create_reps_from_dir(img_dir, glb.models,
                                    detector_name=cdb_params.detector_name,
                                    align=cdb_params.align, show_prog_bar=True,
                                    verifier_names=cdb_params.verifier_names,
                                    normalization=cdb_params.normalization,
                                    tags=cdb_params.tags, uids=cdb_params.uids,
                                    verbose=cdb_params.verbose)

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

    return {'length':len(glb.rep_db)}

# ------------------------------------------------------------------------------

from api_classes       import FaceVerifierOptions, NormalizationTypes,\
                                DistanceMetrics

@fr_router.post("/verify/no_upload", response_model=List[VerificationMatches])
async def verify_no_upload(files: List[UploadFile],
            detector_name: FaceDetectorOptions = FaceDetectorOptions.OPENCV,
            verifier_name: FaceVerifierOptions = FaceVerifierOptions.VGGFace,
            align        : bool                = True,
            normalization: NormalizationTypes  = NormalizationTypes.BASE,
            metric       : DistanceMetrics     = DistanceMetrics.COSINE,
            threshold    : float               = -1):
    
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
            detector_name: FaceDetectorOptions = FaceDetectorOptions.OPENCV,
            verifier_name: FaceVerifierOptions = FaceVerifierOptions.VGGFace,
            align        : bool                = True,
            normalization: NormalizationTypes  = NormalizationTypes.BASE,
            metric       : DistanceMetrics     = DistanceMetrics.COSINE,
            threshold    : float               = -1,
            save_as      : ImageSaveTypes      = ImageSaveTypes.NPY,
            overwrite    : bool                = False):
    
    # Initializes results list, obtains the relevant embeddings from the 
    # representation database and gets all files in the raw directory
    verification_results = []
    all_files            = os.listdir(raw_dir)
    dtb_embs             = get_embeddings_as_array(glb.rep_db, verifier_name)

    # Loops through each file
    for f in files:
        # Obtains contents of the file & transforms it into an image
        data  = np.fromfile(f.file, dtype=np.uint8)
        img   = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        img   = img[:, :, ::-1]

        # Obtains the file's image name and creates the full path
        img_name = f.filename.split('.')[0]
        img_fp   = os.path.join(raw_dir, img_name + '.' + save_as)

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

@fr_router.post("/debug/bug1", response_model=VerificationMatches)
async def bug1(myfile: UploadFile, vf_params: VerificationParams):
    # This bug occurs if a UploadFile ('myfile') is required as input
    # If it is removed, then vf_params change depending on user input.
    # If not, then vf_params ignores user's inputs and uses the default values.
    # No idea why this happening.

    print('Input parameters:\n',
          ' > detector_name'.ljust(18), f': {vf_params.detector_name}\n',
          ' > verifier_name'.ljust(18), f': {vf_params.verifier_name}\n',
          ' > align'.ljust(18)        , f': {vf_params.align}\n',
          ' > normalization'.ljust(18), f': {vf_params.normalization}\n',
          ' > metric'.ljust(18)       , f': {vf_params.metric}\n',
          ' > threshold'.ljust(18)    , f': {vf_params.threshold}\n',
          ' > verbose'.ljust(18)      , f': {vf_params.verbose}\n', '', sep='')

# ------------------------------------------------------------------------------

@fr_router.post("/set_raw_dir")
async def set_raw_dir(path: str):
    path = path.replace("//", "/")
    if os.path.isdir(path):
        glb.RAW_DIR = path
        return {'message':f'RAW DIR set to {path}'}
    else:
        return {'message':f'Path provided is not a valid directory ({path})'}

