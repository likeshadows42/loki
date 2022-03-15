# ==============================================================================
#                        RECOGNITION-RELATED API METHODS
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy             as np

from typing              import List, Optional
from fastapi             import APIRouter, UploadFile
from deepface.DeepFace   import find
from ..api_classes       import FaceVerifierParams, ImageSaveTypes,\
                                CreateDatabaseParams, VerificationParams,\
                                VerificationMatches
from ..api_functions     import DST_ROOT_DIR, RAW_DIR, RDB_DIR,\
                                build_face_verifier, create_reps_from_dir,\
                                calc_embedding, get_embeddings_as_array,\
                                calc_similarity
from .detection          import fd_router, FaceDetectorOptions
from .verification       import fv_router

from ..                  import global_variables            as glb
from matplotlib          import image                       as mpimg

dst_root_dir = DST_ROOT_DIR
raw_dir      = RAW_DIR
rdb_dir      = RDB_DIR

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
    all_files = os.listdir(RAW_DIR)
    
    for f in files:
        data     = np.fromfile(f.file, dtype=np.uint8)
        img      = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        img_name = f.filename.split('.')[0]
        img_fp   = os.path.join(RAW_DIR, img_name + '.' + save_as)

        if (img_name in all_files) or overwrite:
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
          ' > detector_name'.ljust(18) , f': {params.detector_name}\n',
          ' > verifier_name'.ljust(18), f': {params.verifier_name}\n',
          ' > align'.ljust(18)         , f': {params.align}\n',
          ' > normalization'.ljust(18) , f': {params.normalization}\n',
          ' > metric'.ljust(18)        , f': {params.metric}\n',
          ' > threshold'.ljust(18)     , f': {params.threshold}\n',
          ' > verbose'.ljust(18)       , f': {params.verbose}\n', '', sep='')

    return {'params':params}

# ------------------------------------------------------------------------------

@fr_router.post("/debug/inspect_globals")
async def inspect_globals():
    #global app_models
    print('[check_loaded_models] models:')
    for key, value in glb.models.items():
        print(f'   -> {key}'.ljust(16) + f': {value}')
    print('')

    print('[check_loaded_models] database:')
    print('  - there are {} representations'.format(len(glb.rep_db)))
    for i, rep in zip(range(len(glb.rep_db)), glb.rep_db):
        print(f' Entry: {i}  |  ', end='')
        rep.show_summary()

    print(f'[check_loaded_models] database change status: {glb.db_changed}')
    print(f'[check_loaded_models] faiss index: {glb.faiss_index}')

    return {'models': list(glb.models.keys()), 'num_reps':len(glb.rep_db)}


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
    response = find(tgt_img, RAW_DIR, model_name=params.model_name,
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
async def create_database(params: CreateDatabaseParams,
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
                                        detector_name=params.detector_name,
                                        align=params.align, show_prog_bar=True,
                                        verifier_names=params.verifier_names,
                                        normalization=params.normalization,
                                        tags=params.tags, uids=params.uids,
                                        verbose=params.verbose)

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

@fr_router.post("/verify_file", response_model=VerificationMatches)
async def verify_file(myfile: UploadFile, params: VerificationParams,
                      add_to_database: Optional[bool] = False):

    # Obtains contents of the file & transforms it into an image
    data  = np.fromfile(myfile.file, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    image = image[:, :, ::-1]

    print('verifier_name', params.verifier_name)

    # Calculate the face image embedding
    region, embeddings = calc_embedding(image, glb.models,
                                        align=params.align,
                                        detector_name=params.detector_name, 
                                        verifier_names=params.verifier_name,
                                        normalization=params.normalization)

    # Calculates the embedding of the current image and gets all embeddings
    # (relevant to this model) from the database
    print('embeddings', embeddings)

    cur_emb  = embeddings[params.verifier_name]
    dtb_embs = get_embeddings_as_array(glb.rep_db, params.verifier_name)

    # Calculates the similarity between the current embedding and all embeddings
    # from the database
    similarity_obj = calc_similarity(cur_emb, dtb_embs, metric=params.metric,
                                     model_name=params.verifier_name,
                                     threshold=params.threshold)

    # Gets all matches
    mtch_uids  = []
    mtch_tags  = []
    mtch_names = []
    mtch_fps   = []
    mtch_rgns  = []
    mtch_embds = []

    for i in similarity_obj['idxs']:
        rep = glb.rep_db[i]
        mtch_uids.append(rep.unique_id)
        mtch_tags.append(rep.name_tag)
        mtch_names.append(rep.image_name)
        mtch_fps.append(rep.image_fp)
        mtch_rgns.append(rep.region)
        mtch_embds.append(rep.embeddings)

    print(mtch_uids)
    print(mtch_tags)
    print(mtch_names)
    print(mtch_fps)
    print(mtch_rgns)
    print(mtch_embds)

    return {'uids':mtch_uids, 'tags':mtch_tags, 'names':mtch_names,
             'fps':mtch_fps , 'rgns':mtch_rgns, 'embds':mtch_embds}

# ------------------------------------------------------------------------------





