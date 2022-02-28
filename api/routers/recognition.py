# ==============================================================================
#                        RECOGNITION-RELATED API METHODS
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy             as np

from typing              import List
from fastapi             import APIRouter, UploadFile
from deepface            import DeepFace
from deepface.DeepFace   import find
from ..api_functions     import build_face_verifier, verify_faces,\
                               calculate_similarity
from ..api_classes       import FaceVerifierParams, ImageSaveTypes
from ..api_functions     import DST_ROOT_DIR, RAW_DIR
from ..utility_functions import create_dir
from .detection          import fd_router, FaceDetectorOptions
from .verification       import fv_router
from matplotlib          import image as mpimg

dst_root_dir = DST_ROOT_DIR

# --------------------------- ROUTER INITIALIZATION ----------------------------

fr_router = APIRouter()
fr_router.face_verifier = None
fr_router.face_verifier_name = None

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

@fr_router.post("/recognize/single")
async def recognize_single_face(tgt_file: UploadFile,
                                params: FaceVerifierParams):
    """
    API ENDPOINT: recognize_single_face
        abc
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






