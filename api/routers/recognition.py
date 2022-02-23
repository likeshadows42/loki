import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy            as np

from enum               import Enum
from fastapi            import APIRouter, UploadFile
from pydantic           import BaseModel
from ..my_functions     import build_face_verifier, verify_faces
from .detection         import fd_router, FaceDetectorOptions

# Path parameter class for face detector name options
class FaceVerifierOptions(str, Enum):
    """
    Path parameter class: available face detector names
    """
    VGGFace    = "VGG-Face",
    FACENET    = "Facenet",
    FACENET512 = "Facenet512",
    OPENFACE   = "OpenFace",
    DEEPFACE   = "DeepFace",
    DEEPID     = "DeepID",
    ARCFACE    = "ArcFace",
    DLIB       = "Dlib"

# Path parameter class for distance metric options
class DistanceMetrics(str, Enum):
    """
    Path parameter class: available distance metrics
    """
    COSINE       = "cosine",
    EUCLIDEAN    = "euclidean",
    EUCLIDEAN_L2 = "euclidean_l2"

# Path parameter class for image normalization options
class NormalizationTypes(str, Enum):
    """
    Path parameter class: available image normalizations. Certain normalizations
    improve the face verification accuracy depending on the verifier model used.
    Usually the normalization has a similar or matching name to the verifier
    model (e.g. FACENET normalization improves the FACENET model).
    """
    BASE        = "base",
    RAW         = "raw",
    FACENET     = "Facenet",
    FACENET2018 = "Facenet2018",
    VGGFACE     = "VGGFace",
    VGGFACE2    = "VGGFace2",
    ARCFACE     = "ArcFace"

# Path parameter class for face verification options
class FaceVerifierParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the face verification options.
    """
    model_name: FaceVerifierOptions = "VGG-Face",
    #distance_metric: DistanceMetrics = "cosine",
    #enforce_detection: bool = True,
    #detector_backend: FaceDetectorOptions = "opencv",
    #align: bool = True,
    #normalization: NormalizationTypes = "base",
    #threshold: int = -1


# --------------------------- ROUTER INITIALIZATION ----------------------------
fr_router = APIRouter()
fr_router.face_verifier = None
fr_router.face_verifier_name = None

# -------------------------------- API METHODS ---------------------------------
@fr_router.post("/verify")
async def face_verify(tgt_file: UploadFile, ref_file: UploadFile,
                      params: FaceVerifierParams):
    """
    API ENDPOINT: verify_faces
        
    """
    # params: FaceVerifierParams

    # Hard codding everything for now - these will become a single class path
    # parameter later
    # model_name = "ArcFace"
    # distance_metric = "cosine"
    # enforce_detection = True
    # detector_backend = "retinaface"
    # align = True
    # normalization = "base"
    # threshold = -1

    print(f'Model name: {params.model_name}')
        #   f'Distance metric: {distance_metric}',
        #   f'Enforce detection: {enforce_detection}',
        #   f'Detector backend: {detector_backend}',
        #   f'Align: {align}',
        #   f'Normalization: {normalization}',
        #   f'Threshold: {threshold}', sep='\n')

    raise ValueError # used for debugging - remove me later
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    tgt_data = np.fromfile(tgt_file.file, dtype=np.uint8)
    tgt_img  = cv2.imdecode(tgt_data, cv2.IMREAD_UNCHANGED)

    ref_data = np.fromfile(ref_file.file, dtype=np.uint8)
    ref_img  = cv2.imdecode(ref_data, cv2.IMREAD_UNCHANGED)

    print(f'\nFace verifier name: {fr_router.face_verifier_name}',
          f'Face verifier: {fr_router.face_verifier}', sep='\n')

    # Builds the face verifier model
    fr_router.face_verifier_name, metric_name, fr_router.face_verifier = \
                build_face_verifier(model_name=model_name, 
                                    model=fr_router.face_verifier,
                                    distance_metric=distance_metric)

    fr_router.face_verifier_name = fr_router.face_verifier_name[0]
    metric_name = metric_name[0]

    # Check if a face detection model has been used before, if not use a default
    # option
    if fd_router.face_detector_name == None:
        use_backend = "opencv"
    else:
        use_backend = fd_router.face_detector_name

    print(f'\nFace verifier name: {fr_router.face_verifier_name}',
          f'Metric name: {metric_name}',
          f'Face verifier: {fr_router.face_verifier}', sep='\n')

    # Verifies both faces
    # response = verify_faces(tgt_img, img2_path=ref_img,
    #                 model_name = fr_router.face_verifier_name,
    #                 distance_metric = metric_name,
    #                 model = fr_router.face_verifier,
    #                 enforce_detection = enforce_detection,
    #                 detector_backend = use_backend,
    #                 align = align,
    #                 prog_bar = False,
    #                 normalization = normalization,
    #                 threshold = threshold)

    response = verify_faces(tgt_img, img2_path=ref_img)

    # print('', f'Target:\n {tgt_img}', f'Ref:\n {ref_img}', sep='\n')
    # response = verify_faces(tgt_img, ref_img)
    print('\n', '-' * 80, '\n', response, '\n', '-' * 80, '\n')

    return response




