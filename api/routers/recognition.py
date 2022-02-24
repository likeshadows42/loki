# ==============================================================================
#                        RECOGNITION-RELATED API METHODS
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy            as np

from typing             import List
from fastapi            import APIRouter, UploadFile
from deepface           import DeepFace
from ..api_functions    import build_face_verifier, verify_faces,\
                               calculate_similarity
from ..api_classes      import DistanceMetrics, GetRepresentationParams,\
                               FaceVerifierParams
from .detection         import fd_router, FaceDetectorOptions

# --------------------------- ROUTER INITIALIZATION ----------------------------
fr_router = APIRouter()
fr_router.face_verifier = None
fr_router.face_verifier_name = None

# -------------------------------- API METHODS ---------------------------------
