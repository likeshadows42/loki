# ==============================================================================
#                                   API CLASSES
# ==============================================================================

from enum               import Enum
from typing             import List, Tuple
from pydantic           import BaseModel

# ------------------------------------------------------------------------------
#                 FACE RECOGNITION & ATTRIBUTE ANALYSIS CLASSES
# ------------------------------------------------------------------------------

# Path parameter class for face detector name options
class FaceDetectorOptions(str, Enum):
    OPENCV = "opencv",
    SSD    = "ssd",
    DLIB   = "dlib",
    MTCNN  = "mtcnn",
    RETINA = "retinaface"

# Response class for face regions
class Faces(BaseModel):
    faces: List[Tuple[int, int, int, int]]

# ------------------------------------------------------------------------------
#                 FACE RECOGNITION & ATTRIBUTE ANALYSIS CLASSES
# ------------------------------------------------------------------------------

# Path parameter class for face detector name options
class FaceVerifierOptions(str, Enum):
    """
    Path parameter class: available face detector names
    """
    VGGFace    = "VGG-Face"
    FACENET    = "Facenet"
    FACENET512 = "Facenet512"
    OPENFACE   = "OpenFace"
    DEEPFACE   = "DeepFace"
    DEEPID     = "DeepID"
    ARCFACE    = "ArcFace"
    DLIB       = "Dlib"

# Path parameter class for distance metric options
class DistanceMetrics(str, Enum):
    """
    Path parameter class: available distance metrics
    """
    COSINE       = "cosine"
    EUCLIDEAN    = "euclidean"
    EUCLIDEAN_L2 = "euclidean_l2"

# Path parameter class for image normalization options
class NormalizationTypes(str, Enum):
    """
    Path parameter class: available image normalizations. Certain normalizations
    improve the face verification accuracy depending on the verifier model used.
    Usually the normalization has a similar or matching name to the verifier
    model (e.g. FACENET normalization improves the FACENET model).
    """
    BASE        = "base"
    RAW         = "raw"
    FACENET     = "Facenet"
    FACENET2018 = "Facenet2018"
    VGGFACE     = "VGGFace"
    VGGFACE2    = "VGGFace2"
    ARCFACE     = "ArcFace"

# Path parameter class for get representation options
class GetRepresentationParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the get representation options.
    """
    model_name: FaceVerifierOptions = FaceVerifierOptions.VGGFace
    enforce_detection: bool = True
    detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV
    align: bool = True
    normalization: NormalizationTypes = NormalizationTypes.BASE

    # Pydantic expects a dictionary by default. You can configure your model to
    # also support loading from standard ORM parameters (i.e. attributes on the
    # object instead of dictionary lookups):
    class Config:
        orm_mode = True

# Path parameter class for face verification options
class FaceVerifierParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the face verification options.
    """
    model_name: FaceVerifierOptions = FaceVerifierOptions.VGGFace
    distance_metric: DistanceMetrics = DistanceMetrics.COSINE
    enforce_detection: bool = True
    detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV
    align: bool = True
    normalization: NormalizationTypes = NormalizationTypes.BASE
    threshold: int = -1

    # Pydantic expects a dictionary by default. You can configure your model to
    # also support loading from standard ORM parameters (i.e. attributes on the
    # object instead of dictionary lookups):
    class Config:
        orm_mode = True
