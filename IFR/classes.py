# ==============================================================================
#                                   API CLASSES
# ==============================================================================

from enum               import Enum
from uuid               import UUID
from typing             import List, Tuple, Optional
from pydantic           import BaseModel

# IMPLEMENTATION NOTE:
# Pydantic expects a dictionary by default. You can configure your model to also
# support loading from standard ORM parameters (i.e. attributes on the object
# instead of dictionary lookups) by setting:

# class Config:
#     orm_mode = True

# ______________________________________________________________________________
#                          ENUMERATION (OPTIONS) CLASSES
# ------------------------------------------------------------------------------

# Path parameter class for face detector name options
class FaceDetectorOptions(str, Enum):
    OPENCV = "opencv",
    SSD    = "ssd",
    DLIB   = "dlib",
    MTCNN  = "mtcnn",
    RETINA = "retinaface"

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

# Path parameter class for image save types
class ImageSaveTypes(str, Enum):
    """
    Path parameter class: available image save types.
    """
    PNG = "png"
    JPG = "jpg"
    NPY = "npy"


# ______________________________________________________________________________
#                                   DEFAULT VALUES
# ------------------------------------------------------------------------------

default_detector          = FaceDetectorOptions.RETINA
default_verifier          = FaceVerifierOptions.ARCFACE
default_metric            = DistanceMetrics.COSINE
default_normalization     = NormalizationTypes.BASE
default_image_save_type   = ImageSaveTypes.JPG
default_align             = True
default_enforce_detection = True
default_threshold         = -1
default_tags              = []
default_uids              = []
default_verbose           = False


# ______________________________________________________________________________
#                       PATH PARAMETER (API INPUT) CLASSES
# ------------------------------------------------------------------------------

# Path parameter class for get representation options
class GetRepresentationParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the get representation options.
    """
    model_name       : FaceVerifierOptions = default_verifier
    enforce_detection: bool                = default_enforce_detection
    detector_backend : FaceDetectorOptions = default_detector
    align            : bool                = default_align
    normalization    : NormalizationTypes  = default_normalization

    class Config:
        orm_mode = True

# Path parameter class for face verification options
class FaceVerifierParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the face verification options.
    """
    model_name       : FaceVerifierOptions  = default_verifier
    distance_metric  : DistanceMetrics      = default_metric
    enforce_detection: bool                 = default_enforce_detection
    detector_backend : FaceDetectorOptions  = default_detector
    align            : bool                 = default_align
    normalization    : NormalizationTypes   = default_normalization
    threshold        : int                  = default_threshold

    class Config:
        orm_mode = True

# Path parameter class for create_database endpoint parameters
class CreateDatabaseParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the parameters required for database creation.
    """
    detector_name : FaceDetectorOptions       = default_detector
    verifier_names: List[FaceVerifierOptions] = [default_verifier]
    align         : bool                      = default_align
    normalization : NormalizationTypes        = default_normalization
    tags          : Optional[List[str]]       = default_tags
    uids          : Optional[List[str]]       = default_uids
    verbose       : bool                      = default_verbose

    class Config:
        orm_mode = True

# Path parameter class for 
class VerificationParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the parameters required for the verification / recognition process.
    """
    detector_name: FaceDetectorOptions = default_detector
    verifier_name: FaceVerifierOptions = default_verifier
    align        : bool                = default_align
    normalization: NormalizationTypes  = default_normalization
    metric       : DistanceMetrics     = default_metric
    threshold    : float               = default_threshold
    verbose      : bool                = default_verbose

    class Config:
        orm_mode = True


# ______________________________________________________________________________
#                          RESPONSE (API OUTPUT) CLASSES
# ------------------------------------------------------------------------------

# Response class for face regions
class Faces(BaseModel):
    faces: List[Tuple[int, int, int, int]]

# Response class for face verification matches
class VerificationMatches(BaseModel):
    unique_ids : List[UUID]
    name_tags  : List[str]
    image_names: List[str]
    image_fps  : List[str]
    regions    : List[List[int]]
    embeddings : List[List[float]]
    distances  : List[float]
    threshold  : float


# ______________________________________________________________________________
#                                  CUSTOM CLASSES
# ------------------------------------------------------------------------------
# Used internally in functions, may have methods.

# Stores the representation (embeddings) of a face image
class Representation():
    """
    Class to store the model-specific embeddings for an image.
    
    Attributes:
        After __init__:
            1. unique_id  - unique identifier number that represents a face
            2. image_name - image name
            3. image_fp   - image full path
            4. embeddings - model-specific vector representation of the face
        
    Methods:
        1. show_info() - prints the representation information in a condensed,
            easy-to-read form
    """
    def __init__(self, unique_id, image_name='', image_fp='', name_tag='',
                 region=[], embeddings={}):
        """
        Initializes the object with appropriate attributes
        """
        self.unique_id  = unique_id
        self.name_tag   = name_tag
        self.image_name = image_name
        self.image_fp   = image_fp
        self.region     = region
        self.embeddings = embeddings
        
    def show_info(self):
        """
        Shows detailed information about the Representation in a neat layout
        """
        print('Unique ID'.ljust(15) + f': {self.unique_id}',
              'Name'.ljust(15) + f': {self.name_tag}',
              'Image name'.ljust(15) + f': {self.image_name}',
              'Image full path'.ljust(15) + f': {self.image_fp}',
              'Face region'.ljust(15) + f': {self.region}', sep='\n')
        
        if self.embeddings: # embeddings dictionary is NOT empty
            print('Embeddings:')
            for k, v in self.embeddings.items():
                print(f'  > {k}: [{v[0]}, {v[1]}, {v[2]}, ... , {v[-1]}]'\
                    + f'(len={len(v)})')
                
        else:
            print('No embedding found!')

    def show_summary(self):
        """
        Shows summarized information about the Representation in a one-liner
        """
        print(f'UID: {self.unique_id}'.ljust(25),
              f'Name: {self.name_tag}'.ljust(15),
              f'Region: {self.region}'.ljust(15),
              f'FP: {self.image_fp}', sep=' | ')

# 
class VerificationResult():
    def __init__(self, unique_ids, image_names, image_fps, name_tags,
                 regions, embeddings, distances, threshold):
        self.unique_ids  = unique_ids
        self.image_names = image_names
        self.image_fps   = image_fps
        self.name_tags   = name_tags
        self.regions     = regions
        self.embeddings  = embeddings
        self.distances   = distances
        self.threshold   = threshold
        self.num_matches = len(self.unique_ids)



