# ==============================================================================
#                                   API CLASSES
# ==============================================================================

from enum                       import Enum
from typing                     import List, Tuple, Optional
#from xmlrpc.client import Boolean
from pydantic                   import BaseModel

from sqlalchemy                 import Column, String, Integer,\
                                        PickleType, ForeignKey, Boolean
from sqlalchemy.orm             import relationship
from sqlalchemy.dialects.mysql  import INTEGER
from sqlalchemy.ext.declarative import declarative_base

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
    """
    Path parameter class: available face detector names
    """
    OPENCV = "opencv",
    SSD    = "ssd",
    MTCNN  = "mtcnn",
    RETINA = "retinaface"

# Path parameter class for face detector name options
class FaceVerifierOptions(str, Enum):
    """
    Path parameter class: available face verifier names
    """
    VGGFace    = "VGG-Face"
    FACENET    = "Facenet"
    FACENET512 = "Facenet512"
    OPENFACE   = "OpenFace"
    DEEPFACE   = "DeepFace"
    DEEPID     = "DeepID"
    ARCFACE    = "ArcFace"

# Path parameter class for distance metric options
class DistanceMetrics(str, Enum):
    """
    Path parameter class: available distance metrics.
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

# Path parameter class for available representation properties
class AvailableRepProperties(str, Enum):
    """
    Path parameter class: available properties of the Representation class.
    """
    UNIQUE_ID  = "unique_id"
    IMAGE_NAME = "image_name"
    IMAGE_FP   = "image_fp"
    GROUP_NO   = "group_no"
    NAME_TAG   = "name_tag"
    REGION     = "region"
    EMBEDDINGS = "embeddings"

# Path parameter class for available tables in the database
class AvailableTables(str, Enum):
    """
    Path parameter class: available tables in the database.
    """
    PERSON    = "person"
    FACEREP   = "representation"
    PROCFILES = "proc_files"

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
default_auto_grouping     = True
default_eps               = 0.5
default_min_samples       = 2
default_pct               = 0.02
default_check_models      = True
default_verbose           = False
default_property          = AvailableRepProperties.NAME_TAG
default_table_name        = AvailableTables.FACEREP

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
    auto_grouping : Optional[bool]            = default_auto_grouping
    eps           : Optional[float]           = default_eps
    min_samples   : Optional[int]             = default_min_samples
    metric        : Optional[DistanceMetrics] = default_metric
    pct           : Optional[float]           = default_pct
    check_models  : Optional[bool]            = default_check_models
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
    pct          : float               = default_pct
    verbose      : bool                = default_verbose

    class Config:
        orm_mode = True

# ______________________________________________________________________________
#                          RESPONSE (API OUTPUT) CLASSES
# ------------------------------------------------------------------------------

# Response class for face regions
class Faces(BaseModel):
    """
    Response model class: defines the output for face regions.
    """
    faces: List[Tuple[int, int, int, int]]

# Response class for face verification matches
class VerificationMatch(BaseModel):
    """
    Response model class: defines the output for face verification match.
    """
    unique_id : int
    image_name: str
    person_id : Optional[int]
    image_fp  : str
    region    : List[int]
    embeddings: List[str]
    distance  : float

# Response class for the output of the representation table
class FaceRepOutput(BaseModel):
    """
    Response model class: defines the output for the representation table
    information output.
    """
    id              : int
    person_id       : int = None
    image_name      : str
    image_fp        : str
    group_no        : int
    region          : List[int]
    embeddings      : List[str]
    hidden          : bool

# Response class for the output of the person table
class PersonTableOutput(BaseModel):
    """
    Response model class: defines the output for the person table information
    output.
    """
    id       : int
    name     : str
    group_no : int
    note     : str
    hidden   : bool

# Response class for the output of the processed files table
class ProcessedFilesOutput(BaseModel):
    """
    Response model class: defines the output for the processed files table
    information output.
    """
    id       : int
    filename : str
    filesize : int

# ______________________________________________________________________________
#                         SQLALCHEMY TABLES DEFINITIONS
# ------------------------------------------------------------------------------

Base = declarative_base()

class Person(Base):
    """
    Person data structure

    Fields:
        person_id
        name
        note
    """
    # Table name
    __tablename__ = 'person'

    # Object attributes (as database columns)
    id        = Column(Integer, primary_key=True)
    name      = Column(String , default=None)
    group_no  = Column(Integer, default=None)
    note      = Column(String , default=None)
    front_img = Column(Integer, default=None)
    hidden    = Column(Boolean, nullable=False, default=False)

    # Establishes connection to associated Face Representations
    reps = relationship("FaceRep", back_populates="person",
                        cascade="all, delete, delete-orphan") # important for deleting children
    
    # Standard repr for the class
    def __repr__(self):
        return "(id=%s) - %s\n%s" % (self.id, self.name, self.note)

class FaceRep(Base):
    """
    Initializes the object with appropriate attributes
    """
    # Table name
    __tablename__ = 'representation'

    # Object attributes (as database columns)
    id         = Column(Integer   , primary_key=True)
    person_id  = Column(Integer   , ForeignKey('person.id'), default=None)
    image_name = Column(String    , nullable=False)
    image_fp   = Column(String    , nullable=False)
    group_no   = Column(Integer   , nullable=False)
    region     = Column(PickleType, nullable=False)
    embeddings = Column(PickleType, nullable=False)
    hidden     = Column(Boolean   , nullable=False, default=False)

    # Establishes connection to associated Person
    person = relationship("Person", back_populates="reps")

    # Standard repr for the class
    def __repr__(self):
        return "(id=%s)\nimage name: %s\nimage path: %s\ngroup: %s" % (self.id,
                    self.image_name, self.image_fp, self.group_no)

class ProcessedFiles(Base):
    """
    Initializes the object with appropriate attributes
    """
    # Table name
    __tablename__ = 'proc_files'

    # Object attributes (as database columns)
    id       = Column(Integer, primary_key=True)
    filename = Column(String, default=None)
    filesize = Column(INTEGER(unsigned=True), default=None)

    # Standard repr for the class
    def __repr__(self):
        return "(id=%s)\nfile name: %s\nfile size: %s" % (self.id,
                    self.filename, self.filesize)

class ProcessedFilesTemp(Base):
    """
    Initializes the object with appropriate attributes
    """
    # Table name
    __tablename__ = 'proc_files_temp'

    # Object attributes (as database columns)
    id       = Column(Integer, primary_key=True)
    filename = Column(String, default=None)
    filesize = Column(INTEGER(unsigned=True), default=None)

    # Standard repr for the class
    def __repr__(self):
        return "(id=%s)\nfile name: %s\nfile size: %s" % (self.id,
                    self.filename, self.filesize)

class tempClustering(Base):
    """
    Initializes the object with appropriate attributes
    """
    # Table name
    __tablename__ = 'temp_clustering'

    # Object attributes (as database columns)
    id       = Column(Integer, primary_key=True)
    group_no = Column(Integer, default=None)

