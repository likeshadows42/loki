# ==============================================================================
#                                 APP METHODS
# ==============================================================================

from fastapi                        import FastAPI
from .utility_functions             import create_dir
from .api_functions                 import DST_ROOT_DIR, RAW_DIR, GALLERY_DIR,\
                                           TARGETS_DIR

from api.routers.detection          import fd_router
from api.routers.verification       import fv_router
from api.routers.recognition        import fr_router
from api.routers.attribute_analysis import aa_router

app = FastAPI()

app.include_router(fd_router, prefix="/fd", tags=["Face Detection"])
app.include_router(fv_router, prefix="/fv", tags=["Face Verification"])
app.include_router(fr_router, prefix="/fr", tags=["Face Recognition"])
app.include_router(aa_router, prefix="/aa", tags=["Face Attribute Analysis"])

print(DST_ROOT_DIR)
print(RAW_DIR)
print(GALLERY_DIR)
print(TARGETS_DIR)

# -------------------------------- APP METHODS ---------------------------------
@app.on_event("startup")
async def initialize_database():
    # 
    for directory in [DST_ROOT_DIR, RAW_DIR, GALLERY_DIR, TARGETS_DIR]:
        print(f'Creating {directory} directory: ', end='')
    
        if create_dir(directory):
            print('directory exists. Continuing...')
        else:
            print('success.')
