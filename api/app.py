from fastapi import FastAPI

from api.routers.detection          import fd_router
from api.routers.recognition        import fr_router
from api.routers.attribute_analysis import aa_router

app = FastAPI()

app.include_router(fd_router, prefix="/fd", tags=["Face Detection"])
app.include_router(fr_router, prefix="/fr", tags=["Face Recognition"])
app.include_router(aa_router, prefix="/aa", tags=["Face Attribute Analysis"])
