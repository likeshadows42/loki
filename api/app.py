from fastapi import FastAPI

from v1.routers.detection   import fd_router
from v1.routers.recognition import fr_router

app = FastAPI()

app.include_router(fd_router, prefix="/fd", tags=["Face Detection"])
app.include_router(fr_router, prefix="/fr", tags=["Face Recognition"])