# ______________________________________________________________________________
#                                       APP
# ------------------------------------------------------------------------------

import os
from svr_init   import manager_dict as mng_dict

from fastapi                    import FastAPI
from fastapi.middleware.cors    import CORSMiddleware

manager_dict = mng_dict

# ______________________________________________________________________________
#                               APP INITIALIZATION
# ------------------------------------------------------------------------------

app     = FastAPI(name='Face Recognition API')
origins = ['http://localhost:8080']

app.add_middleware(
    CORSMiddleware,
    allow_origins     = origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ______________________________________________________________________________
#                                  APP ENDPOINTS
# ------------------------------------------------------------------------------

@app.get("/get_variable")
async def get_variable():
    # global manager_dict
    # worker_pid = os.getpid()

    # print(f'Worker {worker_pid}:')
    for key, value in manager_dict.items():
        print(f' > {key}:', value)

    return {'msg':'ok'}

# ------------------------------------------------------------------------------

# @app.post("/debug/inspect_globals")
# async def example_func():
#     """
#     Documentation goes here...
#     """
#     # Code goes here...

#     return None

# ------------------------------------------------------------------------------

