"""
RPGCV pipeline deployment main script
"""

import os
import gc
import sys
import socket
import traceback
import torch
import uvicorn
import numpy as np

from PIL import Image
import io

from memory_profiler import profile
from joblib import load, dump
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status, File, UploadFile, Form, Depends
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.models import create_effdet_model, create_deeplabv3plus_model
from src.schema import InferenceInput

# Define ML models dictionary
ml_models = {}

# Define application startup event
@asynccontextmanager
async def startup_event(app: FastAPI):

    # Instantiate marker model
    marker_model = create_effdet_model().eval()
    # marker_model.load_state_dict(
    #     torch.load(
    #         CONFIG['MODEL_PATH'],
    #         map_location=CONFIG['DEVICE']
    #     )
    # )
    # marker_model.to(CONFIG['DEVICE'])

    # Instantiate deeplab model
    pgc_model = create_deeplabv3plus_model(num_classes=2, in_channels=3, encoder_depth=5).eval()
    # pgc_model = load_state_dict(
    #     torch.load(
    #         CONFIG['PGC_MODEL_PATH'],
    #         map_location=CONFIG['DEVICE']
    #     )
    # )
    # pgc_model.to(CONFIG['DEVICE'])

    ml_models['marker_model'] = marker_model
    ml_models['pgc_model'] = pgc_model

    yield 
    
    ml_models.clear()


app = FastAPI(
    lifespan=startup_event,
    title='PGC View Pipeline',
    description='The image analysis pipeline to take images '\
        'of perennial groundcover ROI plots in annual cropping '\
        'systems to return proportions of vegetation within the ROI',
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static folder for hosting any static files
app.mount("/static", StaticFiles(directory="static/"), name='static')


#################
# API ENDPOINTS #
#################

@app.get('/api/v1/status')
async def get_status():
    """
    Get the server information and status
    """

    response = {
        'server_status': {
            'server_name': socket.gethostname(),
            'message': 'Server is currently up',
            'API_docs': "http://0.0.0.0:8000/docs"
        }
    }

    return response

# @profile
# @torch.no_grad
@app.post('/api/v1/predict_markers')
async def predict_markers(file: UploadFile = File(...)):
    """
    Perform ROI marker prediction on the input image
    """

    logger.info('API POST predict_markers endpoint called')

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((1024, 1024))
    image_array = np.moveaxis(np.array(image), source=2, destination=0)
    input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    try:
        with torch.no_grad():
            output = ml_models['marker_model'](input_tensor)
    except Exception as e:
        logger.error(e)
        return {"error": e}

    output_list = output.squeeze(0).tolist()

    response = {
        'filename': file.filename,
        'data': output_list
    }

    del input_tensor
    del output
    torch.cuda.empty_cache()
    gc.collect()

    return response

@profile
@torch.no_grad
@app.post('/api/v1/predict_pgc')
async def predict_pgc(file: UploadFile = File(...)):
    """
    Perform ROI marker prediction on the input image
    """

    logger.info('API POST predict_markers endpoint called')

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((1024, 1024))
    image_array = np.moveaxis(np.array(image), source=2, destination=0)
    input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    try:
        with torch.no_grad():
            output = ml_models['pgc_model'](input_tensor)
    except Exception as e:
        logger.error(e)
        return {"error": e}

    output_list = output.squeeze(0).tolist()

    response = {
        'filename': file.filename,
        'data': output_list
    }

    del input_tensor
    del output
    torch.cuda.empty_cache()
    gc.collect()

    return response

    # logger.info(f"input: {request['image_name']}")
    
    # Prepare input data
    # X = torch.randn((1, 3, 1024, 1024))
    # X = np.array(data.image_array)

    # Conver image to tensor
    # input_tensor = torch.from_numpy(X).unsqueeze(0).to(CONFIG['DEVICE'])
    # input_tensor = torch.from_numpy(X).unsqueeze(0)

    # Run inference with both models
    # marker_out = ml_models['marker_model'](X).tolist()
    # pgc_out = ml_models['pgc_model'](X).tolist()

    # Post processing of the model output
    # results = {
    #     'data_output': 'Hello, looks like you hit the endpoint'
    # }

    # logger.info(f'results: {results}')

    # return {
    #     "data": marker_out
    # }

