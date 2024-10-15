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
import cv2
import json
import numpy as np
from dotenv import load_dotenv
from typing import Optional, List, Union

from PIL import Image
import io

import torch.nn.functional as f

from memory_profiler import profile
from joblib import load, dump
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status, File, UploadFile, Form, Depends, Query, HTTPException
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.models import create_effdet_model, create_deeplabv3plus_model
from src.schema import RawResponse, StatusResponse, MarkerInput, PgcInput, TestResponse, MarkerTypes, MarkerFilteredResponse, MarkerData, PgcFilteredResponse
from src.transforms import get_tensor_transforms, get_seg_transforms

# Define ML models dictionary
ml_models = {}

# Set environment variables
load_dotenv()

# Set the computational device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define application startup event
@asynccontextmanager
async def startup_event(app: FastAPI):

    # Instantiate marker model
    marker_model = create_effdet_model(device)

    # Instantiate deeplab model
    pgc_model = create_deeplabv3plus_model(device)

    # Add models and transforms to model dictionary
    ml_models['marker_model'] = marker_model
    ml_models['pgc_model'] = pgc_model
    ml_models['marker_transforms'] = get_tensor_transforms()
    ml_models['pgc_transforms'] = get_seg_transforms()

    # Yield to the application
    yield 
    
    # Clean up the models before shutting down
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

@app.get('/api/v1/status', response_model=StatusResponse)
async def get_status():
    """
    Get the server information and status
    Returns a StatusResponse response model
    """

    response = StatusResponse(
        server_name=socket.gethostname(),
        message='The server is currently up',
        API_docs=os.getenv('API_DOCS_URL'),
        computation_device=f"{device}",
        device_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None
    )

    return response

@app.post('/api/v1/test_server')
async def get_test_response(input: TestResponse):
    items = input.model_dump()
    print(items)

    return input

@app.post('/api/v1/predict_markers', response_model=Union[RawResponse, MarkerFilteredResponse])
async def predict_markers(
    raw: bool = Query(False, description="Return all of the raw predictions"),
    file: UploadFile = File(...),
    marker_type: Optional[MarkerTypes] = Form(None),
    threshold: Optional[float] = Form(0.5)
    ):
    """
    Perform ROI marker prediction on the input image

    Query Parameters:
        raw: bool=False Return the raw predictions or processed
    
    Form Fields:
        marker_type: Optional[MarkerTypes] The marker type used in the image. Defaults to None.
        threshold: Optional[float] The threshold used for the prediction confidence in the interval [0, 1]. Defaults to 0.5.
    """
    # Define class map
    class_map = {'2': 'marker', '1': 'quadrat'}

    # Read the image, transform, and run inference
    try:
        # Read in the file and convert to Numpy
        contents =  await file.read()
        image = np.array(Image.open(io.BytesIO(contents)))

        # Convert channels and normalize
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        transformed = ml_models['marker_transforms'](image=image)

        # Add a batch dimension and send to device
        input_tensor = transformed['image'].unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = ml_models['marker_model'](input_tensor)
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error during inference: str({e})"
        )

    # Check for raw or filtered responses
    if raw:
        return RawResponse(
            filename=file.filename,
            data=output.squeeze(0).tolist()
        )
    else:
        if threshold < 0.0 or threshold > 1.0:
            raise HTTPException(400, "Threshold is out of bounds. Should be in the interval [0, 1]")
        
        # Remove batch dim and filter for marker_type
        output = output.squeeze(0)
        if marker_type:
            logger.info(f"Filtering the output for marker type '{marker_type}'")
            for k, v in class_map.items():
                if v == marker_type.value:
                    target_class = float(k)
                    # Subset the output array with the target class
                    output = output[torch.where(output[:, 5] == target_class)]
            
        # Get output threshold idx (defaults to 0.5)
        thresh_idx = torch.ge(output[:, -2], threshold)
        output = output[thresh_idx]
        # Get the Class tensor and map to strings
        class_tensor = output[:4, 5].to(torch.int)
        classes = [class_map.get(str(i.item())) for i in class_tensor] # map to strings
        # Subset the coordinates for the top 4 predictions
        coords = output[:4, :4].tolist()

        # Clean up the call
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return MarkerFilteredResponse(
            filename=file.filename,
            data=MarkerData(
                coordinates=coords,
                classes=classes
            )
        )
        
@app.post('/api/v1/predict_pgc', response_model=Union[RawResponse, PgcFilteredResponse])
async def predict_pgc(
    raw: bool = Query(False, description="Return all of the raw predictions or just the cleaned predictions"),
    file: UploadFile = File(...),
    exclude: Optional[str] = Form(None)
    ):
    """
    Perform ROI marker prediction on the input image
    """

       # Read the image, transform, and run inference
    try:
        # Read in the file and convert to Numpy
        contents =  await file.read()
        image = np.array(Image.open(io.BytesIO(contents)))

        # Convert channels and normalize
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        transformed = ml_models['pgc_transforms'](image=image)

        # Add a batch dimension and send to device
        input_tensor = transformed['image'].unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = ml_models['pgc_model'](input_tensor).squeeze(0)
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error during inference: str({e})"
        )

    # Check for raw or filtered responses
    if raw:
        return RawResponse(
            filename=file.filename,
            data=output.tolist()
        )
    else:
        softmax = f.softmax(output, 0)
        preds = torch.argmax(softmax, 0)

        if exclude:
            try:
                exclude_list = json.loads(exclude)
                if not isinstance(exclude_list, list):
                    raise ValueError
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid format for 'exclude', must be a list of integers.")
            print(exclude_list)
            for i in exclude_list:
                preds = torch.where(preds==i, 0, preds)

        return PgcFilteredResponse(
            filename=file.filename,
            data=preds.tolist()
        )




    # Check the query parameters and return the formatted data
    # if raw:
    #     output_list = output.squeeze(0).tolist()
    # else:
    #     softmax = f.softmax(output, 1)
    #     preds = torch.argmax(softmax, 1)
    #     output_list = preds.squeeze(0).tolist()
    # response['data'] = output_list

    # # Clean up the call
    # del input_tensor
    # del output
    # del output_list
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    # gc.collect()

    # return response
