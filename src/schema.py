"""
RPGCV Schemas
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """
    Input image and extra parameters
    """

    image_name: str = Field(..., example='img_3402.jpg', title='The base filename of the image you are uploading.')
    marker_type: str = Field(..., example='marker', title='The type of markers used in the image')
    vi_type: str = Field(..., example='RGBVI', title='The name of the vegetative index you want to use')
    # image: List[List[List[float]]]