"""
RPGCV Schemas
"""

from enum import Enum
from typing import List, Union
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl

# Enumerations
class MarkerTypes(Enum):
    """
    Enumerate the different marker types
    """

    marker: str = 'marker'
    quadrat: str = 'quadrat'

# Pydantic Classes
class StatusResponse(BaseModel):
    """
    Server status response
    """

    server_name: str
    message: str
    API_docs: HttpUrl
    computation_device: str
    device_name: Union[str, None]

class TestResponse(BaseModel):
    """
    Test the data response
    """

    name: str
    message: str
    option1: str
    option2: int


class MarkerInput(BaseModel):
    """
    Input image and extra parameters for the marker model
    """

    marker_type: Optional[str] = MarkerTypes
    threshold: Optional[float] = 0.5

class PgcInput(BaseModel):
    """
    Input image and extra parameters for the PGC model
    """

    exclude_classes: Optional[List[int]]

class RawResponse(BaseModel):
    """
    Image model raw response
    """

    filename: str
    data: Union[List[List[float]], List[List[List[float]]]]

class MarkerData(BaseModel):
    """
    Data object for the filtered response
    """

    coordinates: List
    classes: List

class MarkerFilteredResponse(BaseModel):
    """
    Filtered marker model response
    """
    filename: str
    data: MarkerData

class PgcFilteredResponse(BaseModel):
    """
    Filtered PGC model response
    """

    filename: str
    data: List[List[int]]

