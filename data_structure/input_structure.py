from pydantic import BaseModel
from typing import Union

class InputStructure(BaseModel):
    id : int
    experiment : int
    spectra_path : str 
    pred : int
    acc : float
    probability : Union[float,None] = None


