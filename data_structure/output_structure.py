from pydantic import BaseModel
from typing import Union, List


class OutputStructure(BaseModel):
    pred : List[int]
    acc : float
    probability : List[float] 
    probability_all : List[List[float]]
    model : str
    verbose : bool = False

class OutputStructure_verbose(BaseModel):
    intensity : List[List[float]]
    raman_shift : List[List[float]]
    pred : List[int]
    acc : float
    probability : List[float] 
    probability_all : List[List[float]]
    model : str
    verbose : bool = True

