from pydantic import BaseModel
from typing import Union, List


class InputStructure(BaseModel):
    intensity : List[float]
    raman_shift : List[float]
    pred : int
    acc : float
    probability : Union[float,None] = None


