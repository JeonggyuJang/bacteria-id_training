from pydantic import BaseModel
from typing import Union, List


class InputStructure(BaseModel):
    intensity : List[float]
    raman_shift : List[float]
    pred : Union[int,None] = None
    verbose : bool = False



