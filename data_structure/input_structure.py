from pydantic import BaseModel
from typing import Union, List

class InputStructure(BaseModel):
    intensity : List[float]
    raman_shift : List[float]
    label : int = 0
    verbose : bool = False
    model : str = "resnet14"

class BatchStructure(BaseModel):
    intensity : List[List[float]]
    raman_shift : List[List[float]]
    spectrum_ids : List[int] 
    label : List[int]
    model : str = "finetuned_22-07-17-11-22.ckpt"
    gpu_id : int = 0
    batch_size: int = 0 # 0 : auto
