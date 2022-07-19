class Config:
    def __init__(self, model_fn, gpu_id, batch_size, spectra):
        self.model_fn = model_fn
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.spectra = spectra
        self.top_k = 1
