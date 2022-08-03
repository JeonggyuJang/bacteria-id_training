from fastapi import FastAPI
from typing import List
from classifier import classify
from config.Config import Config
from data_structure.input_structure import InputStructure
from starlette.responses import JSONResponse

app = FastAPI()

@app.post("/predict")
async def Ramcell_classify(input_list: List[InputStructure]):
    #input_list = sorted(input_list, key=lambda input_t: input_t.id)
    print(input_list)
    spectra_for_predict = []
    for input_t in input_list:
        #spectra_for_predict.append(preprocess(input_t.text))
        spectra_for_predict.append({'intensity' : input_t.intensity, 'raman_shift':input_t.raman_shift})
    #print("spectra_path = {}".format(spectra_for_predict))
    config = Config(model_fn="finetuned_22-07-17-11-22.ckpt", gpu_id=1, batch_size=1,spectra=spectra_for_predict)
    classified_spectra = classify(config)
    print("classified_spectra = {}".format(classified_spectra))
    classification_result = []
    for i, classified_spectrum in enumerate(classified_spectra):
        print("classified_spectrum = {}".format(classified_spectrum))
        input_t = InputStructure(
            intensity=spectra_for_predict[i]['intensity'],
            raman_shift=spectra_for_predict[i]['raman_shift'],
            acc = classified_spectrum[1],
            pred = classified_spectrum[0],
            probability = classified_spectrum[2][0][classified_spectrum[0]]
        )
        classification_result.append(input_t)
    return classification_result

@app.get("/input_cond")
async def return_shift_range_for_classify():
    return JSONResponse({
        'min_shift' : MINSHIFT,
        'max_shift' : MAXSHIFT,
        'input_dim' : INPUTDIM
        })
