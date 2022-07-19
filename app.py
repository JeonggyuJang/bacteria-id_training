from fastapi import FastAPI
from typing import List
from classifier import classify
from config.Config import Config
from data_structure.input_structure import InputStructure

app = FastAPI()

@app.post("/predict")
async def Ramcell_classify(input_list: List[InputStructure]):
    input_list = sorted(input_list, key=lambda input_t: input_t.id)
    print(input_list)
    spectra_for_predict = []
    for input_t in input_list:
        #spectra_for_predict.append(preprocess(input_t.text))
        spectra_for_predict.append(input_t.spectra_path)
    print("spectra_path = {}".format(spectra_for_predict))
    config = Config(model_fn="finetuned_22-07-17-11-22.ckpt", gpu_id=1, batch_size=1,spectra=spectra_for_predict)
    classified_spectra = classify(config)
    print("classified_spectra = {}".format(classified_spectra))
    classification_result = []
    for i, classified_spectrum in enumerate(classified_spectra):
        print("classified_spectrum = {}".format(classified_spectrum))
        input_t = InputStructure(
            id=input_list[i].id,
            spectra_path=spectra_for_predict[i],
            experiment = 0,
            acc = classified_spectrum[1],
            pred = classified_spectrum[0],
            probability = classified_spectrum[2][0][classified_spectrum[0]]
        )
        classification_result.append(input_t)
    return classification_result
