from fastapi import FastAPI
from typing import List
from classifier import classify
from config.Config import Config
from data_structure.input_structure import InputStructure
from data_structure.output_structure import OutputStructure, OutputStructure_verbose
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
    classified_results = classify(config)
    print("classified_results = {}".format(classified_results))
    results = []
    for i, classified_result in enumerate(classified_results):
        print("classified_result = {}".format(classified_result))
        if input_t.verbose == True:
            output_t = OutputStructure_verbose(
                intensity=spectra_for_predict[i]['intensity'],
                raman_shift=spectra_for_predict[i]['raman_shift'],
                acc = classified_result[1],
                pred = classified_result[0].tolist(),
                probability_all = classified_result[2].tolist(),
                verbose = True
            )
            prob_all= []
            for j in range(pred.shape[0]):
                prob_all.append(classified_result[2][j][classified_result[j]])
            output_t.probability = prob_all
        else :
            output_t = OutputStructure_verbose(
                acc = classified_result[1],
                pred = classified_result[0].tolist(),
                probability_all = classified_result[2].tolist(),
                verbose = False
            )
            prob_all= []
            for j in range(pred.shape[0]):
                prob_all.append(classified_result[2][j][classified_result[j]])
            output_t.probability = prob_all
        results.append(output_t) #output_t : result of each batch in a request (a request can have multiple spectra)
    return results

@app.get("/input_cond")
async def return_shift_range_for_classify():
    return JSONResponse({
        'min_shift' : MINSHIFT,
        'max_shift' : MAXSHIFT,
        'input_dim' : INPUTDIM
        })
