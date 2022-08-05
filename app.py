from fastapi import FastAPI
from typing import List
from classifier import classify
from config.Config import Config
from data_structure import model_list
from data_structure.input_structure import InputStructure, BatchStructure
from data_structure.output_structure import OutputStructure, OutputStructure_verbose
from starlette.responses import JSONResponse


app = FastAPI()

@app.post("/predict")
async def Ramcell_classify(input_list: List[InputStructure]):
    #input_list = sorted(input_list, key=lambda input_t: input_t.id)
    input_total_cnt = len(input_list)
    print('{} inputs are requested'.format(len(input_list)))
    batch_dic = {}
    #spectra_for_predict = []
    for spectrum_id, input_t in enumerate(input_list):
        if input_t.model in batch_dic:
            batch_dic[input_t.model].intensity.append(input_t.intensity)
            batch_dic[input_t.model].raman_shift.append(input_t.raman_shift)
            batch_dic[input_t.model].spectrum_ids.append(spectrum_id)
            batch_dic[input_t.model].label.append(input_t.label)
            batch_dic[input_t.model].batch_size = max(batch_dic[input_t.model].batch_size + 1,500)
        else :
            batch_dic[input_t.model] = BatchStructure(
                                            intensity = [input_t.intensity],
                                            raman_shift = [input_t.raman_shift] ,
                                            spectrum_ids = [spectrum_id],
                                            label = [input_t.label],
                                            model = model_list[input_t.model],
                                            gpu_id = 0,
                                            batch_size = 1
                                            )
        #spectra_for_predict.append({'intensity' : input_t.intensity, 'raman_shift':input_t.raman_shift})

    #config = Config(model_fn=model_list[input_t.model], gpu_id=1, batch_size=max(input_total_cnt,500),spectra=spectra_for_predict)
    #classified_results = classify(config)

    classified_results = classify(batch_dic)
    print("classified_results = {}".format(classified_results))

    results = []
    for k, classified_result in classified_results.items():
        model_ = model_list[k]
        print("classified_result = {}".format(classified_result))
        if input_t.verbose == True:
            prob_all= []
            for j in range(classified_result[2].shape[0]):
                prob_all.append(classified_result[2][j][classified_result[0][j]])
            output_t = OutputStructure_verbose(
                intensity=batch_dic[model_].intensity,
                raman_shift=batch_dic[model_].raman_shift,
                acc = classified_result[1],
                pred = classified_result[0].tolist(),
                probability = prob_all,
                probability_all = classified_result[2].tolist(),
                model = model_,
                verbose = True
            )
        else :
            prob_all= []
            for j in range(classified_result[2].shape[0]):
                prob_all.append(classified_result[2][j][classified_result[0][j]])
            output_t = OutputStructure_verbose(
                acc = classified_result[1],
                pred = classified_result[0].tolist(),
                probability = prob_all,
                probability_all = classified_result[2].tolist(),
                verbose = False
            )
        results.append(output_t) #output_t : result of each batch in a request (a request can have multiple spectra)
    return results

@app.get("/input_cond")
async def return_shift_range_for_classify():
    return JSONResponse({
        'min_shift' : MINSHIFT,
        'max_shift' : MAXSHIFT,
        'input_dim' : INPUTDIM
        })
