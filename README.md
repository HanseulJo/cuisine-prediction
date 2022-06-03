# AI506 Term Project: Predicting Cuisine from Ingredients

## Data Pre-processing

If you run `data_preproc/make_embedding_from_raw_data.py,` you will get binary/integer embeddings of raw data.

## Data Analysis

Refer to the file ```DataAnalysis/data_EDA.ipynb```

## Collaborative Filtering (CF)

Open `CF/ALS_CF_tutorial.ipynb` to train and test CF model.


## Graph Based Model (Graph)

Open `Graph/Graph_base_tutorial.ipynb` to train and test CF model.


## Fully-Connected Neural Net based Mode (FCN)

Requirements are listed in ```requrements.txt```

### Training
Run ```FCN/DNN_train.py```
e.g.
```python DNN_train.py -t classification -d '../Container/' -b 16 -e 50 -lr 1e-3 -step 10 -f [2048,1024,512,256] -w true```

### Inference
Run ```FCN/DNN_inference.py```
e.g.
```python DNN_inference.py -t completion -n valid_cpl -d '../Container/' -b 16 -f [1024,1024,512,512]```


## Encode, Pool, and Classify by Neural Net (EPCnet)

### Training
```
cd EPCnet
python3 run.py --data-dir ../Container --batch_size 64 --batch_size_eval 2048 --n_epochs 100 --lr 1e-4 --weight-decay 0.01 --dim-embedding 512 --dim-hidden 512 --dropout 0.1 --encoder-mode HYBRID --pooler-mode PMA --cpl-scheme pooled --num-enc-layers 8 --num-dec-layers 1 --loss MultiClassASLoss --optimizer-name AdamW --classify --complete --save_model
```
### Inference
You need checkpoint file.

Example: Cuisine Classification
```
cd EPCnet
python3 test.py -p weights/ckpt_CCNet_clf_EncFC_PoolPMA_CplNone_NumEnc7_NumDec1_Hid512_Emb512_Ind10_Loss0.788_Acc0.766_Topk0.959_F1macro0.687_F1micro0.766_BestEpoch61.pt --dim-embedding 512 --dim-hidden 512 --encoder-mode FC --pooler-mode PMA --num-enc-layers 7 --num-dec-layers 1 --classify
```


## Ensemble

- `Ensemble/ensemble_GD_clf.ipynb` --> Train an ensembling model for cuisine classification
- `Ensemble/ensemble_GD_cpl.ipynb` --> Train an ensembling model for recipe completion
- `Ensemble/ensemble_infer_clf.ipynb` --> Perform inference with ensembled model on cuisine classification
- `Ensemble/ensemble_infer_cpl.ipynb` --> Perform inference with ensembled model on recipe completion




