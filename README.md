# Repository for facilitating and practicing deep-learning on the MYRORSS (1999-2011) Dataset!

This repository contains python scripts useful for converting MYRORSS (tars of radar days) into training samples stored as npys. 
## Table of Contents

**[Background](#background)**<br>
**[Data Extraction/Transformation/Loading on supercomputer via slurm/python/wdss-ii](#placeholder)**<br>
--[U-Net](#vmrms-processing-details)<br>

---

## Background

MYRORSS reflectivity fields + 30-min max swaths of AzShear, Ref_0C + NSE (hourly) -> U-Net training -> next 30min MESH swath 

## Data Extraction/Transformation/Loading on supercomputer via slurm/python/wdss-ii

relevant: extract_shell.py, ext_shell.sh, outer_loop.sh

Follow the main loop in extract_shell.py. 

extract comp ref and MESH to lscratch (2 tasks) -> find storms with localmax -> checkMESH and update csv -> fully extract remaining storms (all fields to netcdf.gz from tarball) -> accumulate (still in lscratch) -> crop and write to /data directory 

### *Notes on checkMESH()*: 

checkMESH checks the MESH field within a 60x60 pixel box centered on maxes in comp ref (using localmax). Critically, checkMESH() calls decide(), which contains the strategy for deciding whether or not to keep the sample. 

Decide strategy: count number of pixels within a (smaller) 50x50 pixel box centered on max composite reflectivity. Count number of pixels with value > 20 mm. If there are 25 or more pixels above this threshold, extract the sample fully. This function is fundamental to forming the dataset and optimally training a predictor.

### Extracting data for multiple days

To run multiple, use outer_loop.sh (adjust start and enddate), which calls ext_shell.sh, then calls extract_shell.py for each day. 

## Data Exploration

When comparing SHAVE and 2011_qc datasets, I get a list ```n_list``` which contains the number of pixels above a chosen threshold for every image in the dataset. The 2011_qc dataset has a much higher proportion of samples with very high (> 1000/3600 pixels) above threshold. A proportion of these could be removed.

Also, the 2011_qc dataset contains no targets where max_MESH < 20, while SHAVE has no such restraint. 

### Dataset Creation

Year: 2010
Months: 05
decide: 100 pixels with MESH > 20 within inner 40x40 pixel box 
### Datasets

Datasets are saved in /myrorss-deep-learning/datasets. ID's include '2011_qc', 'shave'. 

Note that, for any ins, the MESH_Max_30min swath is the last field, i.e. ins[:,:,:,-1]. 

relevant: review.py, plotlib.py

## Deep-Learning

Before training, first use load_data to load the ins and outs, with ID as whatever you choose. For example, the 2011 filtered dataset I call ID = "2011_qc". This is then entered into run_exp to pick the dataset to train on. 

### *U-Net*

Training on a 16k dataset with batch_size=44 takes over 2 hours. Compare with other batch_sizes, if performance increases enough to warrant the possible increase in compute-time.

After training a model, a pickle (python object serialization, pickling is the process whereby a python object hierarchy is converted into a byte stream) will be saved to the /myrorss-deep-learning/results directory. If ID = '2011_qc', the results are saved in 2011_qc_results.pkl. From util module, import open_pickle and load up the pickle. This pickle contains a dictionary with the images used in training and testing sets, as well as the predictions for these sets. More information is there, as you can see by typing 'results_pickle.keys()'. For our purposes, we want the r['predict_testing'], the predictions for the test set, and r['true_testing'], the ground truth for the test set.

From here, we use stats.py to run basic statistics on how well the model performed. To start, we use common metrics used in the field of pixelwise-prediction on images, like the Probability of Detection (POD), False Alarm Rate (FAR), and Critical Success Ratio (see Roebber 2009). Additionally, one needs to load the scaler that was used to scale the ins and outs to be between -1 and 1 (or 0 and 1, I forget). The scaler is a pickle, opened in the same way, saved in /scalers as scaler_ins_2011_qc.pkl. Then, to collect the stats, use stats() from stats.py, loading it up like: 

```
import util
r, scaler = util.op(ID)
container = stats(r['true_testing'], r['predict_testing'], scaler)
POD, FAR, bias, CSI = [x for x in container]
print([x for x in container]
```
relevant: run_exp.py, run_exp_opt.py, u_net_loop.py, stats.py

Results:

Dataset_ID: shave, N ~ 3.5 k, 2 layers
POD: 0.44, FAR: ~ 0.2
Conclusion: The training appears to be happening differently on the supercomputer. Or there is an error in the process. Solution: Replicate on swat-machine. Verirfy model architecture and parameters. 
Param: 586 k

Dataset_ID: 2011_qc, N ~ 15930, 
POD: 0.22, FAR: 0.5, CSI: .44
Architecture:  2 steps (60 -> 30 -> 15 -> 30 -> 60)
Param: 586 k 
train-time: >18 hr (check history)
