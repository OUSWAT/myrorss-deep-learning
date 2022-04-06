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

### Data Exploration

relevant: review.py, plotlib.py

## Deep-Learning

Before training, first use load_data to load the ins and outs, with ID as whatever you choose. For example, the 2011 filtered dataset I call ID = "2011_qc". This is then entered into run_exp to pick the dataset to train on. 

### *U-Net*

Training on a 16k dataset with batch_size=44 takes over 2 hours. 

relevant: run_exp.py, run_exp_opt.py, u_net_loop.py


