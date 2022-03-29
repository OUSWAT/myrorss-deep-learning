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

To run multiple, use outer_loop.sh (adjust start and end), which calls ext_shell.sh, then calls extract_shell.py for each day. 

## Data Exploration

relevant: review.py, load_data.py, plotlib.py

#### *U-Net*

relevant: run_exp.py, run_exp_opt.py, u_net_loop.py

Trains U-Net 
