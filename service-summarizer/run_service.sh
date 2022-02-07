#!/bin/bash
conda activate scaia
nohup bash run_app.sh > log.txt 2>&1 &
