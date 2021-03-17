#!/bin/bash
conda activate scaia
nohup bash run_app.py > log.txt 2>&1 &
