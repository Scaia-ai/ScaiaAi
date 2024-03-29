#!/bin/bash
cd ../test-data
mkdir sample
ls test | sort -R | tail -1000 | xargs -I % cp emails/% sample # generate a random sample of 1000
export DOC_TO_TEST=`ls sample | head -1` # example: enron001_01084.txt
cd ..
python -u docsimilarity.py input_data/sample input_data/sample/$DOC_TO_TEST