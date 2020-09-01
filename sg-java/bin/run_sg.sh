#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
java -cp $DIR/../sg/target/sg-1.0.0-bin.jar org.freeeed.sg.Sg
