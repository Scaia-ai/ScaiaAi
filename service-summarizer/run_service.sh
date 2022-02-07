#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
nohup  ./run_app.sh > log.txt 2>&1 &
