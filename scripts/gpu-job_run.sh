#!/bin/bash

#rm -rf condor_log

# interactive mode (debug only)
#condor_submit -i scripts/chtc/job-gpu.sub

# run a single experiment
#python src/train.py experiment=exp_mlp-mnist.yaml

#rm logs/condor_logs/*

# all experiments
condor_submit experiment=darwin scripts/gpu-job.sub