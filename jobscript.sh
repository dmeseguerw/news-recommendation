#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J testjob
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=1GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s233668@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Set variables (ensure they are set dynamically when submitting the job)
UNO=${UNO:-1}  # Default to 1 if UNO is not set
DOS=${DOS:-2}  # Default to 2 if DOS is not set

nvidia-smi
# Load the cuda module
#module load cuda/11.6

#/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
python3 hpctest.py "$UNO" "$DOS" > outputtest.txt
#hpctest.py "${UNO}" "${DOS}" > outputtest.txt
