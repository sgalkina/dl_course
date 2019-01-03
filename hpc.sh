#!/bin/sh
# embedded options to bsub - start with #BSUB
# -- Name of the job ---
#BSUB -J Metabolites
### General options
### –- specify queue --
#BSUB -q gpuv100
# -- estimated wall clock time (execution time): hh:mm:ss --
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00
# request 100GB of memory
#BSUB -R "rusage[mem=100GB]"
# –- user email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u svegal@biosustain.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# -- run in the current working (submission) directory --
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
if test X$PBS_ENVIRONMENT = XPBS_BATCH; then cd $PBS_O_WORKDIR; fi
# here follow the commands you want to execute
nvidia-smi
#Install pytorch and torchvision
module load python3/3.6.2
module load cuda/9.2
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
module load scipy/0.19.1-python-3.6.2
module load pandas/0.20.3-python-3.6.2
module load matplotlib/2.0.2-python-3.6.2
pip3 install --user http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install --user torchvision
pip3 install --user pytorch-ignite
pip3 install --user scikit-learn

python3 exec_cnn2d.py