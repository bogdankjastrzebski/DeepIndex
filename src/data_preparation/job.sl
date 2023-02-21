#!/bin/bash
#SBATCH --job-name=citation_network_data_prep # Job name
#SBATCH --mail-type=END,FAIL # Mail events
# (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bogdan.jastrzebski.dokt@pw.edu.pl # Where to send mail
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --gpus=0 
#SBATCH --mem=256gb # Job memory request
#SBATCH --time=01:00:00 # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log # Standard output and error log
pwd; hostname; date
echo "Running plot script on a single CPU core"
python3 data_preparation.py
date
