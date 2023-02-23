#!/bin/bash
#SBATCH --job-name=MF # Job name
#SBATCH --mail-type=END,FAIL # Mail events
# (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bogdan.jastrzebski.dokt@pw.edu.pl # Where to send mail
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --mem=32gb # Job memory request
#SBATCH --time=00:00:10 # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log # Standard output and error log
pwd; hostname; date
echo "Start of the procedure..."
python3 extract.py
echo "Done."
date
