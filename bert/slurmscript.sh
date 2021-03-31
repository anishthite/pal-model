#!/bin/bash
#SBATCH --job-name=emade.db
#SBATCH -t 5-02:00        
#SBATCH --mem=60000# Runtime in D-HH:MM
#SBATCH -C "TeslaV100-PCIE-32GB"
#SBATCH --gres=gpu:1
module load anaconda2/4.4.0
source activate humor 
python betterberthumorclassifier.py --traindataset="../humor_challenge_data/bertfinaltrain.tsv" --devdataset="../humor_challenge_data/bertfinalval.tsv" --testdataset="../humor_challenge_data/bertfinaltest.tsv" --outputfile="bert.out" --epochs=2
