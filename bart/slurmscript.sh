#!/bin/bash
#SBATCH --job-name=emade.db
#SBATCH -t 5-02:00        
#SBATCH --mem=60000# Runtime in D-HH:MM
#SBATCH -C "TeslaV100-PCIE-32GB"
#SBATCH --gres=gpu:1
module load anaconda2/4.4.0
source activate humor 
#python barttrain.py --traindataset="../humor_challenge_data/dedup1_train.txt" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="bart_fixed.out"  --outputdir="bart_fixed" --epochs=20 --gradient_acums=5 --maxseqlen=200 --batch=2 
#python barttrain.py --traindataset="../humor_challenge_data/dedup1_train.txt" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="bart_small_fixed.out"  --outputdir="bart_small_fixed" --epochs=20 --gradient_acums=5 --maxseqlen=200 --batch=2 
python barttrain.py --traindataset="../humor_challenge_data/challenge_dedup1_train.txt" --evaldataset="../humor_challenge_data/challenge_dedup1_test.txt" --outputfile="bart_challenge_fixed.out"  --outputdir="bart_challenge_fixed" --epochs=20 --gradient_acums=5 --maxseqlen=200 --batch=2 
#python barttrain.py --traindataset="../humor_challenge_data/dedup3_train.txt" --evaldataset="../humor_challenge_data/dedup3_test.txt" --outputfile="bart_dedup3_fixed.out"  --outputdir="bart_dedup3_fixed" --epochs=20 --gradient_acums=5 --maxseqlen=200 --batch=2 


