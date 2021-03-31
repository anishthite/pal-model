#!/bin/bash
#SBATCH --job-name=emade.db
#SBATCH -t 5-02:00        
#SBATCH --mem=60000# Runtime in D-HH:MM
#SBATCH -C "TeslaV100-PCIE-32GB"
#SBATCH --gres=gpu:1
module load anaconda2/4.4.0
source activate humor 
#python bettertrain.py --traindataset="../humor_challenge_data/dedup3_train.txt" --evaldataset="../humor_challenge_data/dedup3_test.txt" --outputfile="gpt2m_dedup3_fixed.out" --outputdir="gpt2m_dedup3_fixed" --epochs=20 --gradient_acums=2 --maxseqlen=200 --batch=10 
python bettertrain.py --traindataset="../humor_challenge_data/challenge_dedup1_train.txt" --evaldataset="../humor_challenge_data/challenge_dedup1_val.txt" --outputfile="gpt2m_dedup1_challenge.out" --outputdir="gpt2m_dedup3_challenge" --epochs=20 --gradient_acums=2 --maxseqlen=200 --batch=10 
#python bettertrain.py --traindataset="../humor_challenge_data/dedup1_train.txt" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="gpt2m_dedup1_pad.out" --outputdir="gpt2m_dedup1_pad" --epochs=20 --gradient_acums=2 --maxseqlen=200 --batch=10 
#python bettertrain.py --traindataset="../humor_challenge_data/dedup1_train.txt" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="dgpt2m_dedup1_fixed.out" --outputdir="dgpt2m_dedup1_fixed" --epochs=20 --gradient_acums=2 --maxseqlen=200 --batch=10 
#python bettertrain.py --traindataset="../humor_challenge_data/dedup1_train.txt" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="gpt2l_dedup1_fixed.out"  --outputdir="gpt2l_dedup1_fixed" --epochs=20 --gradient_acums=5 --maxseqlen=200 --batch=4 
#python bettertrain.py --traindataset="../humor_challenge_data/dedup1_train.txt" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="gpt2xl_dedup1_ffixed.out"  --outputdir="gpt2xl_dedup1_ffixed" --epochs=20 --gradient_acums=4 --maxseqlen=200 --batch=1 


