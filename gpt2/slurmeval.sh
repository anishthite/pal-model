#!/bin/bash
#SBATCH --job-name=emade.db
#SBATCH -t 5-02:00        
#SBATCH --mem=60000# Runtime in D-HH:MM
#SBATCH -C "TeslaV100-PCIE-32GB"
#SBATCH --gres=gpu:1
module load anaconda2/4.4.0
source activate humor 
#python gpt2eval.py --traindataset="" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="gpt2m_dedup1_fixed_bleu.out" --epochs=5 --gradient_acums=1 --maxseqlen=200 --batch=1 --model="gpt2m_dedup1_fixed/gpt2m_dedup1_fixed_2002105.pt" --config="gpt2m_dedup1/config.json" 
#python gpt2eval.py --traindataset="" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="dgpt2m_dedup1_fixed_bleu.out" --epochs=5 --gradient_acums=1 --maxseqlen=200 --batch=1 --model="dgpt2m_dedup1_fixed/dgpt2m_dedup1_fixed_2002104.pt" --config="dgpt2m_dedup1/config.json" 
#python gpt2eval.py --traindataset="" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="gpt2l_dedup1_fixed_bleu.out" --epochs=5 --gradient_acums=1 --maxseqlen=200 --batch=1 --model="gpt2l_dedup1_fixed/gpt2l_dedup1_fixed_200543.pt" --config="gpt2l_dedup1_fixed/config.json" 
python gpt2eval.py --traindataset="" --evaldataset="../humor_challenge_data/dedup1_test.txt" --outputfile="gpt2xl_dedup1_fixed_bleu.out" --epochs=5 --gradient_acums=1 --maxseqlen=200 --batch=1 --model="gpt2xl_dedup1_fixed/gpt2xl_dedup1_fixed_200521.pt" --config="gpt2xl_dedup1_fixed/config.json" 
#python gpt2eval.py --traindataset="" --evaldataset="../humor_challenge_data/dedup3_test.txt" --outputfile="gpt2m_dedup3_fixed_bleu.out" --epochs=5 --gradient_acums=1 --maxseqlen=200 --batch=1 --model="gpt2m_dedup3_fixed/gpt2m_dedup3_fixed_2002104.pt" --config="gpt2m_dedup3_fixed/config.json" 

