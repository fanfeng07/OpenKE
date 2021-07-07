#!/bin/bash
#SBATCH --job-name=emb
#SBATCH --array=1-16%4
#SBATCH --nodes=2
#SBATCH --cpus-per-task=20
#SBATCH -t 0-11:59:59
#SBATCH --mem=MaxMemPerNode
#SBATCH --partition=kraken_fast
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH --output=out/arrayJob_%A_%a.out
#SBATCH --error=out/arrayJob_%A_%a.err


# activate conda environment kg
# dir: /home/user/feng/anaconda3/envs/kg
# pass current environment-variables to the submission system (e.g. using the "-V" flag  
source /home/user/feng/.bashrc
source activate kg

# or
# source home/user/feng/anaconda3/envs/kg


# ...commands to be run before jobs starts...
# jarname=geo-prep-0.3.2-SNAPSHOT-standalone.jar

# workingDir=/home/slurm/novel-${SLURM_JOB_ID}
WORKDIR=/home/user/feng/work/tool/OpenKE/
echo "working directory = "$WORKDIR
cd $WORKDIR
echo "Job id" $SLURM_JOB_ID
# module load miniconda
# conda activate kg

# ...copy data from home to work folder
# copy jar
# cp ~/geonames/$jarname $workingDir/
# cp ~/geonames/geonames_data_f1/"$SLURM_ARRAY_TASK_ID".nt $workingDir/
# cp ~/geonames/resources/all-GB-SUBJ.txt $workingDir/
# echo "Copying Files "$SLURM_ARRAY_TASK_ID".nt to working directory"$workingDir

# Running the Command

python train_transe_FB13.py
echo "Task Finished"

# java -jar $workingDir/$jarname $workingDir/all-GB-SUBJ.txt $workingDir/"$SLURM_ARRAY_TASK_ID".nt $workingDir/GB-Statement-"$SLURM_ARRAY_TASK_ID".nt
# deactivate
# echo "Running Job-ID:"$SLURM_ARRAY_TASK_ID

# Copy Back Data (Backup)

# cp $workingDir/GB-Statement-"$SLURM_ARRAY_TASK_ID".nt  ~/geonames/results/