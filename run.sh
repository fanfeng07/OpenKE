#!/bin/bash
#SBATCH --job-name=emb
#SBATCH --nodes=1
#SBATCH -t 0-11:59:59
#SBATCH --mem=MaxMemPerNode
#SBATCH --partition=kraken_fast
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

# activate conda environment kg
# dir: /home/user/feng/anaconda3/envs/kg
# pass current environment-variables to the submission system (e.g. using the "-V" flag  
# source /home/user/feng/.bashrc
# source activate kg

# source /home/user/feng/anaconda3/etc/profile.d/conda.sh
# conda activate kg
# ulimit -s unlimited

# CREATING SLURM NODE LISTS 
# export NODELIST=nodelist.$
# srun -l bash -c 'hostname' | sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " Starting..."
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE



# or
# source home/user/feng/anaconda3/envs/kg
# source $HOME/.bash_profile
# module load anaconda3

source /home/user/feng/anaconda3/etc/profile.d/conda.sh
conda activate kg

#...commands to be run before jobs starts...
# jarname=geo-prep-0.3.2-SNAPSHOT-standalone.jar

# workingDir=/home/slurm/novel-${SLURM_JOB_ID}

# WORKDIR=/home/user/feng/work/tool/OpenKE/
#echo "working directory = "$WORKDIR
#cd $WORKDIR
echo "Job ID is: "$SLURM_JOB_ID
# module load miniconda
# conda activate kg

#...copy data from home to work folder
# copy jar
# cp ~/geonames/$jarname $workingDir/
# cp ~/geonames/geonames_data_f1/"$SLURM_ARRAY_TASK_ID".nt $workingDir/
# cp ~/geonames/resources/all-GB-SUBJ.txt $workingDir/
# echo "Copying Files "$SLURM_ARRAY_TASK_ID".nt to working directory"$workingDir

#Running the Command
echo "Task Starting"
# touch test.txt
cd work/tool/OpenKE/
python -u train_transe_FB13.py > test2.out
echo "Task Finished"

# java -jar $workingDir/$jarname $workingDir/all-GB-SUBJ.txt $workingDir/"$SLURM_ARRAY_TASK_ID".nt $workingDir/GB-Statement-"$SLURM_ARRAY_TASK_ID".nt
# deactivate
# echo "Running Job-ID:"$SLURM_ARRAY_TASK_ID

#Copy Back Data (Backup)

# cp $workingDir/GB-Statement-"$SLURM_ARRAY_TASK_ID".nt  ~/geonames/results/
