
#!/bin/bash

#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=13:00:00
#PBS -j oe
#PBS -N train_target
#PBS -o /work/mlitrico/contrastive_test_time_adaptation/out/${PBS_JOBID}
#PBS -q fat
#to have dependendies PBS -W depend=afterany:89808


module load anaconda/3.2020.2
source activate venv

cd /work/mlitrico/contrastive_test_time_adaptation
#python Train_target_domainnet.py --dataset domainnet/clipart --source domainnet_painting_source --batch_size 128 --num_class 126 --lr 0.01 --run_name domainnet_p2c_new --num_epochs 1000 --num_neighbors 100 --wandb
python Train_target.py --dataset visdac/target --source visdac_source --batch_size 64 --num_class 12 --lr 0.01 --run_name visdac_source2target_ours_plot_error_pairs --num_epochs 60 --num_neighbors 100 --temporal_length 5 --wandb
