
DATA_ROOT=${1:-'data'}

python train_source.py --data_dir ${DATA_ROOT} \
--dataset visdac/source --run_name visdac_source \
--num_class 12 --lr 0.001 --batch_size 64 --wandb