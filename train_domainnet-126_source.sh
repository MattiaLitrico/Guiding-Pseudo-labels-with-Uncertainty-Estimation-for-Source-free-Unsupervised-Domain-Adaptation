
SRC_DOMAIN=$1
DATA_ROOT=${2:-'data'}

python train_source.py --data_dir ${DATA_ROOT} --num_class 126 --batch_size 128 \
--dataset domainnet/${SRC_DOMAIN} --run_name domainnet_${SRC_DOMAIN}_source \
--lr 0.001 --wandb