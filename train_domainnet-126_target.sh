
SRC_DOMAIN=$1
TGT_DOMAIN=$2
DATA_ROOT=${3:-'data'}

python train_target.py --data_dir ${DATA_ROOT} --dataset domainnet/${TGT_DOMAIN} \
--source domainnet_${SRC_DOMAIN}_source \
--run_name domainnet_${SRC_DOMAIN}2${SRC_DOMAIN} --batch_size 128 \
--num_class 126 --lr 0.001 \
--num_epochs 1000 --num_neighbors 10 --wandb