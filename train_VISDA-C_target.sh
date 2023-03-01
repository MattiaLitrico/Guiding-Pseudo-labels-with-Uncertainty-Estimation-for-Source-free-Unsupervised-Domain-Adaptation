DATA_ROOT=${1:-'data'}

python train_target.py --data_dir ${DATA_ROOT} --dataset visdac/target \
--source visdac_source --run_name visdac_source2target \
--num_class 12 --batch_size 64 --lr 0.01 \
--num_epochs 100 --num_neighbors 10 --wandb