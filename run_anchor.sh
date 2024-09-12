python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/bonsai -m outputs/mip360/bonsai --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -r 2
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/kitchen -m outputs/mip360/kitchen --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -r 2
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/room -m outputs/mip360/room --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -r 2
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/garden -m outputs/mip360/garden --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -r 4
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/bicycle -m outputs/mip360/bicycle --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 4
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/counter -m outputs/mip360/counter --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 2
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/flowers -m outputs/mip360/flowers --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 4
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/treehill -m outputs/mip360/treehill --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 4
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/stump -m outputs/mip360/stump --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 4

python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/tandt_db/tandt/train -m outputs/tandt/train --eval --voxel_size 0.01 --update_init_factor 16 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/tandt_db/tandt/truck -m outputs/tandt/truck --eval --voxel_size 0.01 --update_init_factor 16 --iterations 30_000

python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/tandt_db/db/drjohnson -m outputs/db/drjohnson --eval --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 --use_c2f
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/tandt_db/db/playroom -m outputs/db/playroom --eval --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 --use_c2f
# ------------------------
module load stack/.2024-04-silent
module load gcc/8.5.0
module load python/3.9
module load cuda/11.8
export PYTHONPATH=
source /cluster/work/cvl/jiezcao/jiameng/Spec-Gaussian/env/bin/activate
# ZoomGS
python train_anchor_zoomgs.py -s ../ZoomGS/ZoomGS/zoomgs_dataset/01 -m ckpt/zoomgs/01 --eval --dataset_split 10 --voxel_size 0.001 --update_init_factor 16
python train_anchor_zoomgs.py -s ../3D-Gaussian/nerf_synthetic/hotdog -m ckpt/syntheyic/hotdog --eval --dataset_split 100 --voxel_size 0.001 --update_init_factor 16
