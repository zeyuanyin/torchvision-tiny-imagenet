torchrun --nproc_per_node=1 classification/train.py \
    --model resnet18 \
    --batch-size 256 \
    --epochs 100 \
    --lr 0.5 \
    --lr-scheduler cosineannealinglr \
    --data-path /home/zeyuan.yin/imagenet \
    --output-dir ./save/tiny/resnet18 \
    --dataset tiny-imagenet