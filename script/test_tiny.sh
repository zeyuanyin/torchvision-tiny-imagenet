torchrun --nproc_per_node=1 classification/train.py \
    --model resnet18 \
    --batch-size 256 \
    --data-path /home/zeyuan.yin/imagenet \
    --dataset tiny-imagenet \
    --resume ./save/tiny/resnet18/checkpoint.pth \
    --test-only
