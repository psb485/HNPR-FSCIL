#!/bin/bash
run_name="CIFAR100-camera_ready-reprodce"
args=(
    --seed                      1
    --run_name                  $run_name
    # ------ model ------ #
    --network                   resnet18
    --proj_feat_dim             2048
    --num_proj_layers           2
    --penalty_k                 2
    --penalty_m                 0.05
    # ------ dataset ------ #
    --dataset                   cifar100
    --data_root                 ./data
)
train_args=(
    --wandb
    # ------ train ------ #
    --n_epochs                  100
    --loss_type                 cosface
        --loss_s                30
        --loss_m                0.3
    --lr                        6e-02
    --momentum                  0.9
    --w_decay                   5e-03
    --batch_size                512
    --min_scale                 0.6
    # ------ top-k options ------ #
    # --use_easy_neg
    # --use_random_topk
    # --use_static_topk
    #     --static_feat_model     $static_model
)
test_args=(
    # ------ test ------ #
    --test_pretrained
        --test_ckpt_file        ./log/cifar100/$run_name/last.pth                  
    --batch_size                256
    --min_scale                 0.8
)
python src/main.py "${args[@]}" "${train_args[@]}"      # train
python src/main.py "${args[@]}" "${test_args[@]}"       # test