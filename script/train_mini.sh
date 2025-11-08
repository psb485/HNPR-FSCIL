#!/bin/bash
run_name="Mini-camera_ready-reproduce"
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
    --dataset                   mini_imagenet
    --data_root                 ./data
    --batch_size                512
)
train_args=(
    --wandb
    # ------ train ------ #
    --n_epochs                  300
    --loss_type                 cosface
        --loss_s                30
        --loss_m                0.4
    --lr                        3e-02
    --momentum                  0.9
    --w_decay                   5e-03
    --min_scale                 0.1
    # ------ top-k options ------ #
    # --use_easy_neg
    # --use_random_topk
    # --use_static_topk
    #     --static_feat_model     $static_model
)
test_args=(
    # ------ test ------ #
    --test_pretrained
        --test_ckpt_file        ./log/mini_imagenet/$run_name/last.pth
    --min_scale                 0.6                  
)
python src/main.py "${args[@]}" "${train_args[@]}"      # train
python src/main.py "${args[@]}" "${test_args[@]}"       # test