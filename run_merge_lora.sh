# openvla
python scripts/merge_lora_weights_and_save.py \
    --base_checkpoint /models/zhangguoxi/openvla-7b \
    --lora_finetuned_checkpoint_dir /root/sylvia/OpenVLA/openvla/outputs/openvla-7b+libero_object_no_noops+b12+lr-0.0005+lora-r32+dropout-0.0--image_aug--step_500000--20260125_175958 \
    --vlm_path /models/zhangguoxi/openvla-7b \
    --use_minivla False \
    --vla_arch openvla \

