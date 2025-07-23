python dbf_ptq.py \
    --gpu 0 \
    --target_bit 1.5 \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --n_calib_data 256 \
    --n_epochs 4 \
    --lr 0.00003 \
    --save_dir /projects/p487-24-1/dbf_rep \
    --is_save_ckpt \
