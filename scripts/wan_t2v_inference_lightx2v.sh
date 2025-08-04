# this is the setting for 1x length T2V inference
dense_layers=1
dense_timesteps=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

prompt=$(cat examples/prompt.txt)

python wan_t2v_inference.py \
    --prompt "$prompt" \
    --height 768 \
    --width 1280 \
    --num_frames 69 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.2 \
    --pattern "radial" \
    --num_inference_steps 4 \
    --output_file "radial_lightx2v.mp4" \
    --guidance_scale 1.0 \
    --flow_shift 8.0 \
    --lora_checkpoint_dir "Kijai/WanVideo_comfy" \
    --lora_checkpoint_name "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors" \
    --prompt "Reflections in the window of a train traveling through the Tokyo suburbs." \
    --use_model_offload \
    --use_sage_attention \

