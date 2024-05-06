export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./finetune_lora_pokemon"
export DATASET_NAME="diffusers/pokemon-llava-captions"
export TRAIN_SCRIPT="./Cos_train.py"

accelerate launch --mixed_precision="fp16" $TRAIN_SCRIPT \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --seed=1337 \
  --llm_loss_weight 0.1 \
  --loss="new" 