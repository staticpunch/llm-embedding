model_name_or_path="llama-7b/"
output_directory="output-llama-lora"
cache_directory="data"
train_file="medi-1000.jsonl"

lr=5e-5
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05
deepspeed_config_file=ds_zero2.json

CUDA_VISIBLE_DEVICES=2,5 torchrun --nnodes 1 --nproc_per_node 2 train_llm.py \
	--deepspeed ${deepspeed_config_file} \
	--model_name_or_path ${model_name_or_path} \
	--output_dir ${output_directory} \
	--cache_dir ${cache_directory} \
  --train_file ${train_file} \
  --max_source_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --bf16 \
  --load_in_kbits 4 \
  --lora_rank ${lora_rank} \
  --lora_alpha ${lora_alpha} \
  --trainable ${lora_trainable} \
  --lora_dropout ${lora_dropout} \
  --modules_to_save ${modules_to_save} \
  --logging_first_step True \
  --logging_strategy steps \
  --logging_steps 20 \
	--num_train_epochs 5 \
	--save_steps 400 \
	--cl_temperature 0.1 \
	--warmup_ratio 0.1 \
	--learning_rate ${lr} \
	--overwrite_output_dir \
  --report_to none \
  --full_finetuning False
