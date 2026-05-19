export BASE="./checkpoints/models"
export MDL=

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

echo "the model is ${MDL}"

if [[ -f "evals/${MDL}/gsm8k_write_out_info.json" ]]; then
    echo 'evaluation already exists, skipping...'
    exit 0
fi

# if there is no dir named "merged" and adapter_config.json exists, merge the adapter config files
if [[ ! -d "${BASE}/${MDL}/merged" && -f "${BASE}/${MDL}/adapter_config.json" ]]; then
   python merge_adapter_bf16.py ${BASE}/${MDL}
fi

if [[ $MDL == *"gsm"* ]]; then
   python lm_eval_main.py \
      --model hf-causal-experimental \
      --model_args pretrained=${BASE}/${MDL}/merged,use_accelerate=True,dtype=bfloat16 \
      --tasks gsm8k \
      --num_fewshot 0 \
      --batch_size auto \
      --write_out \
      --output_base_path evals/${MDL} \
      --no_cache \
      --device cuda
fi

rm -rf "${BASE}/${MDL}/merged"