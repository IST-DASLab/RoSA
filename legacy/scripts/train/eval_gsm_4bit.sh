export BASE="./checkpoints/models"
export PRETRAINED=
export MDL=

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

echo "the model is ${MDL}"

if [[ $MDL == *"gsm"* ]]; then
   python lm_eval_main.py \
      --model hf-causal-experimental \
      --model_args pretrained=${PRETRAINED},use_accelerate=True,dtype=bfloat16,peft=${BASE}/${MDL},load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type=nf4 \
      --tasks gsm8k \
      --num_fewshot 0 \
      --batch_size auto \
      --write_out \
      --output_base_path evals/${MDL} \
      --no_cache \
      --device cuda
fi