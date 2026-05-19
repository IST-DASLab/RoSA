
# hyper-parameters with default values
export DATASET=gsm # gsm, sql and viggo are supported
export MODEL_SIZE=7b # the size of the llama model
export MODEL_PRECISION=bf16 # bf16 or 4bit
export BASE_SAVE_PATH=./checkpoints # where to store the checkpoints and generated masks
export NUM_EPOCHS=1
export WARMUP=20 # the learning rate warmup
export BS=32
export PER_DEVICE_BS=1
export LORA_ALPHA=16
export SCHEDULE=wl64 # the RoSA schedule
export SPA_NUM_GRADS=1 # number of gradients used for mask generation
export SPA_GRAD_ACC_MODE=mean_squared # 'mean' or 'mean_squared': how to accumulate gradients
export SEED=42
export LR=0.0002 # learning rate
export LORA_LR=0.0007 # a separate learning rate for the low-rank adapters

# hyper-parameters without default values
export SPA_DENSITY= # the sparse adapters' density
export LORA_R= # the low-rank adapters' rank

# wandb logging entity and project (optional)
export WANDB_ENTITY=
export WANDB_PROJECT=


# take all the input arguments and put them in environment variables
# this could override the hyper-parameters defined above
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# some post-processing on the inputs
export PRETRAINED=meta-llama/Llama-2-${MODEL_SIZE}-hf
export MAX_DURATION=${NUM_EPOCHS}ep
export CONFIG=yamls/${MODEL_SIZE}_rosa_${DATASET}.yaml
export RUN_NAME=llama2_${MODEL_SIZE}_${MODEL_PRECISION}-${DATASET}-rosa_${SCHEDULE}_d${SPA_DENSITY}_${SPA_NUM_GRADS}grads_${SPA_GRAD_ACC_MODE}_r${LORA_R}_loralr${LORA_LR}_alpha${LORA_ALPHA}-lr${LR}-epochs${NUM_EPOCHS}-wu${WARMUP}-seed${SEED}-$RANDOM

# create directories to save the masks and models
mkdir -p ${BASE_SAVE_PATH}/masks/
mkdir -p ${BASE_SAVE_PATH}/models/

if [[ "$SPA_DENSITY" != "0" ]]
then
    # sparse adaptation exists, so we need to generate masks

    if [[ $LORA_R == 0 ]]
    then
        export SCHEDULE=spa_only
    fi

    # no wandb logging for mask generation
    export WANDB_DISABLED=True

    # generate the masks and terminate
    composer train.py \
        ${CONFIG} \
        model_name_or_path=${PRETRAINED} \
        max_duration=${MAX_DURATION} \
        run_name=${RUN_NAME} \
        optimizer.lr=${LR} \
        global_train_batch_size=${BS} \
        device_train_microbatch_size=${PER_DEVICE_BS} \
        device_eval_batch_size=${PER_DEVICE_BS} \
        scheduler.t_warmup=${WARMUP}ba \
        model.weight_bias_dtype=${MODEL_PRECISION} \
        rosa.spa_d=${SPA_DENSITY} \
        rosa.spa_num_grads=${SPA_NUM_GRADS} \
        rosa.grad_acc_mode=${SPA_GRAD_ACC_MODE} \
        rosa.lora_r=${LORA_R} \
        rosa.lora_alpha=${LORA_ALPHA} \
        rosa.lora_lr=${LORA_LR} \
        rosa.schedule=${SCHEDULE} \
        global_seed=${SEED} \
        seed=${SEED} \
        hf_save_path=${BASE_SAVE_PATH}/models/ \
        rosa.mask_save_path=${BASE_SAVE_PATH}/masks/${RUN_NAME} \
        rosa.terminate_after_mask_generation=true
fi

# now we have the masks ready, so let's restart
export MASK_LOAD_PATH=${BASE_SAVE_PATH}/masks/${RUN_NAME}

# determine the correct RoSA schedule
if [[ "$SPA_DENSITY" != "0" && $LORA_R -ne 0 ]]
then
    export SCHEDULE=default
elif [[ $LORA_R -ne 0 ]]
then
    export SCHEDULE=lora_only
    export MASK_LOAD_PATH=
else
    export SCHEDULE=spa_only
fi

# re-enable wandb logging
export WANDB_DISABLED=False

# start the training with both sparse and low-rank adapters active from the outset
composer train.py \
    ${CONFIG} \
    model_name_or_path=${PRETRAINED} \
    max_duration=${MAX_DURATION} \
    run_name=${RUN_NAME} \
    optimizer.lr=${LR} \
    global_train_batch_size=${BS} \
    device_train_microbatch_size=${PER_DEVICE_BS} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    scheduler.t_warmup=${WARMUP}ba \
    model.weight_bias_dtype=${MODEL_PRECISION} \
    rosa.spa_d=${SPA_DENSITY} \
    rosa.spa_num_grads=${SPA_NUM_GRADS} \
    rosa.grad_acc_mode=${SPA_GRAD_ACC_MODE} \
    rosa.lora_r=${LORA_R} \
    rosa.lora_alpha=${LORA_ALPHA} \
    rosa.lora_lr=${LORA_LR} \
    rosa.schedule=${SCHEDULE} \
    global_seed=${SEED} \
    seed=${SEED} \
    hf_save_path=${BASE_SAVE_PATH}/models/ \
    rosa.mask_load_path=${MASK_LOAD_PATH}

# evaluate
if [ "$DATASET" = "gsm" ]; then
  bash eval_gsm_${MODEL_PRECISION}.sh PRETRAINED=${PRETRAINED} MDL=${RUN_NAME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
else
  bash eval_${DATASET}.sh PRETRAINED=${PRETRAINED} MDL=${RUN_NAME} PRECISION=${MODEL_PRECISION} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
fi
