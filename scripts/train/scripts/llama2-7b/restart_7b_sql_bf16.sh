bash scripts/restarter_llama2.sh \
    MODEL_SIZE=7b \
    MODEL_PRECISION=bf16 \
    DATASET=sql \
    LR=0.0002 \
    LORA_LR=0.0007 \
    SPA_DENSITY=0.0045 \
    LORA_R=4 \
    NUM_EPOCHS=1
    