bash scripts/restarter_llama2.sh \
    MODEL_SIZE=7b \
    MODEL_PRECISION=4bit \
    DATASET=viggo \
    LR=0.0003 \
    LORA_LR=0.0007 \
    SPA_DENSITY=0.009 \
    LORA_R=8 \
    NUM_EPOCHS=1
    