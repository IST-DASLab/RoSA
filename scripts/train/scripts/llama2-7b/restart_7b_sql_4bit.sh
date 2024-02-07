# best 4bit sql results are achieved by low-rank adapters alone.
# that's why SPA_DENSITY is set to 0.

bash scripts/restarter_llama2.sh \
    MODEL_SIZE=7b \
    MODEL_PRECISION=4bit \
    DATASET=sql \
    LR=0.0002 \
    LORA_LR=0.0007 \
    SPA_DENSITY=0 \
    LORA_R=64 \
    NUM_EPOCHS=1
    