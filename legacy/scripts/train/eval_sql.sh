export BASE="./checkpoints/models"
export PRETRAINED=
export MDL=
export PRECISION=

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

echo "the model is ${MDL}"

if [[ $MDL == *"sql"* ]]; then
   python sql_eval.py \
      --model_name_or_path ${PRETRAINED} \
      --peft_path ${BASE}/${MDL} \
      --precision ${PRECISION}
fi
