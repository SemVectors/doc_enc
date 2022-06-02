#!/usr/bin/env bash

inp_dir="$1"

for d in "$inp_dir"/* ; do
    over_file="$d"/.hydra/overrides.yaml
    if [ ! -e "$over_file" ]; then
        echo "$over_file does not exist"
        continue
    fi

    model_path="$d"/model.pt
    if [ ! -e "$model_path" ]; then
        echo "$model_path does not exits"
        continue
    fi

    # - +experiments=micro_lstm_15
    model_id=$(grep -Po "(?<=- \+experiments=).*" "$over_file")

    time ./eval/run_eval.sh +personal/dvzubarev=tsa06-quick print_as_csv=true \
        model_id="$model_id" doc_encoder.model_path="$model_path"

    mv "$d" "$inp_dir"/"$model_id"
done
