#!/usr/bin/env bash

if [ -z "$INIT_TOTAL_SENTS_CNT" ]; then
    INIT_TOTAL_SENTS_CNT=8000
fi
if [ -z "$TUNE_PARAM_NAME" ]; then
    param_name="batches.docs_batch_iterator_conf.batch_generator_conf.batch_total_sents_cnt"
fi

args=( "$@" )

total_sents_cnt=$INIT_TOTAL_SENTS_CNT
i=1

while true ; do
    cur_args=("${args[@]}"
              "$param_name=$total_sents_cnt")
    echo "begin ${i}th iteration; params:"
    echo "${cur_args[@]}"
    python -m  doc_enc.training.run_training --config-path=$(pwd)/train/conf "${cur_args[@]}"
    rcode=$?
    if [ $rcode -eq 0 ]; then
        break
    fi

    total_sents_cnt=$((total_sents_cnt-1000))
    if [[ $total_sents_cnt -le 0 ]]; then
       break
    fi
    i=$((i+1))

done

