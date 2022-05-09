#!/usr/bin/env bash

python -m  doc_enc.eval.run_eval --config-path=$(pwd)/eval/conf "$@"
