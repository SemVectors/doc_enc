#!/usr/bin/env bash


python -m  doc_enc.training.run_training --config-path=$(pwd)/train/conf "$@"
