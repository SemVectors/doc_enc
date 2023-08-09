#!/usr/bin/env bash

print_usage() {
    echo "[-i include_models] - list of models to evaluation (by default all models) "
    echo "[-e exclude_models] - list of models to exclude from evaluation "
    echo "[-m models_dir] - download models to this directory (default: $(pwd)/models) "
    echo "[-d data_dir] - download datasets to this directory (default: $(pwd)/data)"
}

eval_model(){

    local model_id="$1"

    local model_name="models.$model_id.pt"
    if [ ! -e "$MODELS_DIR/$model_name" ]; then
        echo "downloading $model_id"
        wget http://dn11.isa.ru:8080/doc-enc-data/"$model_name" -O "$MODELS_DIR/$model_name"
    fi

    local batch_size=${BATCH_SIZE_PER_MODEL[default]}
    [[ -n ${BATCH_SIZE_PER_MODEL["$model_id"]} ]] && batch_size=${BATCH_SIZE_PER_MODEL["$model_id"]}

    docker run  --gpus=1  --rm  \
        -v "$DATA_DIR":/data/ -v "$MODELS_DIR":/models -v "$(pwd)"/eval:/eval/ \
        -e CURL_CA_BUNDLE="" \
        semvectors/doc_enc_train:0.1.0 \
        run_eval \
        hydra.job_logging.handlers.console.stream=ext://sys.stderr \
        doc_matching.ds_base_dir=/data doc_retrieval.ds_base_dir=/data \
        bench.doc_ds_base_dir=/data \
        print_as_csv=true \
        model_id="$model_id" \
        eval_sent_retrieval=false \
        bench_doc_encoding=true \
        doc_encoder.model_path="/models/$model_name" \
        doc_encoder.async_batch_gen=4 \
        doc_encoder.max_sents=6000 \
        doc_encoder.max_tokens="$batch_size" | tee -a paper1_results.txt


}

dl_data(){

    if [ ! -e "$DATA_DIR"/SimEnWiki ]; then
        (cd "$DATA_DIR" && wget -q -O -  'http://dn11.isa.ru:8080/doc-enc-data/datasets.simenwiki.v4.segmented.tar.gz' | tar zxf -)
    fi
    if [ ! -e "$DATA_DIR"/SimRuWiki ]; then
        (cd "$DATA_DIR" && wget -q -O -  'http://dn11.isa.ru:8080/doc-enc-data/datasets.simruwiki.v4.segmented.tar.gz' | tar zxf -)
    fi
    if [ ! -e "$DATA_DIR"/ParalWiki ]; then
        (cd "$DATA_DIR" && wget -q -O -  'http://dn11.isa.ru:8080/doc-enc-data/datasets.paralwiki.v2.segmented.tar.gz' | tar zxf -)
    fi
    if [ ! -e "$DATA_DIR"/GWikiMatch ]; then
        (cd "$DATA_DIR" && wget -q -O -  'http://dn11.isa.ru:8080/doc-enc-data/datasets.gwikimatch.v1.segmented.tar.gz' | tar zxf -)
    fi
    if [ ! -e "$DATA_DIR"/essay1 ]; then
        (cd "$DATA_DIR" && wget -q -O -  'http://dn11.isa.ru:8080/doc-enc-data/datasets.essay.ru-en.v2.segmented.tar.gz' | tar zxf -)
    fi
}


set -e

MODELS_DIR=""
DATA_DIR=""
EXCLUDE_MODELS=()
EVAL_MODELS=()

while [ $# -gt 0 ] ; do
    case $1 in
        -i       ) EVAL_MODELS=($2)      ; shift 2 ;;
        -e       ) EXCLUDE_MODELS=($2)   ; shift 2 ;;
        -m       ) MODELS_DIR=$2         ; shift 2 ;;
        -d       ) DATA_DIR=$2           ; shift 2 ;;
        -h       ) print_usage           ; exit 0  ;;
        *       ) echo "Unknown option!" ; exit 2  ;;
    esac
done

if ! command -v wget &> /dev/null
then
    echo "wget is required; install it before running the script!"
    exit
fi

if [ ${#EVAL_MODELS[@]} -eq 0 ]; then
    EVAL_MODELS=(distilbert_a1-4 longformer_a1-1 xlm_roberta_a1-1
                 distilroberta-a1-1 distilroberta-fragments-a1-1
                 distiluse-a1-2 distiluse-fragments-a1-2
                 doc_trans_full1-9-4)
fi

if [ -z "$MODELS_DIR" ]; then
    MODELS_DIR="$(pwd)/models"
fi
mkdir -p "$MODELS_DIR"

if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$(pwd)/data"
fi
mkdir -p "$DATA_DIR"

declare -A BATCH_SIZE_PER_MODEL
BATCH_SIZE_PER_MODEL[default]=128_000
BATCH_SIZE_PER_MODEL[longformer_a1-1]=8_000
echo "Download models to $MODELS_DIR"
echo "Download data to $DATA_DIR"
dl_data

for model_id in "${EVAL_MODELS[@]}"; do
    [[ " ${EXCLUDE_MODELS[*]} " =~ " $model_id " ]] && continue
    eval_model "$model_id"

done

echo "Results were written to stdout and to paper1_results.txt file"
