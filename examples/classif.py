#!/usr/bin/env python3

import sys

from doc_enc.doc_encoder import DocEncoderConf
from doc_enc.classif_doc import ClassifDoc


if len(sys.argv) < 2:
    raise RuntimeError("Pass model path as the first positional argument")

conf = DocEncoderConf(model_path=sys.argv[1], use_gpu=0)
doc_encoder = ClassifDoc(conf)

docs = [
    ['simple sentence', 'second sentence'],
    ['Sentences are already segmented'],
    ['Spam spam spam', 'Span Spam Spam'],
]

result = doc_encoder.predict_docs(docs)
print(result)
