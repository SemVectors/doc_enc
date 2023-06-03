#!/usr/bin/env python3

import sys

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf

if len(sys.argv) < 2:
    raise RuntimeError("Pass model path as the first positional argument")

conf = DocEncoderConf(model_path=sys.argv[1], use_gpu=0)
doc_encoder = DocEncoder(conf)

docs = [['simple sentence', 'second sentence'], ['Sentences are already segmented']]

result = doc_encoder.encode_docs(docs)
print("shape", result.shape)
print(result)
