#!/usr/bin/env python3

import sys

import torch

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf


class ClassifDoc(DocEncoder):
    def __init__(self, conf: DocEncoderConf) -> None:
        super().__init__(conf)

        d = self.enc_module().device()
        state_dict = torch.load(conf.model_path, map_location=d)

        if 'fine_tune_cfg' not in state_dict:
            raise RuntimeError("Model should be fine-tuned for the classification task")
        classif_nlabels = state_dict['fine_tune_cfg'].nlabels
        self.classif_layer = torch.nn.Linear(self.enc_module().doc_embs_dim(), classif_nlabels)
        self.classif_layer.load_state_dict(state_dict['classif_layer'])
        self.classif_layer = self.classif_layer.to(device=d)
        self.classif_layer.requires_grad_(False)

    def predict_docs(self, docs):
        def dummy_fetcher(items):
            yield from enumerate(items)

        predictions = []

        batch_generator = self._enc_module.create_batch_generator()
        for batch, doc_fragments, _ in batch_generator.batches(docs, fetcher=dummy_fetcher):
            doc_embs = self._encode_docs(batch, doc_fragments)
            output = self.classif_layer(doc_embs)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.to(device='cpu'))

        stacked = torch.vstack(predictions)
        return stacked.numpy()


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
print("shape", result.shape)
print(result)
