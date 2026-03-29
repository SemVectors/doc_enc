#!/usr/bin/env python

from pathlib import Path
import collections.abc
from typing import Any
import typing

from doc_enc.encoders.enc_in import EncoderInData
from doc_enc.finetune_classif import load_clsf_module
from doc_enc.doc_encoder import DocEncoderConf, TextGenFuncT, create_text_gens_from_ids_list

ClsfResultT = list[list[tuple[str, float]]]


class DocClassifier:
    def __init__(
        self, conf: DocEncoderConf, topk: int | None = None, threshold: float | None = None
    ) -> None:
        self._cls_module = load_clsf_module(conf, topk=topk, threshold=threshold, eval_mode=True)
        self._enc_module = self._cls_module.encoder

        state_dict = self._enc_module._state_dict

        self.labels_index = state_dict['labels_index']
        self.labels_mapping = state_dict['labels_mapping']

    def _predict(self, input_data: EncoderInData):
        return self._cls_module.predictions_with_weights(input_data)

    def clsf_docs_from_path_list(self, path_list: list[str] | list[Path]) -> ClsfResultT:
        results = [[]] * len(path_list)
        batch_gen = self._enc_module.create_batch_async_generator()

        gens = create_text_gens_from_ids_list(path_list, 10 * batch_gen.nproc())
        try:
            for input_data in batch_gen.batches(gens):

                preds = self._predict(input_data)
                for i, orig_i in enumerate(input_data.text_ids):
                    orig_i = typing.cast(int, orig_i)
                    results[orig_i] = preds[i]
            assert len(results) == len(
                path_list
            ), f"Missaligned data with paths: {len(results)} != {len(path_list)}"

            return results
        finally:
            batch_gen.destroy()

    def clsf_docs_from_dir(self, path: Path) -> tuple[list[Path], ClsfResultT]:
        paths = list(path.iterdir())
        paths.sort()
        return paths, self.clsf_docs_from_path_list(paths)

    def clsf_docs(self, docs: list[list[str] | str]) -> ClsfResultT:
        def dummy_fetcher():
            yield from enumerate(docs)

        results = []

        batch_generator = self._enc_module.create_batch_generator()
        for input_data in batch_generator.batches(dummy_fetcher):
            preds = self._predict(input_data)
            results.extend(preds)
        assert len(results) == len(docs)

        return results

    def clsf_docs_from_generators(
        self, generator_funcs: list[TextGenFuncT]
    ) -> collections.abc.Iterable[tuple[list[Any], ClsfResultT]]:
        batch_iter = self._enc_module.create_batch_async_generator()
        try:
            for input_data in batch_iter.batches(generator_funcs):
                preds = self._predict(input_data)
                yield input_data.text_ids, preds
        finally:
            batch_iter.destroy()
