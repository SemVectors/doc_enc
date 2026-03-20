#!/usr/bin/env python

from pathlib import Path
import collections.abc
from typing import Any

from doc_enc.finetune_classif import load_clsf_module
from doc_enc.doc_encoder import file_path_fetcher, DocEncoderConf

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

    def _predict(self, docs, doc_lengths):
        return self._cls_module.predictions_with_weights(docs, doc_lengths)

    def clsf_docs_from_path_list(self, path_list: list[str] | list[Path]) -> ClsfResultT:
        results = [[]] * len(path_list)
        batch_iter = self._enc_module.create_batch_async_generator()

        batch_iter.start_workers_for_item_list(path_list, fetcher=file_path_fetcher)
        try:
            for docs, doc_lengths, idxs in batch_iter.batches():

                preds = self._predict(docs, doc_lengths)
                for i, orig_i in enumerate(idxs):
                    results[orig_i] = preds[i]
            assert len(results) == len(
                path_list
            ), f"Missaligned data with paths: {len(results)} != {len(path_list)}"

            return results
        finally:
            batch_iter.destroy()

    def clsf_docs_from_dir(self, path: Path) -> tuple[list[Path], ClsfResultT]:
        paths = list(path.iterdir())
        paths.sort()
        return paths, self.clsf_docs_from_path_list(paths)

    def clsf_docs(self, docs: list[list[str] | str]) -> ClsfResultT:
        def dummy_fetcher(items):
            yield from enumerate(items)

        results = []

        batch_generator = self._enc_module.create_batch_generator()
        for batch, doc_lengths, _ in batch_generator.batches(docs, fetcher=dummy_fetcher):
            preds = self._predict(batch, doc_lengths)
            results.extend(preds)
        assert len(results) == len(docs)

        return results

    def clsf_docs_stream(
        self, doc_id_generator, fetcher, batch_size: int = 10
    ) -> collections.abc.Iterable[tuple[list[Any], ClsfResultT]]:
        batch_iter = self._enc_module.create_batch_async_generator()
        batch_iter.start_workers_for_stream(
            doc_id_generator, fetcher=fetcher, batch_size=batch_size
        )
        try:
            for docs, doc_lengths, ids in batch_iter.batches():

                preds = self._predict(docs, doc_lengths)
                yield ids, preds
        finally:
            batch_iter.destroy()
