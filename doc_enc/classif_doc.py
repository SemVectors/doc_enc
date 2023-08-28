#!/usr/bin/env python3

from pathlib import Path

import torch

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf, file_path_fetcher


class ClassifDoc(DocEncoder):
    def __init__(self, conf: DocEncoderConf) -> None:
        super().__init__(conf)

        d = self.enc_module().device
        state_dict = self.enc_module()._state_dict
        if 'fine_tune_cfg' not in state_dict:
            raise RuntimeError("Model should be fine-tuned for the classification task!")
        classif_nlabels = state_dict['fine_tune_cfg'].nlabels
        self.classif_layer = torch.nn.Linear(self.enc_module().doc_embs_dim(), classif_nlabels)
        self.classif_layer.load_state_dict(state_dict['classif_layer'])
        self.classif_layer = self.classif_layer.to(device=d)
        self.classif_layer.requires_grad_(False)

        self.labels_index = state_dict['labels_index']
        self.labels_mapping = state_dict['labels_mapping']

    def _map_predictions_to_labels(self, predictions):
        labels = []
        for idx in predictions:
            if idx > len(self.labels_index):
                raise RuntimeError("Prediction's index is out of bound")
            labels.append(self.labels_index[idx])
        return labels

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

        return self._map_predictions_to_labels(torch.hstack(predictions).tolist())

    def predict_docs_from_path_list(self, path_list: list[str] | list[Path]):
        predictions = []
        pred_idxs = []
        batch_iter = self._enc_module.create_batch_iterator()
        batch_iter.start_workers_for_item_list(path_list, fetcher=file_path_fetcher)

        try:
            for docs, doc_lengths, idxs in batch_iter.batches():
                doc_embs = self._encode_docs(docs, doc_lengths)
                output = self.classif_layer(doc_embs)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.to(device='cpu'))
                pred_idxs.extend(idxs)

            stacked = torch.hstack(predictions)
            assert stacked.shape[0] == len(
                path_list
            ), f"Missaligned data: {stacked.shape[0]} != {len(path_list)}"

            reordered_predictions = self._reorder_collected_arrays(stacked, pred_idxs).tolist()

            return self._map_predictions_to_labels(reordered_predictions)

        finally:
            batch_iter.destroy()

    def predict_docs_from_dir(self, path: Path):
        paths = list(path.iterdir())
        paths.sort()
        return paths, self.predict_docs_from_path_list(paths)

    def predict_docs_stream(self, doc_id_generator, fetcher, batch_size: int = 10):
        batch_iter = self._enc_module.create_batch_iterator()
        batch_iter.start_workers_for_stream(
            doc_id_generator, fetcher=fetcher, batch_size=batch_size
        )
        try:
            for docs, doc_lengths, ids in batch_iter.batches():
                doc_embs = self._encode_docs(docs, doc_lengths)
                output = self.classif_layer(doc_embs)
                _, predicted = torch.max(output, 1)
                labels = self._map_predictions_to_labels(predicted.to(device='cpu').tolist())
                yield ids, labels
        finally:
            batch_iter.destroy()
