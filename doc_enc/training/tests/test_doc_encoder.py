#!/usr/bin/env python3

from doc_enc.encoders.enc_in import EncoderInputType
from doc_enc.text_processor import TextProcessorConf
from doc_enc.tokenizer import TokenizerType, TokenizerConf
from doc_enc.doc_encoder import (
    DocEncoderConf,
    BatchAsyncGenerator,
    # SentsBatchAsyncGenerator,
    create_split_of_text_gens,
)


def _create_tp_conf():
    tp_conf = TextProcessorConf(
        TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED),
        min_sent_len=1,
        num_alpha_max_ratio=0.0,
    )
    return tp_conf


class _Fetcher:
    def __init__(self, texts, offs):
        self.texts = texts
        self.offs = offs

    def __call__(self):
        for idx, text in enumerate(self.texts):
            yield self.offs + idx, text


def test_batch_async_gen_1():
    batch_iter = BatchAsyncGenerator(
        EncoderInputType.JAGGED,
        DocEncoderConf(model_path=''),
        other_generator_args=(_create_tp_conf(), {}),
        async_generators=2,
    )

    items = [
        ['111 112', '121', '131'],
        ['211', '221 222'],
        ['331 332', '321', '331'],
        ['411', '421 422', '431 432 433 434'],
    ]
    gens = create_split_of_text_gens(items, 2, _Fetcher)
    try:
        batch_iter.start_workers()
        for batch in batch_iter.batches(gens):
            seb, batch_ids, text_repr = batch
            jgt = seb.get_jagged()

            if batch_ids == [0, 1]:
                assert jgt.data.tolist() == [111, 112, 121, 131, 211, 221, 222]
                assert jgt.lengths.tolist() == [2, 1, 1, 1, 2]
                assert text_repr.second_level_lengths is not None
                assert text_repr.second_level_lengths.tolist() == [3, 2]
                assert text_repr.third_level_lengths is not None
                assert text_repr.third_level_lengths.tolist() == [1, 1]
            elif batch_ids == [2, 3]:
                assert jgt.data.tolist() == [331, 332, 321, 331, 411, 421, 422, 431, 432, 433, 434]
                assert jgt.lengths.tolist() == [2, 1, 1, 1, 2, 4]
                assert text_repr.second_level_lengths is not None
                assert text_repr.second_level_lengths.tolist() == [3, 3]
                assert text_repr.third_level_lengths is not None
                assert text_repr.third_level_lengths.tolist() == [1, 1]
            else:
                raise RuntimeError(f"Unexpected result: {batch_ids}")
    finally:
        batch_iter.destroy()


# def _lookup_fetcher(ids):
#     items = {
#         '1': ['111 112', '121', '131'],
#         '2': ['211', '221 222'],
#         '30': ['331 332', '321', '331'],
#         '40': ['411', '421 422', '431 432 433 434'],
#     }
#     for idx, i in enumerate(ids):
#         yield idx, items[i]


# def test_batch_iter_2():
#     batch_iter = BatchIterator(
#         generator_args=(_create_tp_conf(), DocEncoderConf(model_path=''), {}), async_generators=2
#     )

#     item_ids = ['1', '2', '30', '40']
#     batch_iter.start_workers_for_stream(item_ids, fetcher=_lookup_fetcher, batch_size=2)

#     batches = list(batch_iter.batches())
#     batches.sort(key=lambda t: t[-1][0])
#     assert len(batches) == 2
#     sents_tokens1, _, ids1 = batches[0]
#     assert len(sents_tokens1) == 2
#     assert sents_tokens1[0] == [[111, 112], [121], [131]]
#     assert sents_tokens1[1] == [[211], [221, 222]]
#     assert ids1 == ['1', '2']

#     sents_tokens2, _, ids2 = batches[1]
#     assert sents_tokens2[0] == [[331, 332], [321], [331]]
#     assert ids2 == ['30', '40']


# * SentsBatchAsyncGenerator


class SentsGen:
    def __init__(self, rank: int):
        if rank == 0:
            self.sents = [("id1", "111 112 120"), ("id2", "121")]
        else:
            self.sents = [("id10", "1"), ("id11", ""), ("id12", "2 3 4"), ("id13", "4 5 6 7")]

    def __call__(self):
        for s in self.sents:
            yield s


def test_sents_batch_async_gen_packed_1():
    batch_iter = BatchAsyncGenerator(
        EncoderInputType.PACKED,
        DocEncoderConf(model_path=''),
        other_generator_args=(
            _create_tp_conf(),
            {},
        ),
        async_generators=2,
        shared_tens_slots_cnt=1,
    )

    try:
        batch_iter.start_workers()

        for batch in batch_iter.batches([SentsGen(0), SentsGen(1)], input_are_sents=True):
            seb, batch_ids, *_ = batch
            pd = seb.get_packed_seq()

            if batch_ids == ['id1', 'id2']:
                assert seb.batch_size == 2
                assert seb.max_len == 3

                assert pd.data.shape == (4,)
                assert pd.data.tolist() == [111, 121, 112, 120]
                assert pd.batch_sizes.tolist() == [2, 1, 1]
                assert pd.sorted_indices is not None
                assert pd.sorted_indices.tolist() == [0, 1]
            elif batch_ids == ['id10', 'id11', 'id12', 'id13']:
                assert seb.batch_size == 4
                assert seb.max_len == 4

                assert pd.data.shape == (9,)
                # empty seq is replaced with "<pad>"
                assert pd.data.tolist()[:6] == [4, 2, 1, 0, 5, 3]
                assert pd.batch_sizes.tolist() == [4, 2, 2, 1]
                assert pd.sorted_indices is not None
                assert pd.sorted_indices.tolist() == [3, 2, 0, 1]
            else:
                raise RuntimeError("Unexpected result!")
    finally:
        batch_iter.destroy()


def test_sents_batch_async_gen_jagged_1():
    batch_iter = BatchAsyncGenerator(
        EncoderInputType.JAGGED,
        DocEncoderConf(model_path=''),
        other_generator_args=(
            _create_tp_conf(),
            {},
        ),
        async_generators=2,
    )
    try:
        batch_iter.start_workers()

        for batch in batch_iter.batches([SentsGen(0), SentsGen(1)], input_are_sents=True):
            seb, batch_ids, *_ = batch
            jagt = seb.get_jagged()
            jagt_w_pos = seb.get_jagged_w_pos_ids()

            if batch_ids == ['id1', 'id2']:
                assert seb.batch_size == 2
                assert seb.max_len == 3

                assert jagt.data.shape == (4,)
                assert jagt.data.tolist() == [111, 112, 120, 121]
                assert jagt.lengths.tolist() == [3, 1]

                assert jagt_w_pos.position_ids.tolist() == [0, 1, 2, 0]
            elif batch_ids == ['id10', 'id11', 'id12', 'id13']:
                assert seb.batch_size == 4
                assert seb.max_len == 4

                assert jagt.data.shape == (9,)
                assert jagt.data.tolist() == [1, 0, 2, 3, 4, 4, 5, 6, 7]
                assert jagt.lengths.tolist() == [1, 1, 3, 4]

                assert jagt_w_pos.position_ids.tolist() == [0, 0, 0, 1, 2, 0, 1, 2, 3]

            else:
                raise RuntimeError("Unexpected result!")
    finally:
        batch_iter.destroy()


def test_sents_batch_async_gen_padded_1():

    batch_iter = BatchAsyncGenerator(
        EncoderInputType.PADDED,
        DocEncoderConf(model_path=''),
        other_generator_args=(
            _create_tp_conf(),
            {},
        ),
        async_generators=2,
    )
    try:
        batch_iter.start_workers()

        for batch in batch_iter.batches([SentsGen(0), SentsGen(1)], input_are_sents=True):
            seb, batch_ids, *_ = batch
            padded = seb.get_padded()

            if batch_ids == ['id1', 'id2']:
                assert seb.batch_size == 2
                assert seb.max_len == 3

                assert padded.data.shape == (2, 3)
                assert padded.data.tolist() == [[111, 112, 120], [121, 0, 0]]
                assert padded.lengths.tolist() == [3, 1]

            elif batch_ids == ['id10', 'id11', 'id12', 'id13']:
                assert seb.batch_size == 4
                assert seb.max_len == 4

                assert padded.data.shape == (4, 4)
                assert padded.data.tolist() == [
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [2, 3, 4, 0],
                    [4, 5, 6, 7],
                ]
                assert padded.lengths.tolist() == [1, 1, 3, 4]

            else:
                raise RuntimeError("Unexpected result!")
    finally:
        batch_iter.destroy()
