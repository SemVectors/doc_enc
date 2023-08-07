#!/usr/bin/env python3

from doc_enc.text_processor import TextProcessorConf
from doc_enc.tokenizer import TokenizerType, TokenizerConf
from doc_enc.doc_encoder import DocEncoderConf, BatchIterator


def _create_tp_conf():
    tp_conf = TextProcessorConf(
        TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED),
        min_sent_len=1,
        num_alpha_max_ratio=0.0,
    )
    return tp_conf


def _test_fetcher(items):
    yield from enumerate(items)


def test_batch_iter_1():
    batch_iter = BatchIterator(
        generator_args=(_create_tp_conf(), DocEncoderConf(model_path=''), {}), async_generators=2
    )

    items = [
        ['111 112', '121', '131'],
        ['211', '221 222'],
        ['331 332', '321', '331'],
        ['411', '421 422', '431 432 433 434'],
    ]
    batch_iter.start_workers_for_item_list(items, fetcher=_test_fetcher)

    batches = list(batch_iter.batches())
    assert len(batches) == 2
    batches.sort(key=lambda t: t[-1][0])
    sents_tokens1, _, idxs1 = batches[0]
    assert len(sents_tokens1) == 2
    assert sents_tokens1[0] == [[111, 112], [121], [131]]
    assert sents_tokens1[1] == [[211], [221, 222]]
    assert idxs1 == [0, 1]

    sents_tokens2, _, idxs2 = batches[1]
    assert sents_tokens2[0] == [[331, 332], [321], [331]]
    assert idxs2 == [2, 3]


def _lookup_fetcher(ids):
    items = {
        '1': ['111 112', '121', '131'],
        '2': ['211', '221 222'],
        '30': ['331 332', '321', '331'],
        '40': ['411', '421 422', '431 432 433 434'],
    }
    for idx, i in enumerate(ids):
        yield idx, items[i]


def test_batch_iter_2():
    batch_iter = BatchIterator(
        generator_args=(_create_tp_conf(), DocEncoderConf(model_path=''), {}), async_generators=2
    )

    item_ids = ['1', '2', '30', '40']
    batch_iter.start_workers_for_stream(item_ids, fetcher=_lookup_fetcher, batch_size=2)

    batches = list(batch_iter.batches())
    batches.sort(key=lambda t: t[-1][0])
    assert len(batches) == 2
    sents_tokens1, _, ids1 = batches[0]
    assert len(sents_tokens1) == 2
    assert sents_tokens1[0] == [[111, 112], [121], [131]]
    assert sents_tokens1[1] == [[211], [221, 222]]
    assert ids1 == ['1', '2']

    sents_tokens2, _, ids2 = batches[1]
    assert sents_tokens2[0] == [[331, 332], [321], [331]]
    assert ids2 == ['30', '40']
