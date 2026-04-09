#!/usr/bin/env python3


from contextlib import contextmanager

import torch

from doc_enc.encoders.enc_in import EncoderInData
from doc_enc.shared_tensors import EncInputSharedTensors


def serialize_enc_in_data(in_data: EncoderInData, shared_tensors_holder: EncInputSharedTensors):
    # logging.error("serialize in tensors")
    slot_num, info = shared_tensors_holder.share_input_data(
        in_data.seq_encoder_input, in_data.texts_repr
    )

    # data will be serialized while putting into queue
    return (slot_num, info, in_data.text_ids)


def serialize_training_data(
    src_data: EncoderInData,
    tgt_data: EncoderInData,
    labels: torch.Tensor,
    shared_tensors_holder: EncInputSharedTensors,
):
    slot_num, info = shared_tensors_holder.share_training_data(
        src_data.seq_encoder_input,
        src_data.texts_repr,
        tgt_data.seq_encoder_input,
        tgt_data.texts_repr,
        labels,
    )

    return (slot_num, info, src_data.text_ids, tgt_data.text_ids)


@contextmanager
def deserialize_enc_in_data(ser_data: tuple, shared_tensors_holder: EncInputSharedTensors):
    slot_num, info, text_ids = ser_data
    try:
        seq_encoder_input, texts_repr = shared_tensors_holder.recreate_batch(slot_num, info)
        yield EncoderInData(
            seq_encoder_input=seq_encoder_input,
            texts_repr=texts_repr,
            text_ids=text_ids,
        )
    finally:
        shared_tensors_holder.release_slot(slot_num)


@contextmanager
def deserialize_training_data(ser_data: tuple, shared_tensors_holder: EncInputSharedTensors):
    slot_num, info, src_text_ids, tgt_text_ids = ser_data
    try:
        ((src_inp, src_tr), (tgt_inp, tgt_tr), labels) = (
            shared_tensors_holder.recreate_training_data(slot_num, info)
        )

        yield (
            EncoderInData(
                seq_encoder_input=src_inp,
                texts_repr=src_tr,
                text_ids=src_text_ids,
            ),
            EncoderInData(
                seq_encoder_input=tgt_inp,
                texts_repr=tgt_tr,
                text_ids=tgt_text_ids,
            ),
            labels,
        )
    finally:
        shared_tensors_holder.release_slot(slot_num)
