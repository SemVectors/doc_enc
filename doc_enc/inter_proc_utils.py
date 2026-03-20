#!/usr/bin/env python3


from contextlib import contextmanager

from doc_enc.encoders.enc_in import EncoderInData
from doc_enc.shared_tensors import EncInputSharedTensors


def serialize_enc_in_data(in_data: EncoderInData, shared_tensors_holder: EncInputSharedTensors):
    # logging.error("serialize in tensors")
    slot_num, shapes = shared_tensors_holder.put_tensors(
        in_data.seq_encoder_input, in_data.texts_repr
    )

    # data will be serialized while putting into queue
    # TODO use marshal to speed up?
    # See udpipe_ext BaseFromRefsGenerator._finalyze
    return (slot_num, shapes, in_data.text_ids)


@contextmanager
def deserialize_enc_in_data(ser_data: tuple, shared_tensors_holder: EncInputSharedTensors):
    slot_num, shapes, sent_ids = ser_data
    try:
        seq_encoder_input, texts_repr = shared_tensors_holder.recreate_batch(slot_num, shapes)
        yield EncoderInData(
            seq_encoder_input=seq_encoder_input,
            texts_repr=texts_repr,
            text_ids=sent_ids,
        )
    finally:
        shared_tensors_holder.release_slot(slot_num)
