#!/usr/bin/env python

import functools
import logging
import multiprocessing as mp


import torch

from doc_enc.encoders.enc_in import (
    EncoderInputType,
    JaggedInputTensor,
    PaddedTensor,
    SeqEncoderBatchedInput,
    TextsRepr,
)


class TorchSharedTensorsHolder:
    def __init__(
        self,
        tensors_cnt: int,
        slots_cnt: int,
        buf_shape: tuple[int, ...] | list[tuple[int, ...]],
        dtype: torch.dtype | list[torch.dtype] = torch.float32,
    ):

        self._tensors_cnt = tensors_cnt
        self._slots_cnt = slots_cnt
        if isinstance(dtype, torch.dtype):
            self._dtypes = [dtype] * tensors_cnt
        elif isinstance(dtype, list):
            if len(dtype) != tensors_cnt:
                raise RuntimeError("When dtype is list it should contain tensors_cnt elements")
            if not all(isinstance(v, torch.dtype) for v in dtype):
                raise RuntimeError("dtype should contain only elements of torch.dtype")
            self._dtypes = dtype
        else:
            raise RuntimeError("Unknown type of dtype argument")

        if isinstance(buf_shape, tuple):
            self._buf_shapes = [buf_shape] * tensors_cnt
        elif isinstance(buf_shape, list):
            if len(buf_shape) != tensors_cnt:
                raise RuntimeError("When buf_shape is list it should contain tensors_cnt elements")
            if not all(isinstance(v, tuple) for v in buf_shape):
                raise RuntimeError("buf_shape should contain only elements of tuple")
            self._buf_shapes = buf_shape
        else:
            raise RuntimeError("Unknown type of buf_shape argument")

        self._max_elems = [
            functools.reduce(lambda c, i: i * c, shape, 1) for shape in self._buf_shapes
        ]

        self._shared_tensors_pool = []
        for _ in range(slots_cnt):
            tensors = []
            for shape, dtype in zip(self._buf_shapes, self._dtypes):
                t = torch.empty(shape, dtype=dtype)
                t.share_memory_()
                tensors.append(t)
            self._shared_tensors_pool.append(tensors)

        self._free_slots = mp.Array('l', [1] * slots_cnt, lock=False)
        self._cv = mp.Condition()

    def free_slots_cnt(self):
        with self._cv:
            return sum(self._free_slots)

    def get_tensors(self, slot: int, shapes: list[tuple[int, ...] | None]):
        if self._tensors_cnt != len(shapes):
            raise RuntimeError(
                f"Wrong tensors length: Got {len(shapes)}, expected {self._tensors_cnt}"
            )

        ts = self._shared_tensors_pool[slot]
        tensors = []
        for shared_tensor, dtype, shape in zip(ts, self._dtypes, shapes, strict=True):
            if shape is not None:
                # access to underluing buffer that is resided in shared memory
                buf = shared_tensor.numpy().data
                count = functools.reduce(lambda c, i: i * c, shape, 1)
                t = torch.frombuffer(buf, dtype=dtype, count=count).view(shape)
                tensors.append(t)
            else:
                tensors.append(None)

        return tuple(tensors)

    def _find_free_slot(self):
        while True:
            for i in range(self._slots_cnt):
                if self._free_slots[i] == 1:
                    self._free_slots[i] = 0
                    return i
            self._cv.wait()

    def release_slot(self, slot_num):
        with self._cv:
            self._free_slots[slot_num] = 1
            self._cv.notify()

    def put_tensors(self, tensors: tuple[torch.Tensor, ...]):
        if self._tensors_cnt != len(tensors):
            raise RuntimeError(
                f"Wrong tensors length: Got {len(tensors)}, expected {self._tensors_cnt}"
            )
        with self._cv:
            free_slot = self._find_free_slot()
        shared_tensors = self._shared_tensors_pool[free_slot]

        for shared_tensor, ten, max_elems, dtype in zip(
            shared_tensors, tensors, self._max_elems, self._dtypes, strict=True
        ):
            numel = ten.numel()
            if numel == 0:
                continue
            if numel > max_elems:
                raise RuntimeError(
                    f"Tensor has shape {ten.shape} - not compat with max elems {max_elems}"
                )
            if ten.dtype != dtype:
                raise RuntimeError(f"Mismatching dtype: got {ten.dtype}, expected {dtype}!")

            # access to underluing buffer that is resided in shared memory
            buf = shared_tensor.numpy().data
            # We need to create temp tensor that has the same number of elements that ten
            temp_tensor = torch.frombuffer(buf, dtype=ten.dtype, count=numel).view(ten.size())
            # copy tensor to shared memory
            temp_tensor[:] = ten
        return free_slot


class EncInputSharedTensors:
    def __init__(
        self, enc_input_type: EncoderInputType, max_tokens: int, max_seqs: int, slots_cnt: int
    ):
        self.enc_input_type = enc_input_type
        # TODO TEMP
        self.ser_full_text_info = False

        if enc_input_type == EncoderInputType.PADDED:
            shapes = [
                (max_seqs, max_tokens),
                (max_seqs,),  # lengths
            ]
            logging.error('create shared padded tensor with shapes %s', shapes)
            dtypes = [torch.int32, torch.int32]
        elif enc_input_type == EncoderInputType.PACKED:
            shapes = [
                (max_tokens,),  # tokens (PackedSequence.data)
                (max_tokens,),  # batch_sizes
                (max_seqs,),  # sorted_indices
                (max_seqs,),  # unsorted_indices
            ]
            logging.error('create shared tensor with shapes %s', shapes)
            dtypes = [torch.int32, torch.int64, torch.int64, torch.int64]
        elif enc_input_type == EncoderInputType.JAGGED:
            shapes = [
                (max_tokens,),  # tokens
                (max_seqs,),  # lengths
            ]
            logging.error('create shared jagged tensor with shapes %s', shapes)
            dtypes = [torch.int32, torch.int32]
        else:
            raise RuntimeError(f"Unsupported enc input type: {enc_input_type}")

        if not self.ser_full_text_info:
            shapes = shapes + [
                (max_seqs,),  # text repr 2nd level
                (max_seqs,),  # text repr 3rd level
            ]
            dtypes = dtypes + [torch.int32, torch.int32]

        self._shared_tensors_holder = TorchSharedTensorsHolder(
            len(shapes), slots_cnt, shapes, dtypes
        )

    def put_tensors(
        self, input_data: SeqEncoderBatchedInput, text_repr: TextsRepr
    ) -> tuple[int, tuple]:
        # logging.error("input tipy %s", self.enc_input_type)
        if self.enc_input_type == EncoderInputType.PACKED:
            pd = input_data.get_packed_seq()
            shapes = (
                tuple(pd.data.shape),
                tuple(pd.batch_sizes.shape),
                (tuple(pd.sorted_indices.shape) if pd.sorted_indices is not None else None),
                (tuple(pd.unsorted_indices.shape) if pd.unsorted_indices is not None else None),
            )

            tens = (
                pd.data,
                pd.batch_sizes,
                (
                    pd.sorted_indices
                    if pd.sorted_indices is not None
                    else torch.empty(0, dtype=torch.int64)
                ),
                (
                    pd.unsorted_indices
                    if pd.unsorted_indices is not None
                    else torch.empty(0, dtype=torch.int64)
                ),
            )

        elif self.enc_input_type in (EncoderInputType.JAGGED, EncoderInputType.PADDED):
            pd = (
                input_data.get_jagged()
                if self.enc_input_type == EncoderInputType.JAGGED
                else input_data.get_padded()
            )
            shapes = (
                tuple(pd.data.shape),
                tuple(pd.lengths.shape),
            )
            tens = (pd.data, pd.lengths)
            # logging.error(
            #     'PUT_TENSORS, type %s, bs=%s, max_len = %s, shapes %s',
            #     self.enc_input_type,
            #     input_data.batch_size,
            #     input_data.max_len,
            #     shapes,
            # )
        else:
            raise RuntimeError(f"Unsupported enc input type: {self.enc_input_type}")

        if not self.ser_full_text_info:
            shapes = shapes + (
                (
                    tuple(text_repr.second_level_lengths.shape)
                    if text_repr.second_level_lengths is not None
                    else None
                ),
                (
                    tuple(text_repr.third_level_lengths.shape)
                    if text_repr.third_level_lengths is not None
                    else None
                ),
            )
            tens = tens + (
                (
                    text_repr.second_level_lengths
                    if text_repr.second_level_lengths is not None
                    else torch.empty(0, dtype=torch.int32)
                ),
                (
                    text_repr.third_level_lengths
                    if text_repr.third_level_lengths is not None
                    else torch.empty(0, dtype=torch.int32)
                ),
            )

        return self._shared_tensors_holder.put_tensors(tens), shapes

    def recreate_batch(self, slot_num: int, shapes: list[tuple[int] | None]):
        tens = self._shared_tensors_holder.get_tensors(slot_num, shapes)
        inp_tens = tens[:-2]
        seb = SeqEncoderBatchedInput(self.enc_input_type)
        if self.enc_input_type == EncoderInputType.PACKED:
            pd = torch.nn.utils.rnn.PackedSequence(*inp_tens)
            seb.batch = pd
            seb.batch_size = int(pd.batch_sizes[0])
            seb.max_len = int(pd.batch_sizes.shape[0])
        elif self.enc_input_type in (EncoderInputType.JAGGED, EncoderInputType.PADDED):
            cls = (
                JaggedInputTensor
                if self.enc_input_type == EncoderInputType.JAGGED
                else PaddedTensor
            )
            pd = cls(*inp_tens)
            seb.batch = pd
            seb.batch_size = int(pd.lengths.shape[0])
            seb.max_len = int(pd.lengths.max())
        else:
            raise RuntimeError(f"Unsupported enc input type: {self.enc_input_type}")

        text_repr = TextsRepr([], [])
        text_repr.second_level_lengths = tens[-2]
        text_repr.third_level_lengths = tens[-1]
        return seb, text_repr

    def release_slot(self, slot_num):
        self._shared_tensors_holder.release_slot(slot_num)
