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

        self._shared_tensors_pool: list[list[torch.Tensor]] = []
        for _ in range(slots_cnt):
            tensors: list[torch.Tensor] = []
            for shape, dtype in zip(self._buf_shapes, self._dtypes):
                count = functools.reduce(lambda c, i: i * c, shape, 1)
                t = torch.empty((count,), dtype=dtype)
                t.share_memory_()
                tensors.append(t)
            self._shared_tensors_pool.append(tensors)

        self._free_slots = mp.Array('l', [1] * slots_cnt, lock=False)
        self._cv = mp.Condition()

    def reset(self):
        self._cv = mp.Condition()
        for i in range(self._slots_cnt):
            self._free_slots[i] = 1

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
                # Shared tensor has shape that is >= the shape of actual tensor,
                # so we cant just use .view() on shared tensor. Create tensor
                # that has the same number of elements that actual tensor.
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

            shared_tensor[:numel] = ten.view(numel)
        return free_slot


class EncInputSharedTensors:
    def __init__(
        self,
        enc_input_type: EncoderInputType,
        max_tokens: int,
        max_seqs: int,
        slots_cnt: int,
        is_training: bool = False,
        max_seq_length: int | None = None,
    ):
        self.enc_input_type = enc_input_type
        # TODO TEMP
        self.ser_full_text_info = False
        self.is_training = is_training

        if enc_input_type == EncoderInputType.PADDED:
            seq_len = max_tokens
            if max_seq_length is not None:
                seq_len = min(max_seq_length, max_tokens)
            shapes = [
                (max_seqs, seq_len),
                (max_seqs,),  # lengths
            ]
            dtypes = [torch.int32, torch.int32]
        elif enc_input_type == EncoderInputType.PACKED:
            max_len = max_tokens
            if max_seq_length is not None:
                max_len = 2 * max_seq_length if is_training else max_seq_length

            shapes = [
                (max_tokens,),  # tokens (PackedSequence.data)
                (max_len,),  # batch_sizes
                (max_seqs,),  # sorted_indices
                (max_seqs,),  # unsorted_indices
            ]
            dtypes = [torch.int32, torch.int64, torch.int64, torch.int64]
        elif enc_input_type == EncoderInputType.JAGGED:
            shapes = [
                (max_tokens,),  # tokens
                (max_seqs,),  # lengths
            ]
            dtypes = [torch.int32, torch.int32]
        else:
            raise RuntimeError(f"Unsupported enc input type: {enc_input_type}")

        if not self.ser_full_text_info:
            shapes = shapes + [
                (max_seqs,),  # text repr 2nd level
                (max_seqs,),  # text repr 3rd level
            ]
            dtypes = dtypes + [torch.int32, torch.int32]

        if is_training:
            # labels
            shapes = shapes + [(max_seqs, max_seqs)]
            dtypes = dtypes + [torch.float32]

        self._shared_tensors_holder = TorchSharedTensorsHolder(
            len(shapes), slots_cnt, shapes, dtypes
        )

    def reset(self):
        self._shared_tensors_holder.reset()

    def share_input_data(
        self,
        input_data: SeqEncoderBatchedInput,
        text_repr: TextsRepr,
        labels: torch.Tensor | None = None,
    ) -> tuple[int, dict]:

        if labels is not None and not self.is_training:
            raise RuntimeError(
                "serialize_input_data: labels is not None, but EncInputSharedTensors created without is_training=True"
            )

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

        if labels is not None:
            shapes = shapes + (tuple(labels.shape),)
            tens = tens + (labels,)

        info = {
            'shapes': shapes,
            'text_repr_type': text_repr.text_repr_type,
        }
        if input_data.sorted_by_length:
            info['sorted_by_length'] = True
        return self._shared_tensors_holder.put_tensors(tens), info

    def _recreate_derivatives(self, seb: SeqEncoderBatchedInput, info: dict):
        if info.get('sorted_by_length'):
            seb.sorted_by_length = True
        if self.enc_input_type == EncoderInputType.PACKED:
            pd = seb.get_packed_seq()
            seb.batch_size = int(pd.batch_sizes[0])
            seb.max_len = int(pd.batch_sizes.shape[0])
        elif self.enc_input_type in (EncoderInputType.JAGGED, EncoderInputType.PADDED):
            assert isinstance(
                seb.batch, (PaddedTensor, JaggedInputTensor)
            ), "_recreate_derivatives, seb.batch has wrong type"
            seb.batch_size = int(seb.batch.lengths.shape[0])
            seb.max_len = int(seb.batch.lengths.max())
        else:
            raise RuntimeError(f"Unsupported enc input type: {self.enc_input_type}")

    def recreate_batch(self, slot_num: int, info: dict):
        shapes = info['shapes']
        tens = self._shared_tensors_holder.get_tensors(slot_num, shapes)
        inp_tens = tens[:-2]
        seb = SeqEncoderBatchedInput(self.enc_input_type)
        if self.enc_input_type == EncoderInputType.PACKED:
            pd = torch.nn.utils.rnn.PackedSequence(*inp_tens)
        elif self.enc_input_type in (EncoderInputType.JAGGED, EncoderInputType.PADDED):
            if self.enc_input_type == EncoderInputType.JAGGED:
                pd = JaggedInputTensor(*inp_tens)
            else:
                pd = PaddedTensor(*inp_tens, padding_mask=None)
        else:
            raise RuntimeError(f"Unsupported enc input type: {self.enc_input_type}")

        seb.batch = pd

        self._recreate_derivatives(seb, info)

        text_repr = TextsRepr(info['text_repr_type'], [], [])
        text_repr.second_level_lengths = tens[-2]
        text_repr.third_level_lengths = tens[-1]
        return seb, text_repr

    def share_training_data(
        self,
        src_input_data: SeqEncoderBatchedInput,
        src_text_repr: TextsRepr,
        tgt_input_data: SeqEncoderBatchedInput,
        tgt_text_repr: TextsRepr,
        labels: torch.Tensor,
    ):
        assert (
            self.is_training
        ), "EncInputSharedTensors::serialize_training_data: created without is_training=True"

        assert (
            src_input_data.enc_input_type == tgt_input_data.enc_input_type
        ), "src and tgt have different enc_input_type."

        combined = SeqEncoderBatchedInput(src_input_data.enc_input_type)
        combined.sorted_by_length = src_input_data.sorted_by_length
        if self.enc_input_type == EncoderInputType.PACKED:
            spd = src_input_data.get_packed_seq()
            tpd = tgt_input_data.get_packed_seq()

            assert (
                tpd.sorted_indices is not None
            ), "It is expected that sorted_indices in target data is always non-None"
            assert (
                tpd.unsorted_indices is not None
            ), "It is expected that unsorted_indices in target data is always non-None"

            combined.batch = torch.nn.utils.rnn.PackedSequence(
                torch.cat((spd.data, tpd.data)),
                torch.cat((spd.batch_sizes, tpd.batch_sizes)),
                (
                    torch.cat((spd.sorted_indices, tpd.sorted_indices))
                    if spd.sorted_indices is not None
                    else tpd.sorted_indices
                ),
                (
                    torch.cat((spd.unsorted_indices, tpd.unsorted_indices))
                    if spd.unsorted_indices is not None
                    else tpd.unsorted_indices
                ),
            )
            offs = {
                'data': int(spd.data.shape[0]),
                'bs': int(spd.batch_sizes.shape[0]),
                'idx': 0 if spd.sorted_indices is None else int(spd.sorted_indices.shape[0]),
            }

        elif self.enc_input_type == EncoderInputType.JAGGED:
            spd = src_input_data.get_jagged()
            tpd = tgt_input_data.get_jagged()

            combined.batch = JaggedInputTensor(
                data=torch.cat((spd.data, tpd.data)), lengths=torch.cat((spd.lengths, tpd.lengths))
            )
            offs = {'data': int(spd.data.shape[0]), 'lng': int(spd.lengths.shape[0])}
        elif self.enc_input_type == EncoderInputType.PADDED:
            spd = src_input_data.get_padded()
            tpd = tgt_input_data.get_padded()

            sdv = spd.data.view(-1)
            tdv = tpd.data.view(-1)

            combined.batch = PaddedTensor(
                data=torch.cat((sdv, tdv)), lengths=torch.cat((spd.lengths, tpd.lengths))
            )
            offs = {'data': int(sdv.shape[0]), 'lng': int(spd.lengths.shape[0])}

        else:
            raise RuntimeError(f"Unsupported enc input type: {self.enc_input_type}")

        assert (
            src_text_repr.text_repr_type == tgt_text_repr.text_repr_type
        ), "src and tgt have different text_repr_type."

        combined_texts_repr = TextsRepr(src_text_repr.text_repr_type, [], [])
        # TODO TEMP
        if not self.ser_full_text_info:
            if (
                src_text_repr.second_level_lengths is not None
                and tgt_text_repr.second_level_lengths is not None
            ):

                combined_texts_repr.second_level_lengths = torch.cat(
                    (src_text_repr.second_level_lengths, tgt_text_repr.second_level_lengths)
                )
                offs['tr_s'] = int(src_text_repr.second_level_lengths.shape[0])

            if (
                src_text_repr.third_level_lengths is not None
                and tgt_text_repr.third_level_lengths is not None
            ):
                combined_texts_repr.third_level_lengths = torch.cat(
                    (src_text_repr.third_level_lengths, tgt_text_repr.third_level_lengths)
                )
                offs['tr_t'] = int(src_text_repr.third_level_lengths.shape[0])

        slot_num, info = self.share_input_data(combined, combined_texts_repr, labels)
        info['offs'] = offs

        return slot_num, info

    def recreate_training_data(self, slot_num: int, info: dict):
        assert (
            self.is_training
        ), "EncInputSharedTensors::serialize_training_data: created without is_training=True"

        shapes = info['shapes']
        offs = info['offs']
        tens = self._shared_tensors_holder.get_tensors(slot_num, shapes)
        inp_tens = tens[:-3]
        src_inp = SeqEncoderBatchedInput(self.enc_input_type)
        tgt_inp = SeqEncoderBatchedInput(self.enc_input_type)
        if self.enc_input_type == EncoderInputType.PACKED:
            d_offs = offs['data']
            bs_offs = offs['bs']
            i_offs = offs['idx']
            src_inp.batch = torch.nn.utils.rnn.PackedSequence(
                inp_tens[0][:d_offs],
                inp_tens[1][:bs_offs],
                None if i_offs == 0 else inp_tens[2][:i_offs],
                None if i_offs == 0 else inp_tens[3][:i_offs],
            )
            tgt_inp.batch = torch.nn.utils.rnn.PackedSequence(
                inp_tens[0][d_offs:],
                inp_tens[1][bs_offs:],
                inp_tens[2][i_offs:],
                inp_tens[3][i_offs:],
            )
        elif self.enc_input_type == EncoderInputType.JAGGED:
            d_offs = offs['data']
            l_offs = offs['lng']
            src_inp.batch = JaggedInputTensor(inp_tens[0][:d_offs], inp_tens[1][:l_offs])
            tgt_inp.batch = JaggedInputTensor(inp_tens[0][d_offs:], inp_tens[1][l_offs:])
        elif self.enc_input_type == EncoderInputType.PADDED:
            d_offs = offs['data']
            l_offs = offs['lng']
            src_inp.batch = PaddedTensor(inp_tens[0][:d_offs], inp_tens[1][:l_offs])
            tgt_inp.batch = PaddedTensor(inp_tens[0][d_offs:], inp_tens[1][l_offs:])
        else:
            raise RuntimeError(f"Unsupported enc input type: {self.enc_input_type}")

        self._recreate_derivatives(src_inp, info)
        if 'sorted_by_length' in info:
            # Since src and tgt are combined into one tensor they share the same
            # info object. We pull this property from src data, since usually
            # when src data is sorted then tgt data is not sorted to keep
            # alignment.
            del info['sorted_by_length']
        self._recreate_derivatives(tgt_inp, info)

        if self.enc_input_type == EncoderInputType.PADDED:
            sb = src_inp.batch
            src_inp.batch = sb._replace(data=sb.data.reshape(src_inp.batch_size, src_inp.max_len))
            tb = tgt_inp.batch
            tgt_inp.batch = tb._replace(data=tb.data.reshape(tgt_inp.batch_size, tgt_inp.max_len))

        src_text_repr = TextsRepr(info['text_repr_type'], [], [])
        tgt_text_repr = TextsRepr(info['text_repr_type'], [], [])
        second_lvl_t = tens[-3]
        if second_lvl_t is not None:
            s_offs = offs['tr_s']
            src_text_repr.second_level_lengths = second_lvl_t[:s_offs]
            tgt_text_repr.second_level_lengths = second_lvl_t[s_offs:]
        third_lvl_t = tens[-2]
        if third_lvl_t is not None:
            t_offs = offs['tr_t']
            src_text_repr.third_level_lengths = third_lvl_t[:t_offs]
            tgt_text_repr.third_level_lengths = third_lvl_t[t_offs:]

        labels = tens[-1]
        return (src_inp, src_text_repr), (tgt_inp, tgt_text_repr), labels

    def release_slot(self, slot_num):
        self._shared_tensors_holder.release_slot(slot_num)
