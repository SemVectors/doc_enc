#!/usr/bin/env python3

import itertools
import logging
import enum

from typing import NamedTuple


import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from doc_enc.embs.token_embed import TokenEmbedding
from doc_enc.encoders.pad_utils import PadOpts, create_padded_tensor, pad_embs_seq

# * In


class EncoderInputType(enum.Enum):
    PADDED = 0
    PACKED = 1
    JAGGED = 2


class JaggedInputTensor(NamedTuple):
    data: torch.Tensor
    lengths: torch.Tensor


class JaggedWithPosIds(NamedTuple):
    data: torch.Tensor
    lengths: torch.Tensor
    position_ids: torch.Tensor


class PaddedTensor(NamedTuple):
    data: torch.Tensor
    lengths: torch.Tensor
    padding_mask: torch.Tensor | None = None


# for padded
# def _prepare_input_data(self, doc_segments: list[list[int]], already_sorted=False):
#     max_len = len(max(doc_segments, key=len))

#     tokens_tensor, lengths_tensor = create_padded_tensor(
#         doc_segments,
#         max_len,
#         pad_idx=self._pad_idx,
#         device=self.device,
#         pad_to_multiple_of=self.first_encode_layer().pad_to_multiple_of,
#         padding_side=self.first_encode_layer().get_padding_side(),
#     )

#     return _InputData(
#         tokens_tensor=tokens_tensor,
#         lengths_tensor=lengths_tensor,
#         already_sorted=already_sorted,
#     )


class TextReprType(enum.Enum):
    SEQ_OF_TOKENS = 0
    SEQ_OF_SENTS = 1
    SEQ_OF_FRAGMENTS = 2
    SEQ_OF_FRAGMENTS_OF_SENTS = 3


class TextsRepr:
    def __init__(
        self,
        text_repr_type: TextReprType,
        text_segments: list[list[int]],
        text_lengths: list[list[int]],
    ):
        """text_segments - flat list that holds multiple texts. Each element of
        the list is a sequence of tokens that might be part of a text (fragment,
        sentence) or the whole text. In combination with text_lengths encodes
        various text representations. The simplest case: text is a sequence of
        tokens.

        text_segments: [ [text1_tok0, text1_tok1,..., text1_tok99],
                         [text2_tok0, text2_tok1, ...] ]

        text_lengths: [ [1], [1], ...]

        Text is the sequence of sentences:

        text_segments: [ [text1_sent0_tok0, ..., text1_sent0_tok39],
                         ...
                         [text1_sent9_tok0, ..., text1_sent9_tok9],
                         ...
                         [text2_sent0_tok0, ..., text1_sent0_tok19]]

        text_lengths: [ [10], [20], ...]

        Text is the sequence of sentences grouped into fragments:

        text_segments: [ [text1_sent0_frag0_tok0, ..., text1_sent0_frag0_tok39],
                         ...
                         [text1_sent15_frag0_tok0, ..., text1_sent15_frag0_tok39],
                         [text1_sent0_frag1_tok0, ..., text1_sent0_frag1_tok19],
                         ...
                       ]

        text_lengths: [ [16, 16, 5], [16, 16, 4], ...]
        """
        self.text_repr_type = text_repr_type
        self.flat_tokens = torch.tensor([t for s in text_segments for t in s], dtype=torch.int32)
        self.first_level_lengths = torch.tensor([len(s) for s in text_segments], dtype=torch.int32)
        self.second_level_lengths = None
        self.third_level_lengths = None

        if text_lengths:
            self.second_level_lengths = torch.tensor(
                [lng for s in text_lengths for lng in s], dtype=torch.int32
            )
            self.third_level_lengths = torch.tensor(
                [len(s) for s in text_lengths], dtype=torch.int32
            )

    def nsents(self) -> int:
        if self.text_repr_type in (
            TextReprType.SEQ_OF_FRAGMENTS_OF_SENTS,
            TextReprType.SEQ_OF_SENTS,
        ):
            assert self.second_level_lengths is not None, "nsents: second level is none"
            return int(self.second_level_lengths.sum().item())
        return 0

    def nfrags(self) -> int:
        if self.text_repr_type == TextReprType.SEQ_OF_FRAGMENTS_OF_SENTS:
            assert self.second_level_lengths is not None, "nfrags: second level is none"
            return int(self.second_level_lengths.shape[0])
        if self.text_repr_type == TextReprType.SEQ_OF_FRAGMENTS:
            assert self.second_level_lengths is not None, "nfrags: second level is none"
            return int(self.second_level_lengths.sum().item())
        return 0

    def fragment_lengths_in_sents_tensor(self) -> torch.Tensor:
        if self.text_repr_type == TextReprType.SEQ_OF_FRAGMENTS_OF_SENTS:
            assert (
                self.second_level_lengths is not None
            ), "fragments_lengths_in_sents: second level is none"
            return self.second_level_lengths
        return torch.empty(0)

    def fragment_lengths_in_sents(self) -> list[int]:
        return self.fragment_lengths_in_sents_tensor().tolist()

    def text_lengths_in_sents_tensor(self) -> torch.Tensor:
        if self.text_repr_type == TextReprType.SEQ_OF_FRAGMENTS_OF_SENTS:
            assert (
                self.third_level_lengths is not None and self.second_level_lengths is not None
            ), "text_lengths_in_sents: third or second levels is None"

            frags_len = self.third_level_lengths.tolist()
            sl = []
            offs = 0
            for fl in frags_len:
                tl = self.second_level_lengths[offs : offs + fl].sum().item()
                sl.append(int(tl))
                offs += fl
            return torch.tensor(sl)
        if self.text_repr_type == TextReprType.SEQ_OF_SENTS:
            assert (
                self.second_level_lengths is not None
            ), "text_lengths_in_sents: second level is none"
            return self.second_level_lengths

        return torch.empty(0)

    def text_lengths_in_sents(self) -> list[int]:
        return self.text_lengths_in_sents_tensor().tolist()

    def text_lengths_in_fragments_tensor(self) -> torch.Tensor:
        if self.text_repr_type == TextReprType.SEQ_OF_FRAGMENTS_OF_SENTS:
            assert (
                self.third_level_lengths is not None
            ), "text_lengths_in_fragments: third level is None"

            return self.third_level_lengths
        if self.text_repr_type == TextReprType.SEQ_OF_FRAGMENTS:
            assert (
                self.second_level_lengths is not None
            ), "text_lengths_in_fragments: second level is none"
            return self.second_level_lengths

        return torch.empty(0)

    def text_lengths_in_fragments(self) -> list[int]:
        return self.text_lengths_in_fragments_tensor().tolist()

    def summary(self) -> str:
        if self.text_repr_type == TextReprType.SEQ_OF_FRAGMENTS_OF_SENTS:
            nsents = self.nsents()
            nfrags = self.nfrags()
            return f'nsents={nsents}, nfrags={nfrags}'
        elif self.text_repr_type == TextReprType.SEQ_OF_FRAGMENTS:
            return f'nfrags={self.nfrags()}'
        elif self.text_repr_type == TextReprType.SEQ_OF_SENTS:
            return f'nsents={self.nsents()}'
        return ''


def _call_fn_on_packed_seq(fn, ps: PackedSequence):
    return PackedSequence(fn(ps.data), ps.batch_sizes, ps.sorted_indices, ps.unsorted_indices)


class SeqEncoderBatchedInput:
    @classmethod
    def from_input_ids(
        cls,
        input_type: EncoderInputType,
        text_segments: list[list[int]],
        pad_idx: int,
        pad_opts: PadOpts,
        sorted_by_length: bool = False,
    ):
        batched_input = cls(input_type)
        batched_input.sorted_by_length = sorted_by_length
        batched_input.max_len = max(len(s) for s in text_segments)
        batched_input.batch_size = len(text_segments)
        # logging.error(
        #     'call %s SeqEncoderBatchedInput %s max len %s',
        #     input_type,
        #     batched_input.batch_size,
        #     batched_input.max_len,
        # )

        if input_type == EncoderInputType.PADDED:
            max_len = len(max(text_segments, key=len))
            tokens_tensor, lengths_tensor = create_padded_tensor(
                text_segments, max_len, pad_idx=pad_idx, device=torch.device('cpu'), opts=pad_opts
            )
            batched_input.batch = PaddedTensor(tokens_tensor, lengths_tensor)
            return batched_input
        if input_type == EncoderInputType.PACKED:
            # logging.error('packed %s')
            tensor_list = [torch.tensor(s, dtype=torch.int32) for s in text_segments]
            # logging.error('tensor list %s')
            batched_input.batch = pack_sequence(tensor_list, enforce_sorted=sorted_by_length)
            # logging.error('gatched input %s', batched_input.batch)
            return batched_input
        if input_type == EncoderInputType.JAGGED:
            data = torch.tensor([t for s in text_segments for t in s], dtype=torch.int32)
            len_tensor = torch.tensor([len(s) for s in text_segments], dtype=torch.int32)
            batched_input.batch = JaggedInputTensor(data, len_tensor)
            return batched_input

        raise RuntimeError(f"Unsupported enc input type: {input_type}")

    @classmethod
    def from_embs(
        cls,
        input_type: EncoderInputType,
        embs: torch.Tensor,
        lengths: torch.Tensor,
        padded_prepend_with_zero: bool = False,
    ):

        if len(embs.shape) != 2:
            raise RuntimeError("It is expected that embs is 2d tensor for Jagged input type.")

        batched_input = cls(input_type)
        batched_input.embedded = True
        if input_type == EncoderInputType.JAGGED:
            batched_input.batch = JaggedInputTensor(embs, lengths)
        elif input_type == EncoderInputType.PADDED:
            embs, mask = pad_embs_seq(embs, lengths, prepend_with_zero=padded_prepend_with_zero)
            extra_len = int(padded_prepend_with_zero)
            batched_input.batch = PaddedTensor(embs, lengths + extra_len, mask)
            batched_input.padded_prepended_with_0 = padded_prepend_with_zero
        else:
            raise RuntimeError("Unsupported input type %s", input_type)

        return batched_input

    def __init__(
        self,
        input_type: EncoderInputType,
        # text_segments: list[list[int]],
        # text_lengths: list[list[int]],
        # text_ids: list[str | int],
    ):
        self.enc_input_type = input_type

        # self.text_lengths = text_lengths
        # self.text_ids = text_ids

        self.batch: None | PackedSequence | JaggedInputTensor | PaddedTensor = None
        # self.seqs_cnt = len(text_segments)
        # self.total_tokens_cnt = sum(len(s) for s in text_segments)
        self.embedded = False
        self.sorted_by_length = False

        self.max_len = 0
        self.batch_size = 0

        # props
        self.padded_prepended_with_0 = False

    def embed_(self, embed: TokenEmbedding):
        self.embedded = True
        if isinstance(self.batch, PackedSequence):
            self.batch = _call_fn_on_packed_seq(embed, self.batch)
        elif isinstance(self.batch, (JaggedInputTensor, PaddedTensor)):
            self.batch = self.batch._replace(data=embed(self.batch.data))
        else:
            raise RuntimeError(
                f"Batch is either uninitialized or has wrong format: {type(self.batch)}!"
            )

    def to_(self, device: torch.device):
        if isinstance(self.batch, PackedSequence):
            self.batch = self.batch.to(device)
            return self
        if isinstance(self.batch, (JaggedInputTensor, PaddedTensor)):
            b = self.batch
            self.batch = b._replace(data=b.data.to(device), lengths=b.lengths.to(device))
            return self

        raise RuntimeError(
            f"Batch is either uninitialized or has wrong format: {type(self.batch)}!"
        )

    def init_padding_mask_(self, pad_idx: int, padding_side: str):
        if self.enc_input_type == EncoderInputType.PADDED:
            if self.embedded:
                raise RuntimeError(
                    "Init padding mask is not supported when input is already embedded."
                )
            pd = self.get_padded()
            mask = torch.full_like(pd.data, True, dtype=torch.bool)
            mask[pd.data == pad_idx] = False
            if padding_side == 'left' and pd.data[0, -1] == pad_idx:
                # In some models pad_token == end_of_text_token (Qwen3-emb)
                mask[:, -1] = True
            self.batch = pd._replace(padding_mask=mask)

    def prepend_tensor_(self, prep_ten: torch.Tensor):

        if isinstance(self.batch, JaggedInputTensor):
            lengths = self.batch.lengths
            orig = self.batch.data
            assert len(orig.shape) == 2, "Unsupported dimension prepend_tensor_ jagged"
            emb_dim = orig.shape[-1]
            new_data = torch.zeros(
                orig.shape[0] + lengths.shape[0],
                emb_dim,
                dtype=orig.dtype,
                device=orig.device,
            )
            lengths_as_list = lengths.tolist()
            idxs = []
            prep_poses = []
            offs = 1
            for lt in lengths_as_list:
                prep_poses.append(offs - 1)
                idxs.extend([i] for i in range(offs, offs + lt))
                offs += lt + 1

            # idxs_t: [N, 1]
            idxs_t = torch.tensor(idxs, dtype=torch.int64, device=new_data.device)
            new_data.scatter_(0, idxs_t.expand(-1, emb_dim), orig)
            new_data[torch.tensor(prep_poses, device=orig.device)] = prep_ten.to(dtype=orig.dtype)
            self.batch = self.batch._replace(data=new_data, lengths=self.batch.lengths + 1)
            return self
        if isinstance(self.batch, PaddedTensor):
            if not self.padded_prepended_with_0:
                raise RuntimeError(
                    "prepend_tensor, padded input type, unsupported with padded_prepended_with_0 == False "
                )
            new_data = self.batch.data
            new_data[:, 0] = prep_ten
            self.batch = self.batch._replace(data=new_data, lengths=self.batch.lengths)
            self.padded_prepended_with_0 = False
            return self
        raise RuntimeError(
            f"SeqEncoderBatchedInput::prepend_tensor_: unsupported enc input type: {self.enc_input_type}"
        )

    def ntokens(self):
        if isinstance(self.batch, (PackedSequence, JaggedInputTensor)):
            return int(self.batch.data.shape[0])
        if isinstance(self.batch, PaddedTensor):
            return int(self.batch.lengths.sum().item())
        raise RuntimeError(
            f"Batch is either uninitialized or has wrong format: {type(self.batch)}!"
        )

    def get_packed_seq(self):
        if isinstance(self.batch, PackedSequence):
            return self.batch
        raise RuntimeError(
            f"Batch is either uninitialized or has wrong format: {type(self.batch)}!"
        )

    def get_jagged(self):
        if isinstance(self.batch, JaggedInputTensor):
            return self.batch
        raise RuntimeError(
            f"Batch is either uninitialized or has wrong format: {type(self.batch)}!"
        )

    def get_nested(self):
        if isinstance(self.batch, JaggedInputTensor):
            return torch.nested.nested_tensor_from_jagged(
                self.batch.data, lengths=self.batch.lengths
            )
        raise RuntimeError(
            f"Batch is either uninitialized or has wrong format: {type(self.batch)}!"
        )

    def get_jagged_w_pos_ids(self):
        if isinstance(self.batch, JaggedInputTensor):
            pos_ids = torch.tensor(
                list(itertools.chain.from_iterable(range(s) for s in self.batch.lengths)),
                dtype=torch.int32,
                device=self.batch.data.device,
            )
            return JaggedWithPosIds(self.batch.data, self.batch.lengths, pos_ids)
        raise RuntimeError(
            f"Batch is either uninitialized or has wrong format: {type(self.batch)}!"
        )

    def get_padded(self):
        if isinstance(self.batch, PaddedTensor):
            return self.batch
        raise RuntimeError(
            f"Batch is either uninitialized or has wrong format: {type(self.batch)}!"
        )

    # def tokens_cnt(self):
    #     pass

    # def seqs_cnt(self):
    #     pass


class EncoderInData(NamedTuple):
    seq_encoder_input: SeqEncoderBatchedInput
    text_ids: list[str | int]
    texts_repr: TextsRepr
