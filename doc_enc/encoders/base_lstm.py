#!/usr/bin/env python3
#!/usr/bin/env python3

import torch
from torch import nn

from doc_enc.common_types import EncoderKind, PoolingStrategy
from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.encoders.enc_in import EncoderInputType, SeqEncoderBatchedInput
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.base_pooler import BasePoolerConf


class RNNBasePooler:

    def __init__(self, emb_dim, conf: BasePoolerConf):
        if conf.out_size is not None and conf.out_size != emb_dim:
            raise RuntimeError("out_size is not equal to emb_dim. It is not supported in RNNPooler")
        if conf.force_dense_layer:
            # TODO it is quite easy to add support
            raise RuntimeError("force dense layer is not supported in RNNPooler")
        if conf.use_activation:
            # TODO it is quite easy to add support
            raise RuntimeError("activation is not supported in RNNPooler")

        if conf.pooling_strategy not in (PoolingStrategy.MAX, PoolingStrategy.MEAN):
            raise RuntimeError(f"RNNPooler: unsupported pooling strategy: {conf.pooling_strategy}")

        self.cap_m = 20

    def __call__(
        self, packed_seq: nn.utils.rnn.PackedSequence
    ) -> tuple[torch.Tensor, torch.Tensor]:

        data = packed_seq.data
        pbs = packed_seq.batch_sizes.tolist()

        emb_dim = data.shape[-1]
        max_real_seq_length = int(packed_seq.batch_sizes.size(0))
        max_batch_size = pbs[0]

        result = torch.zeros(max_batch_size, emb_dim, device=data.device, dtype=data.dtype)
        lengths: list[int] = []

        # Packed sequence's data is 1d tensor:
        # [i1b1, i1b2, i1b3, ..., i1bn,i2b1, i2b2, ..., i2bn, i3b1, ... i3bn-1, ... isb1, isb2]
        # and corresponding batch sizes:
        # [n, n, ni-1, ... 2]

        # The conventional method is to represent this sequence as padded tensor
        # and pool embeddings from it. But pad_packed_sequence uses slice/copy
        # operations in a for loop that is quite slow especially at backward
        # step. We try here to represent some chunk of packed sequence as padded
        # tensor and run reduce operation on it. Chunk is created efficiently
        # using scatter operator.

        # Temp padded tensor of shape [q+1,m] (in code [buf_len, buf_max_batch_size]), where m <= n, q <= s:
        # txby is intermediate results from previous chunks if any.
        # t1b1, t1b2, t1b3,   ...,    t1bm
        # i1b1, i1b2, i1b3,   ...,    i1bm
        # i2b1, i2b2, i2b3,   ...,    i2bm
        # i3b1, ...,  ...,    i3bn-1, x
        # ...
        # iqb1, iqb2, x, ..., x,      x

        interim_results: torch.Tensor | None = None
        data_offset = 0
        done_offs = max_batch_size

        cur_len = 0
        buf_max_batch_size: int = max_batch_size
        buf_len: int = 0
        buf_offs: int = 0
        buf_max_elems_cnt: int = max_batch_size * self.cap_m
        buf_t = torch.full(
            (buf_max_elems_cnt, emb_dim),
            self._pad_value(),
            dtype=data.dtype,
            device=data.device,
        )
        while cur_len < max_real_seq_length:
            buf_offs = 0 if interim_results is None else int(interim_results.shape[0])
            buf_len = (buf_max_elems_cnt - buf_offs) // buf_max_batch_size
            buf_len = min(buf_len, max_real_seq_length - cur_len)

            if interim_results is not None:
                buf_t.fill_(self._pad_value())
                buf_t[:buf_offs] = interim_results

            idxs = []

            data_elems_cnt = 0
            batch_size = buf_max_batch_size
            prev_batch_size = batch_size
            for i in range(cur_len, cur_len + buf_len):
                batch_size = int(pbs[i])
                data_elems_cnt += batch_size
                idxs.extend(range(buf_offs, buf_offs + batch_size))
                pad_elems_cnt = buf_max_batch_size - batch_size
                buf_offs += batch_size + pad_elems_cnt
                # update lengths
                for _ in range(prev_batch_size - batch_size):
                    lengths.append(i)
                prev_batch_size = batch_size

            cur_len += buf_len
            idx_t = torch.tensor(idxs, dtype=torch.int64, device=buf_t.device)
            data_slice = data[data_offset : data_offset + data_elems_cnt]
            data_offset += data_elems_cnt

            buf_t.scatter_(0, idx_t.unsqueeze(1).expand(-1, emb_dim), data_slice)
            real_len = buf_len + (1 if interim_results is not None else 0)
            interim_results = self._reduce_impl(
                buf_t[:buf_offs].view(real_len, buf_max_batch_size, emb_dim)
            )

            # Finalize sequences that have no more elements. They are resided in
            # the end of interim_results since all sequences are sorted in
            # decreasing of their lengths.
            done_cnt = (
                buf_max_batch_size - batch_size
                if cur_len < max_real_seq_length
                else buf_max_batch_size
            )

            if done_cnt:
                result[done_offs - done_cnt : done_offs] = interim_results[-done_cnt:]
                # Leave only sequences that are not ready yet.
                # interim_results: [batch_size, *]
                interim_results = interim_results[:-done_cnt]
                done_offs -= done_cnt

            buf_max_batch_size = batch_size

        for _ in range(buf_max_batch_size):
            lengths.append(max_real_seq_length)
        lengths_t = torch.tensor(list(reversed(lengths)), dtype=torch.int64, device=data.device)
        result = self._post(result, lengths_t)

        if packed_seq.unsorted_indices is not None:
            return (
                result.index_select(0, packed_seq.unsorted_indices),
                lengths_t[packed_seq.unsorted_indices],
            )

        return result, lengths_t

    def _reduce_impl(self, ten: torch.Tensor) -> torch.Tensor:
        """ten is 3d tensor [cur_seq_len, cur_batch_size, emb_dim]"""
        raise NotImplementedError("reduce_impl")

    def _post(self, accum: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_post")

    def _pad_value(self) -> float:
        raise NotImplementedError("_pad_value")


class RNNMaxPooler(RNNBasePooler):

    def _reduce_impl(self, ten: torch.Tensor) -> torch.Tensor:
        return torch.max(ten, dim=0)[0]

    def _pad_value(self):
        return float('-inf')

    def _post(self, accum: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return accum


class RNNMeanPooler(RNNBasePooler):

    def _reduce_impl(self, ten: torch.Tensor) -> torch.Tensor:
        return torch.sum(ten, dim=0)

    def _pad_value(self):
        return 0.0

    def _post(self, accum: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return accum / lengths.unsqueeze(-1)


class RNNPooler:
    def __init__(self, emb_dim, conf: BasePoolerConf):
        if conf.pooling_strategy == PoolingStrategy.MAX:
            self.pooler = RNNMaxPooler(emb_dim, conf)
        elif conf.pooling_strategy == PoolingStrategy.MEAN:
            self.pooler = RNNMeanPooler(emb_dim, conf)
        else:
            raise RuntimeError(f"Unsupported pooling strategy: {conf.pooling_strategy}")

    def __call__(
        self, packed_seq: nn.utils.rnn.PackedSequence
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pooler(packed_seq)


class BaseRNNEncoder(BaseEncoder):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__()

        self.conf = conf
        if conf.input_type is not None and conf.input_type != EncoderInputType.PACKED:
            raise RuntimeError(f"Unsupported input type: {self.conf.input_type}")

        proj_size = 0
        if conf.proj_size is not None:
            proj_size = conf.proj_size

        if conf.encoder_kind == EncoderKind.LSTM:
            rnn_cls = nn.LSTM
            kwargs = {'proj_size': proj_size}
        elif conf.encoder_kind == EncoderKind.GRU:
            rnn_cls = nn.GRU
            kwargs = {}
        else:
            raise RuntimeError(f"Unsuppored rnn kind: {conf.encoder_kind}")

        self.rnn = rnn_cls(
            input_size=conf.input_size,
            hidden_size=conf.hidden_size,
            num_layers=conf.num_layers,
            bidirectional=conf.bidirectional,
            dropout=conf.dropout,
            **kwargs,
        )
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                param.data.uniform_(-0.1, 0.1)

        rnn_output_units = conf.hidden_size if not proj_size else proj_size
        if conf.bidirectional:
            rnn_output_units *= 2

        self.pooler = RNNPooler(rnn_output_units, conf.pooler)

        self.rnn_output_units = rnn_output_units
        self.output_units = rnn_output_units
        if self.conf.pooler.out_size is not None:
            self.output_units = self.conf.pooler.out_size

    def input_type(self) -> EncoderInputType:
        return EncoderInputType.PACKED

    def out_embs_dim(self):
        return self.output_units

    def forward(self, input_batch: SeqEncoderBatchedInput, **kwargs) -> BaseEncoderOut:
        if not input_batch.embedded:
            raise RuntimeError("Only embs as input are supported")

        # bsz, seqlen = input_embs.size()[:2]
        # # BS x SeqLen x Dim -> SeqLen x BS x DIM
        # x = input_embs.transpose(0, 1)

        # # pack embedded source tokens into a PackedSequence
        # packed_x = nn.utils.rnn.pack_padded_sequence(
        #     x, lengths.cpu(), enforce_sorted=enforce_sorted
        # )

        packed_x = input_batch.get_packed_seq()
        packed_outs, _ = self.rnn(packed_x)

        # Conventional implementation with pad_packed_sequence and then pooling
        # padded tensor is quite slow especially at backward step. We do not
        # need padded tensor anyway so just pool embeddings from packed
        # sequence.
        pooled_out, lengths = self.pooler(packed_outs)

        return BaseEncoderOut(pooled_out, None, lengths)


class LSTMEncoder(BaseRNNEncoder):
    pass


class GRUEncoder(BaseRNNEncoder):
    pass
