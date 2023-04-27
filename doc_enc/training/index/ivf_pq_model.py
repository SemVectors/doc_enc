#!/usr/bin/env python3

import logging
from typing import Dict, List
import os
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np

import faiss


from doc_enc.training.index.index_train_conf import IndexTrainConf
from doc_enc.training.dist_util import dist_gather_varsize_tensor, dist_gather_tensor

# modified version from the LibVQ library
class IVFCPU(nn.Module):
    """
    Efficiently training IVF on CPU. In this way you can use a large-scale IVF index.
    """

    def __init__(self, center_vecs: np.ndarray, id2center: Dict[int, int]):
        """
        :param center_vecs: Vectors for all centers
        :param id2center: Mapping Doc id to center id
        """
        super().__init__()
        self.center_vecs = center_vecs
        self.id2center = id2center
        self.center_grad = np.zeros_like(center_vecs)
        self.ivf_centers_num = len(center_vecs)

    @classmethod
    def _compute_id2center(cls, ivf_index, centers_cnt):
        invlists = ivf_index.invlists
        id2center = {}
        max_per_centroid = 0
        min_per_centroid = 0
        avg_per_centroid = 0
        vectors = 0
        for i in range(centers_cnt):
            ls = invlists.list_size(i)
            list_ids = faiss.rev_swig_ptr(invlists.get_ids(i), ls)

            max_per_centroid = max(max_per_centroid, len(list_ids))
            min_per_centroid = min(min_per_centroid, len(list_ids))
            avg_per_centroid += len(list_ids)
            vectors += len(list_ids)

            for docid in list_ids:
                id2center[docid] = i

        avg_per_centroid /= centers_cnt
        logging.info(
            "max_per_centroid=%s, min_per_centroid=%s, avg_per_centroid=%s, vectors=%s",
            max_per_centroid,
            min_per_centroid,
            avg_per_centroid,
            vectors,
        )
        return id2center

    @classmethod
    def from_faiss_index(cls, index_file):
        logging.info('loading IVF from Faiss index: %s', index_file)

        index = faiss.read_index(index_file)
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = index

        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        coarse_embeds = np.empty(coarse_quantizer.codes.size() // 4, dtype=np.float32)
        faiss.memcpy(
            faiss.swig_ptr(coarse_embeds), coarse_quantizer.codes.data(), coarse_embeds.nbytes
        )
        center_vecs = coarse_embeds.reshape((-1, ivf_index.d))
        id2center = cls._compute_id2center(ivf_index, len(center_vecs))

        ivf = cls(center_vecs, id2center)
        return ivf

    def reassign_id2center(self, index_file):
        index = faiss.read_index(index_file)
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = index
        self.id2center = self._compute_id2center(ivf_index, len(self.center_vecs))

    def set_batch_centers(self, batch_centers_index: List[int], device: torch.device):
        c_embs = self.center_vecs[batch_centers_index]
        self.batch_centers_index = batch_centers_index
        self.batch_center_vecs = torch.FloatTensor(c_embs).to(device)
        self.batch_center_vecs.requires_grad = True

    def merge_and_dispatch(self, doc_ids: List[int], world_size: int):
        dc_ids = [self.id2center[i] for i in doc_ids]

        if world_size > 1:
            all_dc_ids = dist_gather_varsize_tensor(
                torch.LongTensor(dc_ids).cuda(), world_size=world_size
            )
            all_dc_ids = torch.cat(all_dc_ids, dim=0)
            all_dc_ids = list(all_dc_ids.detach().cpu().numpy())
        else:
            all_dc_ids = dc_ids

        batch_cids = sorted(list(set(all_dc_ids)))
        cid2bid = {}
        for i, c in enumerate(batch_cids):
            cid2bid[c] = i

        batch_dc_ids = torch.LongTensor([cid2bid[x] for x in dc_ids])
        return batch_cids, batch_dc_ids

    def select_centers(self, doc_ids: List[int], device: torch.device, world_size: int):
        batch_cids, batch_dc_ids = self.merge_and_dispatch(doc_ids, world_size=world_size)

        self.set_batch_centers(batch_cids, device)
        batch_dc_ids = batch_dc_ids.to(device)
        dc_emb = self.batch_center_vecs.index_select(dim=0, index=batch_dc_ids)
        return dc_emb

    def grad_accumulate(self, world_size: int):
        if world_size > 1:
            grad = dist_gather_tensor(
                self.batch_center_vecs.grad.unsqueeze(0), world_size=world_size
            )
            grad = torch.mean(grad, dim=0)
            self.batch_center_vecs.grad = grad

        clip_grad_norm_(self.batch_center_vecs, 1.0)
        grad = self.batch_center_vecs.grad.detach().cpu().numpy()

        self.center_grad[self.batch_centers_index] += grad

    def update_centers(self, lr: float):
        self.center_vecs = self.center_vecs - lr * self.center_grad

    def zero_grad(self):
        self.center_grad = np.zeros_like(self.center_grad)

    def update_faiss_index(self, index):
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
            coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        else:
            coarse_quantizer = faiss.downcast_index(index.quantizer)

        faiss.memcpy(
            coarse_quantizer.codes.data(),
            faiss.swig_ptr(self.center_vecs.ravel()),
            self.center_vecs.nbytes,
        )


class Quantization(nn.Module):
    """
    End-to-end Product Quantization
    """

    def __init__(
        self,
        emb_size: int = 768,
        subvector_num: int = 96,
        subvector_bits: int = 8,
        rotate: np.ndarray = None,
        codebook: np.ndarray = None,
    ):
        """
        :param emb_size: Dim of embeddings
        :param subvector_num: The number of codebooks
        :param subvector_bits: The number of codewords in each codebook
        :param rotate:  The rotate Matrix. Used for OPQ
        :param codebook: The parameter for codebooks. If set None, it will randomly initialize the codebooks.
        """
        super().__init__()

        if codebook is not None:
            self.codebook = nn.Parameter(torch.FloatTensor(codebook), requires_grad=True)
        else:
            self.codebook = nn.Parameter(
                torch.empty(subvector_num, 2**subvector_bits, emb_size // subvector_num).uniform_(
                    -0.1, 0.1
                )
            ).type(torch.FloatTensor)
        self.subvector_num = self.codebook.size(0)
        self.subvector_bits = int(math.log2(self.codebook.size(1)))

        if rotate is not None:
            self.rotate = nn.Parameter(torch.FloatTensor(rotate), requires_grad=False)
        else:
            self.rotate = None

    @classmethod
    def from_faiss_index(cls, index_file: str):
        logging.info('loading PQ from Faiss index: %s', index_file)
        index = faiss.read_index(index_file)

        if isinstance(index, faiss.IndexPreTransform):
            vt = faiss.downcast_VectorTransform(index.chain.at(0))
            assert isinstance(vt, faiss.LinearTransform)
            rotate = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
            pq_index = faiss.downcast_index(index.index)
        else:
            pq_index = index
            rotate = None

        centroid_embeds = faiss.vector_to_array(pq_index.pq.centroids)
        codebook = centroid_embeds.reshape(pq_index.pq.M, pq_index.pq.ksub, pq_index.pq.dsub)
        subvector_num = pq_index.pq.M

        pq = cls(subvector_num=subvector_num, rotate=rotate, codebook=codebook)
        return pq

    def rotate_vec(self, vecs):
        if self.rotate is None:
            return vecs
        return torch.matmul(vecs, self.rotate.T)

    def code_selection(self, vecs):
        vecs = vecs.view(vecs.size(0), self.subvector_num, -1)
        codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)
        proba = -torch.sum((vecs.unsqueeze(-2) - codebook) ** 2, -1)
        assign = F.softmax(proba, -1)
        return assign

    def STEstimator(self, assign):
        index = assign.max(dim=-1, keepdim=True)[1]
        assign_hard = torch.zeros_like(assign, device=assign.device, dtype=assign.dtype).scatter_(
            -1, index, 1.0
        )
        return assign_hard.detach() - assign.detach() + assign

    def quantized_vecs(self, assign):
        assign = self.STEstimator(assign)
        assign = assign.unsqueeze(2)
        codebook = self.codebook.unsqueeze(0).expand(assign.size(0), -1, -1, -1)
        quantized_vecs = torch.matmul(assign, codebook).squeeze(2)
        quantized_vecs = quantized_vecs.view(assign.size(0), -1)
        return quantized_vecs

    def quantization(self, vecs):
        assign = self.code_selection(vecs)
        quantized_vecs = self.quantized_vecs(assign)
        return quantized_vecs

    def forward(self, vecs):
        return self.quantization(vecs)

    def update_faiss_index(self, index):
        cb = self.codebook.detach().cpu().numpy().ravel()
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
            faiss.copy_array_to_vector(cb, ivf_index.pq.centroids)
        else:
            faiss.copy_array_to_vector(cb, index.pq.centroids)


class TrainableIvfPQ(nn.Module):
    """
    Learnable VQ model, supports both IVF and PQ.
    """

    def __init__(self, config: IndexTrainConf):
        super().__init__()

        self._config = config
        self._world_size = dist.get_world_size()
        self.ivf: IVFCPU = IVFCPU.from_faiss_index(config.init_index_file)
        self.pq: Quantization = Quantization.from_faiss_index(config.init_index_file)

    def compute_score(self, query_vecs: torch.Tensor, tgt_vecs: torch.Tensor, normalize: bool):
        if normalize:
            query_vecs = F.normalize(query_vecs, p=2, dim=1)
            tgt_vecs = F.normalize(tgt_vecs, p=2, dim=1)

        m = torch.mm(query_vecs, tgt_vecs.t())  # bsz x target_bsz
        return m

    def forward(
        self,
        query_vecs: torch.FloatTensor,
        tgt_vecs: torch.FloatTensor,
        tgt_ids: torch.LongTensor,
        normalize=True,
    ):

        dc_emb = self.ivf.select_centers(tgt_ids, query_vecs.device, world_size=self._world_size)

        residual_tgt_vecs = tgt_vecs - dc_emb
        quantized_doc = self.pq(residual_tgt_vecs) + dc_emb

        ivf_score_matrix = self.compute_score(query_vecs, dc_emb, normalize=normalize)

        pq_score_matrix = self.compute_score(query_vecs, quantized_doc, normalize=normalize)
        return ivf_score_matrix, pq_score_matrix

    def update_faiss_index(self, index):
        if not self._config.ivf.fixed:
            self.ivf.update_faiss_index(index)
        self.pq.update_faiss_index(index)

    def index_path(self, save_path, name_prefix):
        c = self._config
        filename = f"trained_{name_prefix}_IVF{c.ivf_centers_num}_PQ{c.subvector_num}x{c.subvector_bits}.faiss"
        index_path = os.path.join(save_path, filename)
        return index_path

    def save_as_faiss_index(self, save_path, name_prefix):
        index_path = self.index_path(save_path, name_prefix)
        if not os.path.exists(index_path):
            index = faiss.read_index(self._config.init_index_file)
            index.reset()
        else:
            index = faiss.read_index(index_path)

        self.update_faiss_index(index)
        faiss.write_index(index, index_path)
        return index_path
