#!/usr/bin/env python3

from doc_enc.encoders.frag_encoder import FragEncoder
from doc_enc.encoders.enc_config import DocEncoderConf


class DocEncoder(FragEncoder):
    def __init__(self, conf: DocEncoderConf, encoder, prev_output_size):
        super().__init__(conf, encoder, prev_output_size)
