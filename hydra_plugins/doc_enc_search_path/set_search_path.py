#!/usr/bin/env python3
#
import os
import sys

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class SetSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:

        if sys.argv[0].endswith('run_training'):
            p = os.environ.get("TRAIN_CONFIG_PATH", "/train/conf")
        elif sys.argv[0].endswith('run_eval'):
            p = os.environ.get("EVAL_CONFIG_PATH", "/eval/conf")
        else:
            p = None

        if p is not None:
            search_path.prepend(provider="searchpath-plugin", path=f"file://{p}")
