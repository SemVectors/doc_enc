#!/usr/bin/env python3
#
import os

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class SetSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        p = os.environ.get("TRAIN_CONFIG_PATH", "/train/conf")
        search_path.prepend(provider="searchpath-plugin", path=f"file://{p}")
