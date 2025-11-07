import os
from vaderx.pipeline import EditConfig, process_video

def test_import_and_cfg():
    cfg = EditConfig()
    assert cfg.start_sec == 0.0
