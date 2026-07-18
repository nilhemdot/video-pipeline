"""conftest.py — shared fixtures + mock heavy AI deps before import.

Tobu's engine modules import ffmpeg, torch, faster_whisper, sentence_transformers,
lancedb, transformers, fitz, frontmatter at module level. In CI / lightweight test
environments these aren't installed. This conftest injects mock modules into
sys.path BEFORE the real imports fire, so tests can run with only pytest + pydantic.
"""

import sys
import types
from unittest.mock import MagicMock


def _make_stub(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


# --- Stub heavy external deps so engine imports don't fail ---

if "ffmpeg" not in sys.modules:
    sys.modules["ffmpeg"] = MagicMock()

if "torch" not in sys.modules:
    torch_stub = MagicMock()
    torch_stub.cuda = MagicMock()
    torch_stub.cuda.is_available.return_value = False
    sys.modules["torch"] = torch_stub

if "faster_whisper" not in sys.modules:
    fw_stub = MagicMock()
    fw_stub.WhisperModel = MagicMock()
    sys.modules["faster_whisper"] = fw_stub

if "sentence_transformers" not in sys.modules:
    st_stub = MagicMock()
    st_stub.SentenceTransformer = MagicMock()
    sys.modules["sentence_transformers"] = st_stub

if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = MagicMock()

if "transformers" not in sys.modules:
    tf_stub = MagicMock()
    tf_stub.AutoTokenizer = MagicMock()
    tf_stub.AutoModelForSeq2SeqLM = MagicMock()
    sys.modules["transformers"] = tf_stub

if "fitz" not in sys.modules:
    sys.modules["fitz"] = MagicMock()

if "frontmatter" not in sys.modules:
    fm_stub = MagicMock()
    fm_stub.load = MagicMock()
    sys.modules["frontmatter"] = fm_stub

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

if "PIL" not in sys.modules:
    pil_stub = MagicMock()
    pil_stub.Image = MagicMock()
    sys.modules["PIL"] = pil_stub

if "pandas" not in sys.modules:
    sys.modules["pandas"] = MagicMock()

if "watchdog" not in sys.modules:
    wd_stub = types.ModuleType("watchdog")
    wd_observers = types.ModuleType("watchdog.observers")
    wd_observers.Observer = MagicMock()
    wd_events = types.ModuleType("watchdog.events")
    wd_events.FileSystemEventHandler = MagicMock
    wd_stub.observers = wd_observers
    wd_stub.events = wd_events
    sys.modules["watchdog"] = wd_stub
    sys.modules["watchdog.observers"] = wd_observers
    sys.modules["watchdog.events"] = wd_events
