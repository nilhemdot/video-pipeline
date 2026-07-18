"""Unit tests for aural_engine.py — audio extraction + transcription pipeline.

Whisper model is mocked (no GPU / model download needed).
"""

import os
import json
from unittest.mock import patch, MagicMock

from backend.search_and_index import aural_engine


class TestExtractAudio:
    @patch("backend.search_and_index.aural_engine.ffmpeg")
    def test_calls_ffmpeg_correctly(self, mock_ffmpeg, tmp_path):
        mock_input = MagicMock()
        mock_output = MagicMock()
        mock_ffmpeg.input.return_value = mock_input
        mock_input.output.return_value = mock_output
        mock_output.overwrite_output.return_value = mock_output
        mock_output.run.return_value = (b"", b"")

        input_path = str(tmp_path / "video.mp4")
        open(input_path, "wb").write(b"fake")

        result = aural_engine.extract_audio(input_path)
        assert result is not None
        mock_ffmpeg.input.assert_called_once_with(input_path)

    @patch("backend.search_and_index.aural_engine.ffmpeg")
    def test_returns_none_on_ffmpeg_error(self, mock_ffmpeg, tmp_path):
        # ffmpeg.Error must be a real exception class for the except clause to work
        class FakeFFmpegError(Exception):
            def __init__(self, *args, stderr=b"error", **kwargs):
                super().__init__(*args)
                self.stderr = stderr

        mock_ffmpeg.Error = FakeFFmpegError
        mock_input = MagicMock()
        mock_ffmpeg.input.return_value = mock_input
        mock_output = MagicMock()
        mock_input.output.return_value = mock_output
        mock_output.overwrite_output.return_value = mock_output
        mock_output.run.side_effect = FakeFFmpegError("error")
        result = aural_engine.extract_audio("/nonexistent.mp4")
        assert result is None


class TestGetWhisper:
    def test_lazy_loads_model(self, monkeypatch):
        monkeypatch.setattr(aural_engine, "_WHISPER_MODEL", None)
        mock_model = MagicMock()
        with patch(
            "backend.search_and_index.aural_engine.WhisperModel",
            return_value=mock_model,
        ):
            # Patch MODEL_WHISPER_PATH to nonexistent so it uses model name
            monkeypatch.setattr(aural_engine, "MODEL_WHISPER_PATH", "/nonexistent/path")
            result = aural_engine.get_whisper()
            assert result is mock_model
            # Second call reuses cached model
            assert aural_engine.get_whisper() is mock_model


class TestTranscribeAudio:
    @patch("backend.search_and_index.aural_engine.get_whisper")
    def test_returns_transcript_list(self, mock_get_whisper, tmp_path):
        # Create fake segments matching faster_whisper output
        segment1 = MagicMock()
        segment1.start = 0.0
        segment1.end = 2.5
        segment1.text = "hello world"
        segment2 = MagicMock()
        segment2.start = 2.5
        segment2.end = 5.0
        segment2.text = "testing transcription"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([segment1, segment2], MagicMock())
        mock_get_whisper.return_value = mock_model

        input_audio = str(tmp_path / "audio.wav")
        open(input_audio, "wb").write(b"fake")

        result = aural_engine.transcribe_audio(input_audio)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.5
        assert result[0]["text"] == "hello world"
        assert result[1]["text"] == "testing transcription"

    @patch("backend.search_and_index.aural_engine.get_whisper")
    def test_writes_json_output(self, mock_get_whisper, tmp_path):
        seg = MagicMock()
        seg.start = 1.0
        seg.end = 2.0
        seg.text = "test"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock())
        mock_get_whisper.return_value = mock_model

        out_path = str(tmp_path / "transcript.json")
        input_audio = str(tmp_path / "audio.wav")
        open(input_audio, "wb").write(b"fake")

        aural_engine.transcribe_audio(input_audio, output_path=out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["text"] == "test"

    @patch("backend.search_and_index.aural_engine.get_whisper")
    def test_empty_transcript(self, mock_get_whisper, tmp_path):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        mock_get_whisper.return_value = mock_model

        input_audio = str(tmp_path / "audio.wav")
        open(input_audio, "wb").write(b"fake")

        result = aural_engine.transcribe_audio(input_audio)
        assert result == []
