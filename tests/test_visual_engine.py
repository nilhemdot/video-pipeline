"""Unit tests for visual_engine.py — CLIP frame extraction + indexing.

cv2, PIL, torch, sentence_transformers, and lancedb are mocked via conftest.
No GPU or model download required.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.search_and_index import visual_engine


class TestClearVisualForMedia:
    @patch("backend.search_and_index.visual_engine.lancedb")
    def test_deletes_from_visual_moments_table(
        self, mock_lancedb, tmp_path, monkeypatch
    ):
        """clear_visual_for_media deletes from visual_moments table + thumbnails."""
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()
        (thumb_dir / "1_frame1.jpg").write_bytes(b"thumb1")
        (thumb_dir / "1_frame2.jpg").write_bytes(b"thumb2")
        (thumb_dir / "2_frame1.jpg").write_bytes(b"thumb2_other")
        monkeypatch.setattr(visual_engine, "THUMBNAIL_PATH", str(thumb_dir))

        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["visual_moments"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        visual_engine.clear_visual_for_media(media_id=1)

        mock_table.delete.assert_called_once_with("media_id = 1")
        # Thumbnails for media_id=1 deleted, media_id=2 kept
        assert not (thumb_dir / "1_frame1.jpg").exists()
        assert not (thumb_dir / "1_frame2.jpg").exists()
        assert (thumb_dir / "2_frame1.jpg").exists()

    @patch("backend.search_and_index.visual_engine.lancedb")
    def test_no_tables_no_error(self, mock_lancedb, tmp_path, monkeypatch):
        monkeypatch.setattr(
            visual_engine, "THUMBNAIL_PATH", str(tmp_path / "no_thumbs")
        )
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db

        # Should not raise even with no tables
        visual_engine.clear_visual_for_media(media_id=99)

    @patch("backend.search_and_index.visual_engine.lancedb")
    def test_no_thumbnail_dir(self, mock_lancedb, tmp_path, monkeypatch):
        # THUMBNAIL_PATH doesn't exist
        monkeypatch.setattr(
            visual_engine, "THUMBNAIL_PATH", str(tmp_path / "nonexistent")
        )
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["visual_moments"]
        mock_lancedb.connect.return_value = mock_db

        # Should not raise
        visual_engine.clear_visual_for_media(media_id=5)


class TestGetVisualModel:
    def test_raises_if_model_missing(self, monkeypatch):
        monkeypatch.setattr(visual_engine, "_visual_model", None)
        monkeypatch.setattr(visual_engine, "MODEL_VISUAL_PATH", "/nonexistent/model")
        with pytest.raises(RuntimeError, match="Visual model not found"):
            visual_engine.get_visual_model()

    @patch("backend.search_and_index.visual_engine.SentenceTransformer")
    def test_lazy_loads_once(self, mock_st, monkeypatch):
        import tempfile

        monkeypatch.setattr(visual_engine, "_visual_model", None)
        monkeypatch.setattr(visual_engine, "MODEL_VISUAL_PATH", tempfile.mkdtemp())
        mock_st.return_value = MagicMock()
        m1 = visual_engine.get_visual_model()
        m2 = visual_engine.get_visual_model()
        assert m1 is m2
        assert mock_st.call_count == 1


class TestIndexVideoVisually:
    @patch("backend.search_and_index.visual_engine.lancedb")
    @patch("backend.search_and_index.visual_engine.get_visual_model")
    @patch("backend.search_and_index.visual_engine.clear_visual_for_media")
    def test_invalid_fps_returns_early(
        self, mock_clear, mock_get_model, mock_lancedb, tmp_path
    ):
        """Video with 0 FPS should return without indexing."""
        video_path = str(tmp_path / "video.mp4")
        open(video_path, "wb").write(b"fake")

        with patch("backend.search_and_index.visual_engine.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.get.return_value = 0.0  # FPS = 0
            mock_cv2.VideoCapture.return_value = mock_cap

            visual_engine.index_video_visually(video_path, media_id=1)

        mock_clear.assert_called_once()
        mock_cap.release.assert_called_once()
        # Should not have tried to read frames
        mock_cap.read.assert_not_called()

    @patch("backend.search_and_index.visual_engine.lancedb")
    @patch("backend.search_and_index.visual_engine.get_visual_model")
    @patch("backend.search_and_index.visual_engine.clear_visual_for_media")
    def test_extracts_frames_at_interval(
        self, mock_clear, mock_get_model, mock_lancedb, tmp_path, monkeypatch
    ):
        """Should extract one frame every INTERVAL_SECONDS (2s)."""
        video_path = str(tmp_path / "video.mp4")
        open(video_path, "wb").write(b"fake")
        monkeypatch.setattr(visual_engine, "THUMBNAIL_PATH", str(tmp_path / "thumbs"))

        # Mock CLIP model
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        mock_get_model.return_value = mock_model

        with (
            patch("backend.search_and_index.visual_engine.cv2") as mock_cv2,
            patch("backend.search_and_index.visual_engine.Image") as mock_pil,
        ):
            mock_cap = MagicMock()
            mock_cap.get.side_effect = [30.0, True]  # FPS=30, first read=True
            # Simulate: first read succeeds, then EOF
            mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
            mock_cv2.VideoCapture.return_value = mock_cap

            # Mock PIL Image.fromarray
            mock_img = MagicMock()
            mock_pil.fromarray.return_value = mock_img

            visual_engine.index_video_visually(video_path, media_id=1)

        # At least one frame should have been read
        assert mock_cap.read.call_count >= 2  # read until False

    @patch("backend.search_and_index.visual_engine.lancedb")
    @patch("backend.search_and_index.visual_engine.get_visual_model")
    @patch("backend.search_and_index.visual_engine.clear_visual_for_media")
    def test_creates_thumbnail_dir(
        self, mock_clear, mock_get_model, mock_lancedb, tmp_path, monkeypatch
    ):
        video_path = str(tmp_path / "video.mp4")
        open(video_path, "wb").write(b"fake")
        thumb_path = tmp_path / "thumbs"
        monkeypatch.setattr(visual_engine, "THUMBNAIL_PATH", str(thumb_path))

        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [[0.1]]
        mock_get_model.return_value = mock_model

        with (
            patch("backend.search_and_index.visual_engine.cv2") as mock_cv2,
            patch("backend.search_and_index.visual_engine.Image"),
        ):
            mock_cap = MagicMock()
            mock_cap.get.side_effect = [30.0]
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap

            visual_engine.index_video_visually(video_path, media_id=1)

        # Thumbnail directory should be created
        assert thumb_path.exists()

    @patch("backend.search_and_index.visual_engine.lancedb")
    @patch("backend.search_and_index.visual_engine.get_visual_model")
    @patch("backend.search_and_index.visual_engine.clear_visual_for_media")
    def test_clears_existing_visual_data_first(
        self, mock_clear, mock_get_model, mock_lancedb, tmp_path
    ):
        """Should call clear_visual_for_media before indexing new frames."""
        video_path = str(tmp_path / "video.mp4")
        open(video_path, "wb").write(b"fake")

        with patch("backend.search_and_index.visual_engine.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.get.return_value = 0.0
            mock_cv2.VideoCapture.return_value = mock_cap

            visual_engine.index_video_visually(video_path, media_id=42)

        mock_clear.assert_called_once_with(42, visual_engine.VECTOR_DB_PATH)

    @patch("backend.search_and_index.visual_engine.lancedb")
    @patch("backend.search_and_index.visual_engine.get_visual_model")
    @patch("backend.search_and_index.visual_engine.clear_visual_for_media")
    def test_releases_video_capture(
        self, mock_clear, mock_get_model, mock_lancedb, tmp_path
    ):
        """Should always release the cv2 VideoCapture, even on early return."""
        video_path = str(tmp_path / "video.mp4")
        open(video_path, "wb").write(b"fake")

        with patch("backend.search_and_index.visual_engine.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.get.return_value = 0.0
            mock_cv2.VideoCapture.return_value = mock_cap

            visual_engine.index_video_visually(video_path, media_id=1)

        mock_cap.release.assert_called_once()


class TestConstants:
    def test_interval_seconds(self):
        assert visual_engine.INTERVAL_SECONDS == 2

    def test_batch_size(self):
        assert visual_engine.BATCH_SIZE == 50

    def test_thumbnail_max_size(self):
        assert visual_engine.THUMBNAIL_MAX_SIZE == (320, 320)

    def test_thumbnail_quality(self):
        assert visual_engine.THUMBNAIL_QUALITY == 80
