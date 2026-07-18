"""Unit tests for api_models.py — Pydantic model validation."""

import pytest
from pydantic import ValidationError

from backend.search_and_index.api_models import (
    HybridSearchRequest,
    HybridResultItem,
    JobItem,
    EnvelopeSuccess,
    EnvelopeError,
    ErrorBody,
)


class TestHybridSearchRequest:
    def test_valid_defaults(self):
        req = HybridSearchRequest(query="test")
        assert req.query == "test"
        assert req.limit == 20
        assert req.semantic_limit == 40
        assert req.keyword_limit == 40
        assert req.k == 60
        assert req.min_score == 0.0

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="")

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="x", limit=0)
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="x", limit=201)

    def test_k_bounds(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="x", k=0)
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="x", k=201)

    def test_min_score_negative_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="x", min_score=-0.1)

    def test_optional_fields_default_none(self):
        req = HybridSearchRequest(query="x")
        assert req.source_types is None
        assert req.folders is None
        assert req.date_from is None
        assert req.date_to is None

    def test_custom_values_accepted(self):
        req = HybridSearchRequest(
            query="test",
            limit=50,
            k=100,
            source_types=["video", "pdf"],
            folders=["/media"],
            min_score=0.5,
        )
        assert req.limit == 50
        assert req.k == 100
        assert len(req.source_types) == 2
        assert req.min_score == 0.5


class TestHybridResultItem:
    def test_minimal_valid(self):
        item = HybridResultItem(file_path="/v.mp4", score=0.5, matched_by=["semantic"])
        assert item.file_path == "/v.mp4"
        assert item.score == 0.5
        assert item.matched_by == ["semantic"]

    def test_optional_fields_default_none(self):
        item = HybridResultItem(file_path="/v.mp4", score=0.0, matched_by=[])
        assert item.file_name is None
        assert item.start is None
        assert item.end is None
        assert item.text is None

    def test_matched_by_required(self):
        with pytest.raises(ValidationError):
            HybridResultItem(file_path="/v.mp4", score=0.5)

    def test_score_required(self):
        with pytest.raises(ValidationError):
            HybridResultItem(file_path="/v.mp4", matched_by=[])


class TestJobItem:
    def test_valid(self):
        job = JobItem(
            id=1,
            file_path="/v.mp4",
            source_type="video",
            status="queued",
            stage="pending",
            progress=0.0,
            retries=0,
            max_retries=3,
            error_message=None,
            created_at="2026-01-01 00:00:00",
            updated_at="2026-01-01 00:00:00",
        )
        assert job.id == 1
        assert job.max_retries == 3

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            JobItem(id=1, file_path="/v.mp4")  # missing required fields


class TestEnvelopeModels:
    def test_success_envelope(self):
        env = EnvelopeSuccess(data={"count": 5})
        assert env.ok is True
        assert env.data["count"] == 5

    def test_error_envelope(self):
        env = EnvelopeError(error=ErrorBody(code="NOT_FOUND", message="not found"))
        assert env.ok is False
        assert env.error.code == "NOT_FOUND"
        assert env.error.message == "not found"
