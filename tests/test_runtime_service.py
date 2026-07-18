"""Unit tests for runtime_service.py — RRF hybrid search, filtering, process_media dispatch.

No AI models required. Semantic + keyword search are monkeypatched to return
canned results so RRF fusion logic is tested in isolation.
"""

import os

import pytest

from backend.search_and_index import runtime_service as rs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_search(monkeypatch):
    """Replace semantic_search, search_to_json, and _load_meta_by_paths with stubs."""
    sem_results = []
    kw_results = []

    def fake_sem(query, limit):
        return list(sem_results)

    def fake_kw(query):
        return list(kw_results)

    def fake_meta(file_paths):
        return {}

    monkeypatch.setattr(rs, "semantic_search", fake_sem)
    monkeypatch.setattr(rs, "search_to_json", fake_kw)
    monkeypatch.setattr(rs, "_load_meta_by_paths", fake_meta)
    return sem_results, kw_results


# ---------------------------------------------------------------------------
# _result_key
# ---------------------------------------------------------------------------

class TestResultKey:
    def test_same_item_same_key(self):
        a = {"file_path": "/v.mp4", "start": 1.0, "end": 2.0, "text": "hi"}
        b = {"file_path": "/v.mp4", "start": 1.0, "end": 2.0, "text": "hi"}
        assert rs._result_key(a) == rs._result_key(b)

    def test_different_path_different_key(self):
        a = {"file_path": "/a.mp4", "start": 0, "end": 1, "text": "x"}
        b = {"file_path": "/b.mp4", "start": 0, "end": 1, "text": "x"}
        assert rs._result_key(a) != rs._result_key(b)

    def test_different_text_different_key(self):
        a = {"file_path": "/v.mp4", "start": 0, "end": 1, "text": "a"}
        b = {"file_path": "/v.mp4", "start": 0, "end": 1, "text": "b"}
        assert rs._result_key(a) != rs._result_key(b)

    def test_whitespace_trimmed(self):
        a = {"file_path": "/v.mp4", "start": 0, "end": 1, "text": "  hi  "}
        b = {"file_path": "/v.mp4", "start": 0, "end": 1, "text": "hi"}
        assert rs._result_key(a) == rs._result_key(b)

    def test_absolute_path_normalization(self):
        a = {"file_path": "relative/path.mp4", "start": 0, "end": 1, "text": "x"}
        b = {"file_path": os.path.abspath("relative/path.mp4"), "start": 0, "end": 1, "text": "x"}
        assert rs._result_key(a) == rs._result_key(b)


# ---------------------------------------------------------------------------
# _rrf_add
# ---------------------------------------------------------------------------

class TestRRFAdd:
    def test_single_item_gets_score(self):
        scores = {}
        ranks = {}
        items = [{"file_path": "/v.mp4", "start": 0, "end": 1, "text": "a"}]
        rs._rrf_add(scores, ranks, items, "semantic", k=60)
        assert len(scores) == 1
        key = list(scores.keys())[0]
        assert scores[key] == pytest.approx(1.0 / (60 + 1))

    def test_higher_rank_lower_score(self):
        scores = {}
        ranks = {}
        items = [
            {"file_path": f"/v{i}.mp4", "start": 0, "end": 1, "text": str(i)}
            for i in range(3)
        ]
        rs._rrf_add(scores, ranks, items, "semantic", k=60)
        score_vals = list(scores.values())
        assert score_vals[0] > score_vals[1] > score_vals[2]

    def test_two_sources_merge_scores(self):
        scores = {}
        ranks = {}
        item = {"file_path": "/v.mp4", "start": 0, "end": 1, "text": "a"}
        rs._rrf_add(scores, ranks, [item], "semantic", k=60)
        rs._rrf_add(scores, ranks, [item], "keyword", k=60)
        key = list(scores.keys())[0]
        assert scores[key] == pytest.approx(2 * (1.0 / 61))
        assert "semantic" in ranks[key]
        assert "keyword" in ranks[key]


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_full_datetime(self):
        d = rs._parse_date("2026-07-18 12:00:00")
        assert d is not None
        assert d.year == 2026
        assert d.month == 7
        assert d.day == 18

    def test_date_only(self):
        d = rs._parse_date("2026-07-18")
        assert d is not None
        assert d.day == 18

    def test_iso_format(self):
        d = rs._parse_date("2026-07-18T12:00:00")
        assert d is not None
        assert d.hour == 12

    def test_invalid_returns_none(self):
        assert rs._parse_date("not a date") is None

    def test_empty_returns_none(self):
        assert rs._parse_date("") is None
        assert rs._parse_date(None) is None


# ---------------------------------------------------------------------------
# _passes_filters
# ---------------------------------------------------------------------------

class TestPassesFilters:
    base_item = {
        "score": 0.5,
        "source_type": "video",
        "file_path": "/media/videos/v.mp4",
        "added_at": "2026-07-18 12:00:00",
    }

    def test_no_filters_passes(self):
        assert rs._passes_filters(self.base_item, None, None, None, None, 0.0)

    def test_min_score_filter(self):
        assert rs._passes_filters(self.base_item, None, None, None, None, 0.3)
        assert not rs._passes_filters(self.base_item, None, None, None, None, 0.6)

    def test_source_type_filter(self):
        assert rs._passes_filters(self.base_item, {"video"}, None, None, None, 0.0)
        assert not rs._passes_filters(self.base_item, {"pdf"}, None, None, None, 0.0)

    def test_folder_filter(self):
        assert rs._passes_filters(self.base_item, None, ["/media"], None, None, 0.0)
        assert not rs._passes_filters(self.base_item, None, ["/other"], None, None, 0.0)

    def test_date_from_filter(self):
        dt = rs._parse_date("2026-01-01")
        assert rs._passes_filters(self.base_item, None, None, dt, None, 0.0)
        dt_future = rs._parse_date("2027-01-01")
        assert not rs._passes_filters(self.base_item, None, None, dt_future, None, 0.0)


# ---------------------------------------------------------------------------
# hybrid_search_rrf
# ---------------------------------------------------------------------------

class TestHybridSearchRRF:
    def test_empty_results(self, stub_search):
        sem, kw = stub_search
        results = rs.hybrid_search_rrf("test")
        assert results == []

    def test_semantic_only(self, stub_search):
        sem, kw = stub_search
        sem.append({"file_path": "/v.mp4", "start": 0, "end": 1, "text": "test", "score": 0.9})
        results = rs.hybrid_search_rrf("test", limit=10)
        assert len(results) == 1
        assert "semantic" in results[0]["matched_by"]

    def test_keyword_only(self, stub_search):
        sem, kw = stub_search
        kw.append({"file_path": "/v.mp4", "start": 0, "end": 1, "text": "test", "score": 0.8})
        results = rs.hybrid_search_rrf("test", limit=10)
        assert len(results) == 1
        assert "keyword" in results[0]["matched_by"]

    def test_merged_result_from_both(self, stub_search):
        sem, kw = stub_search
        item = {"file_path": "/v.mp4", "start": 0, "end": 1, "text": "test", "score": 0.9}
        sem.append(item)
        kw.append(item)
        results = rs.hybrid_search_rrf("test", limit=10)
        assert len(results) == 1
        assert "semantic" in results[0]["matched_by"]
        assert "keyword" in results[0]["matched_by"]
        assert results[0]["semantic_rank"] == 1
        assert results[0]["keyword_rank"] == 1

    def test_limit_applied(self, stub_search):
        sem, kw = stub_search
        for i in range(10):
            sem.append({"file_path": f"/v{i}.mp4", "start": 0, "end": 1, "text": str(i), "score": 0.1 * i})
        results = rs.hybrid_search_rrf("test", limit=5)
        assert len(results) == 5

    def test_sorted_by_score_desc(self, stub_search):
        sem, kw = stub_search
        for i in range(5):
            sem.append({"file_path": f"/v{i}.mp4", "start": 0, "end": 1, "text": str(i), "score": 0.1})
        results = rs.hybrid_search_rrf("test", limit=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filter(self, stub_search):
        sem, kw = stub_search
        sem.append({"file_path": "/v.mp4", "start": 0, "end": 1, "text": "test", "score": 0.01})
        results = rs.hybrid_search_rrf("test", limit=10, min_score=0.1)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# process_media dispatch
# ---------------------------------------------------------------------------

class TestProcessMediaDispatch:
    def test_unsupported_extension_raises(self, tmp_path, monkeypatch):
        f = tmp_path / "file.xyz"
        f.write_bytes(b"x")
        # Monkeypatch should_process to return True
        monkeypatch.setattr(rs, "should_process", lambda p: (True, "fake_hash"))
        with pytest.raises(RuntimeError, match="Unsupported file type"):
            rs.process_media(str(f))

    def test_skips_unchanged_file(self, tmp_path, monkeypatch):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        monkeypatch.setattr(rs, "should_process", lambda p: (False, None))
        result = rs.process_media(str(f))
        assert result == "skipped"
