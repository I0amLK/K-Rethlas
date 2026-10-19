"""Disposable index checks use synthetic vectors, never live retrieval evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from rethlas_core.embedding import DIMENSIONS
from rethlas_core.private_fs import digest
from rethlas_core.retrieval_index import (
    MAX_CHUNKS, RetrievalIndexError, chunk_revisions, lexical_rank,
    make_index, rank, tokenize, validate_index,
)


def revision(label, text):
    return {"entity_id": "entity/" + label, "revision_id": "revision/" + label,
            "body": text, "body_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def vector(axis=0, value=1):
    return [float(value) if index == axis else 0.0 for index in range(DIMENSIONS)]


def binding():
    return {"service_id": "1" * 64, "store_id": "2" * 64, "store_config_sha256": "3" * 64,
            "graph": {"head": "4" * 64, "generation": 1, "graph_sha256": "5" * 64}}


def rehash(index):
    index["index_sha256"] = digest({key: value for key, value in index.items() if key != "index_sha256"})
    return index


class RetrievalIndexTests(unittest.TestCase):
    def test_utf8_chunks_preserve_exact_full_markdown_and_offsets(self):
        text = "# 一致连续\r\n∀ε>0，存在δ>0。\n🙂\n"
        item = revision("中文", text)
        chunks = chunk_revisions([item], chunk_bytes=7)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), text)
        position = 0
        for chunk in chunks:
            self.assertEqual(chunk["byte_start"], position)
            raw = text.encode()[chunk["byte_start"]:chunk["byte_end"]]
            self.assertEqual(raw.decode(), chunk["text"])
            self.assertLessEqual(len(raw), 7)
            position = chunk["byte_end"]
        self.assertEqual(position, len(text.encode()))

    def test_short_body_remains_exact_for_real_cache_replay(self):
        text = "A fixed short text.\n"
        chunks = chunk_revisions([revision("one", text)])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(chunks[0]["byte_start"], 0)
        self.assertEqual(chunks[0]["byte_end"], len(text.encode()))

    def test_chunking_is_deterministic_and_does_not_merge_same_text_identities(self):
        items = [revision("b", "same"), revision("a", "same")]
        first = chunk_revisions(items)
        self.assertEqual(first, chunk_revisions(list(reversed(items))))
        self.assertEqual(len(first), 2)
        self.assertNotEqual(first[0]["chunk_id"], first[1]["chunk_id"])
        self.assertEqual(first[0]["body_sha256"], first[1]["body_sha256"])
        self.assertEqual(items[0]["body"], "same")

    def test_invalid_revision_hash_utf8_duplicate_and_oversize_rejected(self):
        fixed = revision("one", "fixed")
        for values in ([{**fixed, "body_sha256": "0" * 64}], [fixed, fixed],
                       [{**fixed, "body": "\ud800"}], [revision("one", "x" * 131073)],
                       [{**fixed, "revision_id": ""}]):
            with self.subTest(count=len(values)):
                with self.assertRaises(RetrievalIndexError):
                    chunk_revisions(values)
        for limit in (True, 0, 3, 32769):
            with self.assertRaises(RetrievalIndexError):
                chunk_revisions([fixed], chunk_bytes=limit)

    def test_chunk_capacity_is_rejected_not_silently_truncated(self):
        with self.assertRaises(RetrievalIndexError):
            chunk_revisions([revision("one", "x" * (MAX_CHUNKS * 4 + 1))], chunk_bytes=4)

    def test_make_validate_and_rebuild_are_identical_and_do_not_mutate_inputs(self):
        chunks = chunk_revisions([revision("one", "fixed")])
        vectors = [vector()]
        before_chunks, before_vectors = deepcopy(chunks), deepcopy(vectors)
        first = make_index(chunks, vectors, binding=binding(), mode="bge_cache_only")
        rebuilt = make_index(chunks, vectors, binding=binding(), mode="bge_cache_only")
        self.assertEqual(first, rebuilt)
        self.assertEqual(validate_index(first, binding=binding()), first)
        self.assertEqual(chunks, before_chunks)
        self.assertEqual(vectors, before_vectors)
        vectors[0][0] = 9
        self.assertEqual(first["chunks"][0]["vector"][0], 1)

    def test_lexical_projection_has_no_fake_bge_vectors(self):
        chunks = chunk_revisions([revision("one", "一致连续 uniform continuity")])
        index = make_index(chunks, None, binding=binding(), mode="lexical")
        self.assertIsNone(index["model"])
        self.assertEqual(index["dimensions"], 0)
        self.assertNotIn("vector", index["chunks"][0])
        self.assertTrue(lexical_rank(index["chunks"], "一致连续"))
        with self.assertRaises(RetrievalIndexError):
            rank(index, vector())
        with self.assertRaises(RetrievalIndexError):
            make_index(chunks, [vector()], binding=binding(), mode="lexical")

    def test_exact_project_config_and_graph_binding_are_required(self):
        index = make_index(chunk_revisions([revision("one", "fixed")]), [vector()],
                           binding=binding(), mode="bge_live")
        for path in ("service_id", "store_id", "store_config_sha256"):
            wrong = binding()
            wrong[path] = "0" * 64
            with self.assertRaises(RetrievalIndexError):
                validate_index(index, binding=wrong)
        for path in ("head", "graph_sha256", "generation"):
            wrong = binding()
            wrong["graph"][path] = 2 if path == "generation" else "0" * 64
            with self.assertRaises(RetrievalIndexError):
                validate_index(index, binding=wrong)
        wrong = binding()
        wrong["graph"]["generation"] = True
        with self.assertRaises(RetrievalIndexError):
            make_index([], [], binding=wrong, mode="bge_live")

    def test_index_hash_and_model_tamper_are_rejected(self):
        index = make_index(chunk_revisions([revision("one", "fixed")]), [vector()],
                           binding=binding(), mode="bge_live")
        for field, value in (("version", "other"), ("model", "other"), ("dimensions", 1024.0),
                             ("input_version", "other"), ("mode", "other"), ("index_sha256", "0" * 64)):
            modified = deepcopy(index)
            modified[field] = value
            with self.subTest(field=field):
                with self.assertRaises(RetrievalIndexError):
                    validate_index(modified)

    def test_rehashed_chunk_hash_body_gap_or_vector_damage_is_rejected(self):
        original = make_index(chunk_revisions([revision("one", "fixed-body")], chunk_bytes=5),
                              [vector(), vector(1)], binding=binding(), mode="bge_live")
        for field, value in (("body_sha256", "0" * 64), ("byte_start", 1), ("text", "changed"),
                             ("text_sha256", "0" * 64), ("chunk_id", "0" * 64), ("vector", [0] * 1024)):
            index = deepcopy(original)
            index["chunks"][0][field] = value
            rehash(index)
            with self.subTest(field=field):
                with self.assertRaises(RetrievalIndexError):
                    validate_index(index)
        index = deepcopy(original)
        index["chunks"].pop()
        rehash(index)
        with self.assertRaises(RetrievalIndexError):
            validate_index(index)

    def test_canonical_chunk_order_and_duplicates_are_required(self):
        chunks = chunk_revisions([revision("a", "a"), revision("b", "b")])
        for value in (list(reversed(chunks)), [chunks[0], chunks[0]]):
            with self.assertRaises(RetrievalIndexError):
                make_index(value, [vector()] * len(value), binding=binding(), mode="bge_live")

    def test_cosine_ranking_and_stable_tie_order(self):
        chunks = chunk_revisions([revision("a", "same"), revision("b", "same"), revision("c", "other")])
        index = make_index(chunks, [vector(0), vector(0, 2), vector(1)], binding=binding(), mode="bge_live")
        hits = rank(index, vector(), top_k=64)
        self.assertEqual([item["revision_id"] for item in hits], ["revision/a", "revision/b", "revision/c"])
        self.assertEqual([item["score"] for item in hits], [1.0, 1.0, 0.0])
        self.assertFalse(any("vector" in hit for hit in hits))
        self.assertEqual(len(rank(index, vector(), top_k=1)), 1)

    def test_rank_can_return_all_chunks_for_revision_deduplication(self):
        chunks = chunk_revisions([revision("a", "a" * 300)], chunk_bytes=4)
        index = make_index(chunks, [vector()] * len(chunks), binding=binding(), mode="bge_cache_only")
        self.assertEqual(len(rank(index, vector(), top_k=1024)), len(chunks))
        self.assertGreater(len(chunks), 64)

    def test_extreme_finite_vectors_have_finite_cosine(self):
        chunks = chunk_revisions([revision("a", "fixed")])
        index = make_index(chunks, [vector(0, 1e300)], binding=binding(), mode="bge_live")
        hits = rank(index, vector(0, 1e-300))
        self.assertEqual(hits[0]["score"], 1.0)

    def test_invalid_query_vectors_and_result_limits_rejected(self):
        index = make_index([], [], binding=binding(), mode="bge_live")
        for limit in (0, True, 1025, 1.5):
            with self.assertRaises(RetrievalIndexError):
                rank(index, vector(), top_k=limit)
            with self.assertRaises(RetrievalIndexError):
                lexical_rank([], "fixed", top_k=limit)
        with self.assertRaises(RetrievalIndexError):
            rank(index, [0] * DIMENSIONS)

    def test_unicode_lexical_baseline_and_empty_matches_are_explicit(self):
        chunks = chunk_revisions([revision("a", "一致连续 uniform continuity"),
                                 revision("b", "pointwise convergence")])
        self.assertEqual(lexical_rank(chunks, "一致连续")[0]["revision_id"], "revision/a")
        self.assertEqual(lexical_rank(chunks, "CONTINUITY")[0]["revision_id"], "revision/a")
        self.assertEqual(lexical_rank(chunks, "unmatched"), [])
        self.assertIn("一致", tokenize("一致连续"))
        for query in ("", "  ", "\ud800", "x" * 32769):
            with self.assertRaises(RetrievalIndexError):
                lexical_rank(chunks, query)

    def test_empty_project_index_is_valid_but_returns_no_hits(self):
        index = make_index([], [], binding=binding(), mode="bge_live")
        self.assertEqual(rank(index, vector(), top_k=1024), [])
        lexical = make_index([], None, binding=binding(), mode="lexical")
        self.assertEqual(lexical_rank(lexical["chunks"], "fixed"), [])


if __name__ == "__main__":
    unittest.main()
