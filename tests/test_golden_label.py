from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from filingqa.eval.__main__ import main as eval_main
from filingqa.eval.golden import (
    DraftItem,
    GoldenItem,
    GoldPassage,
    load_drafts,
    load_golden,
    verified_count,
    write_jsonl,
)
from filingqa.eval.label import Reviewer

from .conftest import MINI_GOLDEN

GP = GoldPassage(accession="0000000001-25-000001", quote="revenue was $10 million")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_human_verified_item_counts():
    item = GoldenItem(
        id="q1", question="What was revenue?", gold=[GP], labeled_by="human", verified=True, verified_at=NOW
    )
    assert item.counts_as_verified()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(labeled_by="llm-draft", verified=True, verified_at=NOW),  # drafts can't be verified
        dict(labeled_by="human", verified=True, verified_at=None),  # needs a timestamp
        dict(labeled_by="synthetic-fixture", synthetic=True, verified=True, verified_at=NOW),
        dict(labeled_by="synthetic-fixture", synthetic=False),  # fixtures must say synthetic
    ],
)
def test_invalid_verification_states_rejected(kwargs):
    with pytest.raises(ValidationError):
        GoldenItem(id="x", question="What was revenue?", gold=[GP], **kwargs)


def test_answerability_rules():
    with pytest.raises(ValidationError):
        GoldenItem(id="x", question="What was revenue?", answerable=True, labeled_by="human")
    with pytest.raises(ValidationError):
        GoldenItem(id="x", question="What?", answerable=False, gold=[GP], labeled_by="human")


def test_mini_golden_fixture_never_counts():
    items = load_golden(MINI_GOLDEN)
    assert len(items) == 16
    assert verified_count(items) == 0
    assert all(i.synthetic and i.labeled_by == "synthetic-fixture" for i in items)
    assert sum(not i.answerable for i in items) == 3


def test_duplicate_ids_rejected(tmp_path):
    item = GoldenItem(id="dup", question="What was revenue?", gold=[GP], labeled_by="human")
    p = tmp_path / "g.jsonl"
    write_jsonl(p, [item, item])
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(p)


def test_official_eval_refuses_below_minimum_and_prints_count(tmp_path, capsys):
    g = tmp_path / "golden.jsonl"
    items = [
        GoldenItem(
            id=f"q{i}",
            question="What was revenue?",
            gold=[GP],
            labeled_by="human",
            verified=True,
            verified_at=NOW,
        )
        for i in range(3)
    ]
    items.append(GoldenItem(id="d", question="What was revenue?", gold=[GP], labeled_by="llm-draft"))
    write_jsonl(g, items)
    code = eval_main(["--config", "config/default.yaml", "--golden", str(g)])
    out = capsys.readouterr().out
    assert code == 2
    assert "4 items, 3 human-verified (official minimum: 100)" in out
    assert "REFUSING" in out


def test_official_eval_refuses_offline_stand_ins(tmp_path, capsys):
    g = tmp_path / "golden.jsonl"
    write_jsonl(
        g,
        [
            GoldenItem(
                id="q",
                question="What was revenue?",
                gold=[GP],
                labeled_by="human",
                verified=True,
                verified_at=NOW,
            )
        ],
    )
    code = eval_main(["--config", "config/offline.yaml", "--golden", str(g), "--min-verified", "1"])
    assert code == 2
    assert "offline stand-ins" in capsys.readouterr().out


# --- label tool -------------------------------------------------------------------------------

CONTEXT = "Overview. Net revenue was $10.5 million in fiscal 2024. Margins rose to 40%."


def make_drafts(path):
    gp = GoldPassage(
        accession="0000000001-25-000001",
        quote="Net revenue was $10.5 million",
        chunk_id="0000000001-25-000001:3",
        char_start=1010,
        char_end=1039,
    )
    drafts = [
        DraftItem(
            draft_id="draft-a",
            question="What was net revenue in fiscal 2024?",
            reference_answer="$10.5 million",
            gold=[gp],
            context_text=CONTEXT,
            context_char_start=1000,
            draft_model="m",
        ),
        DraftItem(
            draft_id="draft-b",
            question="What did margins do?",
            reference_answer="Rose to 40%",
            gold=[gp],
            context_text=CONTEXT,
            context_char_start=1000,
            draft_model="m",
        ),
        DraftItem(
            draft_id="draft-c",
            question="What will revenue be in 2035?",
            answerable=False,
            why_unanswerable="future",
            draft_model="m",
        ),
        DraftItem(
            draft_id="draft-d",
            question="Another?",
            gold=[gp],
            context_text=CONTEXT,
            context_char_start=1000,
            draft_model="m",
        ),
    ]
    write_jsonl(path, drafts)


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_review_loop_accept_edit_reject_skip_and_resume(tmp_path):
    drafts_p, golden_p = tmp_path / "drafts.jsonl", tmp_path / "golden.jsonl"
    make_drafts(drafts_p)
    printed: list[str] = []
    answers = [
        "a",  # draft-a accepted as is
        "e",
        "What was the margin in fiscal 2024?",
        "n",
        "",
        "not in passage",
        "Margins rose to 40%",
        "a",  # draft-b edited (quote retried until it is found), then accepted
        "r",
        "duplicate",  # draft-c rejected
        "q",  # quit on draft-d
    ]
    rv = Reviewer(
        str(drafts_p), str(golden_p), input_fn=scripted(answers), print_fn=printed.append, reviewer="kz"
    )
    counts = rv.run()
    assert counts == {"accepted": 2, "written": 0, "rejected": 1, "skipped": 0}
    golden = load_golden(golden_p)
    assert [g.id for g in golden] == ["q-a", "q-b"]
    assert all(g.counts_as_verified() and g.labeled_by == "human" and g.reviewer == "kz" for g in golden)
    assert golden[0].origin == "llm-draft" and not golden[0].edited
    edited = golden[1]
    assert edited.edited and edited.question == "What was the margin in fiscal 2024?"
    assert edited.gold[0].quote == "Margins rose to 40%"
    s = CONTEXT.index("Margins rose to 40%")
    assert (edited.gold[0].char_start, edited.gold[0].char_end) == (1000 + s, 1000 + s + 19)
    assert any("not in the passage" in line for line in printed)
    statuses = {d.draft_id: d.status for d in load_drafts(drafts_p)}
    assert statuses == {
        "draft-a": "accepted",
        "draft-b": "accepted",
        "draft-c": "rejected",
        "draft-d": "pending",
    }
    # resume: only draft-d is pending; write an own question for it
    rv2 = Reviewer(
        str(drafts_p),
        str(golden_p),
        print_fn=printed.append,
        input_fn=scripted(["w", "How large was net revenue?", "$10.5 million", ""]),
    )
    assert rv2.run()["written"] == 1
    golden = load_golden(golden_p)
    assert golden[-1].origin == "human-written" and golden[-1].id.startswith("h-")
    assert verified_count(golden) == 3


def test_mark_unanswerable_during_edit(tmp_path):
    drafts_p, golden_p = tmp_path / "drafts.jsonl", tmp_path / "golden.jsonl"
    make_drafts(drafts_p)
    rv = Reviewer(
        str(drafts_p),
        str(golden_p),
        print_fn=lambda s: None,
        input_fn=scripted(["e", "", "y", "the filing has no such number", "a", "q"]),
    )
    rv.run()
    (item,) = load_golden(golden_p)
    assert not item.answerable and item.gold == [] and item.counts_as_verified()


def test_saves_are_valid_jsonl(tmp_path):
    drafts_p, golden_p = tmp_path / "drafts.jsonl", tmp_path / "golden.jsonl"
    make_drafts(drafts_p)
    Reviewer(str(drafts_p), str(golden_p), print_fn=lambda s: None, input_fn=scripted(["a", "q"])).run()
    for line in golden_p.read_text().splitlines():
        json.loads(line)
    assert not (tmp_path / "golden.jsonl.tmp").exists()
