from __future__ import annotations

import json
from pathlib import Path

from aviary.io.jsonl import read_jsonl
from aviary.io.store import RunStore
from aviary.lanes.b_fiction.books import BookConfig, chunk_text, load_book_text, strip_gutenberg
from aviary.lanes.b_fiction.pipeline import LaneBConfig, run_lane_b
from aviary.render.serializer import ThoughtMode, render_conversation
from aviary.schema.records import ConversationRecord
from aviary.teacher.client import ChatRequest
from aviary.teacher.fake import FakeTeacherClient
from aviary.teacher.prompts import PromptSet

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "tests" / "fixtures" / "books" / "tiny.txt"

PROFILES_JSON = json.dumps(
    {
        "work_id": "",
        "characters": [
            {
                "name": "Agnes",
                "aliases": ["the replacement"],
                "description": "A newly certified lighthouse keeper from the mainland.",
                "personality": "Unmoved by condescension; competence-first.",
                "speech_style": "Clipped, declarative.",
                "relationships": {"Thomas": "her reluctant senior keeper"},
            },
            {
                "name": "Thomas",
                "aliases": ["the keeper"],
                "description": "Veteran keeper of eleven years.",
                "personality": "Territorial, dry, secretly fair.",
                "speech_style": "Gruff, procedural.",
                "relationships": {"Agnes": "unwanted new colleague"},
            },
        ],
    }
)

SCENES_JSON = json.dumps(
    {
        "scenes": [
            {
                "setting": "The keeper's cottage on the island, late afternoon.",
                "participants": ["Agnes", "Thomas"],
                "context": "A new keeper arrives from the mainland to a post whose last occupant quit after three weeks.",
            }
        ]
    }
)

DIALOGUE_JSON = json.dumps(
    {
        "turns": [
            {
                "speaker": "Thomas",
                "content": "(not looking up) You're the replacement. They send me a girl from the mainland with soft hands and a certificate.",
                "thought": "Another mainlander. Three weeks, if the gulls are kind.",
            },
            {
                "speaker": "Agnes",
                "content": "They sent you a keeper. The certificate is in the case. The hands are my own business.",
                "thought": "Don't rise to it. He wants a flinch.",
            },
            {
                "speaker": "Thomas",
                "content": "(looking up) The last one lasted three weeks.",
                "thought": "She didn't flinch. Interesting.",
            },
            {
                "speaker": "Agnes",
                "content": "The last one wasn't me.",
                "thought": "Hold his eye. This is the whole interview.",
            },
            {
                "speaker": "Thomas",
                "content": "No. Apparently not. Supper's at six. The lamp is trimmed at dusk, whatever else is happening. You'll learn the order of things.",
                "thought": "Give her the rules. If she keeps them, the rest can be forgiven.",
            },
            {
                "speaker": "Agnes",
                "content": "I already know the order of things. Lamp first. Everything else after.",
                "thought": "Lamp first. He needs to hear that I know it in my bones.",
            },
        ]
    }
)


def scripted(req: ChatRequest) -> str:
    if "extract character profiles" in req.system:
        return PROFILES_JSON
    if "self-contained dialogue scenes" in req.system:
        return SCENES_JSON
    if "structured roleplay dialogue" in req.system:
        return DIALOGUE_JSON
    raise AssertionError(f"unexpected system prompt: {req.system[:60]}")


def make_prompts() -> PromptSet:
    return PromptSet.load(
        {
            "laneb_profiles": REPO / "datagen" / "prompts" / "laneb_profiles.md",
            "laneb_scenes": REPO / "datagen" / "prompts" / "laneb_scenes.md",
            "laneb_dialogue": REPO / "datagen" / "prompts" / "laneb_dialogue.md",
        }
    )


def test_book_loading_strips_gutenberg():
    text = load_book_text(BOOK)
    assert "PROJECT GUTENBERG" not in text
    assert "keeper's cottage" in text


def test_chunking_is_paragraph_aligned():
    text = strip_gutenberg(BOOK.read_text())
    chunks = chunk_text(text, target_chars=500)
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)


def test_lane_b_end_to_end(tmp_path):
    store = RunStore("testrun", root=tmp_path)
    cfg = LaneBConfig(
        books=[BookConfig(work_id="lighthouse_accord", path=BOOK, holdout=False)],
        chunk_target_chars=100_000,
    )
    client = FakeTeacherClient(script=scripted)
    models = {
        "profiles": "deepseek-v4-flash-20260610",
        "scenes": "deepseek-v4-flash-20260610",
        "dialogue": "deepseek-v4-flash-20260610",
    }
    n = run_lane_b(cfg, store, client, make_prompts(), models)
    assert n == 1

    rec = next(read_jsonl(store.raw("b"), ConversationRecord))
    assert rec.provenance.lane == "b"
    assert rec.provenance.family == "lighthouse_accord"
    assert rec.speakers == ["Thomas", "Agnes"]
    assert all(m.thought for m in rec.messages)

    rendered = render_conversation(rec, ThoughtMode.WITH)
    assert "<think>" in rendered.text
    assert "Thomas: " in rendered.text
    assert rec.messages[0].thought in rendered.text

    # Contract v3: reasoning-off is the EMPTY think block, not the absence of one —
    # that is what `enable_thinking=false` primes at inference. What must disappear
    # is the thought text, not the framing.
    without = render_conversation(rec, ThoughtMode.WITHOUT)
    assert rec.messages[0].thought not in without.text
    assert "<think>\n\n</think>\n\n" in without.text
    assert without.text.count("<think>") == without.text.count("<think>\n\n</think>\n\n")

    # stage checkpoints exist -> rerun consumes them, no new teacher calls
    calls_before = len(client.requests)
    n2 = run_lane_b(cfg, store, client, make_prompts(), models)
    assert n2 == 1
    assert len(client.requests) == calls_before


def test_lane_b_runs_books_in_parallel(tmp_path):
    # Independent books run concurrently (max_workers>1) and still produce one record
    # per book with the right family; separate per-work checkpoints don't collide.
    store = RunStore("parrun", root=tmp_path)
    cfg = LaneBConfig(
        books=[
            BookConfig(work_id="book_a", path=BOOK, holdout=False),
            BookConfig(work_id="book_b", path=BOOK, holdout=False),
        ],
        chunk_target_chars=100_000,
    )
    models = {k: "deepseek-v4-flash-20260610" for k in ("profiles", "scenes", "dialogue")}
    n = run_lane_b(
        cfg, store, FakeTeacherClient(script=scripted), make_prompts(), models, max_workers=2
    )
    assert n == 2
    fams = {r.provenance.family for r in read_jsonl(store.raw("b"), ConversationRecord)}
    assert fams == {"book_a", "book_b"}


def test_alias_resolution_and_speaker_gate(tmp_path):
    bad_dialogue = json.loads(DIALOGUE_JSON)
    bad_dialogue["turns"][0]["speaker"] = "Mysterious Stranger"

    def bad_script(req: ChatRequest) -> str:
        if "structured roleplay dialogue" in req.system:
            return json.dumps(bad_dialogue)
        return scripted(req)

    store = RunStore("testrun2", root=tmp_path)
    cfg = LaneBConfig(
        books=[BookConfig(work_id="lighthouse_accord", path=BOOK)], chunk_target_chars=100_000
    )
    n = run_lane_b(
        cfg,
        store,
        FakeTeacherClient(script=bad_script),
        make_prompts(),
        {k: "deepseek-v4-flash-20260610" for k in ("profiles", "scenes", "dialogue")},
    )
    assert n == 0  # unresolvable speaker drops the scene, never bends attribution
