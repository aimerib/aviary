"""LLM extraction stages: profiles, scenes, dialogue-with-thoughts.

Prompt templates come from the run's frozen PromptSet (datagen/prompts/laneb_*.md);
this module never hardcodes prompt prose, only wiring and structural validation.
"""

from __future__ import annotations

from aviary.lanes.b_fiction.books import Chunk
from aviary.lanes.b_fiction.models import (
    DialogueResult,
    ExtractedScene,
    ProfileSet,
    SceneList,
    SceneSetup,
)
from aviary.lanes.common import call_json
from aviary.teacher.client import ChatRequest, TeacherClient
from aviary.teacher.prompts import PromptSet


def _profiles_text(profiles: ProfileSet, names: list[str] | None = None) -> str:
    chars = profiles.characters
    if names is not None:
        chars = [c for c in chars if any(c.matches(n) for n in names)]
    return "\n".join(
        f"- {c.name} ({', '.join(c.aliases)}): {c.description} "
        f"Personality: {c.personality} Voice: {c.speech_style}"
        for c in chars
    )


def extract_profiles(
    work_id: str,
    sample_text: str,
    client: TeacherClient,
    prompts: PromptSet,
    model: str,
) -> ProfileSet:
    req = ChatRequest(
        model=model,
        system=prompts["laneb_profiles"],
        messages=[{"role": "user", "content": sample_text}],
        temperature=0.3,
        max_tokens=4096,
    )
    result = call_json(client, req, ProfileSet, lane="b")
    result.work_id = work_id
    return result


def find_scenes(
    chunk: Chunk,
    profiles: ProfileSet,
    client: TeacherClient,
    prompts: PromptSet,
    model: str,
) -> list[SceneSetup]:
    req = ChatRequest(
        model=model,
        system=prompts["laneb_scenes"],
        messages=[
            {
                "role": "user",
                "content": (
                    f"## Known characters\n{_profiles_text(profiles)}\n\n## Text\n{chunk.text}"
                ),
            }
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    scenes = call_json(client, req, SceneList, lane="b").scenes
    kept = []
    for i, scene in enumerate(scenes):
        resolved = [profiles.resolve(p) for p in scene.participants]
        if any(r is None for r in resolved) or len(scene.participants) < 2:
            continue  # unresolvable participant or monologue: not a usable scene
        scene.participants = [r.name for r in resolved if r is not None]
        scene.chunk_idx = chunk.idx
        scene.scene_idx = i
        kept.append(scene)
    return kept


def extract_dialogue(
    chunk: Chunk,
    scene: SceneSetup,
    profiles: ProfileSet,
    client: TeacherClient,
    prompts: PromptSet,
    model: str,
    min_turns: int = 6,
    min_thought_coverage: float = 0.5,
) -> ExtractedScene | None:
    req = ChatRequest(
        model=model,
        system=prompts["laneb_dialogue"],
        messages=[
            {
                "role": "user",
                "content": (
                    f"## Scene\nSetting: {scene.setting}\n"
                    f"Participants: {', '.join(scene.participants)}\n"
                    f"Context: {scene.context}\n\n"
                    f"## Character notes\n{_profiles_text(profiles, scene.participants)}\n\n"
                    f"## Source text\n{chunk.text}"
                ),
            }
        ],
        temperature=0.3,
        max_tokens=8192,
    )
    turns = call_json(client, req, DialogueResult, lane="b").turns

    resolved_turns = []
    for t in turns:
        profile = profiles.resolve(t.speaker)
        if profile is None or profile.name not in scene.participants:
            return None  # speaker outside the scene: extraction is unreliable, drop
        t.speaker = profile.name
        resolved_turns.append(t)

    speakers = {t.speaker for t in resolved_turns}
    with_thought = sum(1 for t in resolved_turns if t.thought)
    if (
        len(resolved_turns) < min_turns
        or len(speakers) < 2
        or with_thought / len(resolved_turns) < min_thought_coverage
    ):
        return None
    return ExtractedScene(work_id=profiles.work_id, setup=scene, turns=resolved_turns)
