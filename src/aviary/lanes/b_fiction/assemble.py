"""ExtractedScene -> ConversationRecord. Pure; the system prompt is rendered from a
byte-stable template so cache discipline and the renderer see identical prefixes."""

from __future__ import annotations

from aviary.lanes.b_fiction.models import ExtractedScene, ProfileSet
from aviary.schema.records import (
    ConversationRecord,
    Message,
    Provenance,
    SourceRef,
    make_record_id,
)

SCENE_SYSTEM_TEMPLATE = (
    "Roleplay the following scene faithfully.\n\n"
    "## Characters\n{characters}\n\n"
    "## Scene\n{setting}\n\n{context}"
)


def scene_system_prompt(scene: ExtractedScene, profiles: ProfileSet) -> str:
    chars = "\n".join(
        f"{c.name}: {c.description} {c.personality}".strip()
        for name in scene.setup.participants
        if (c := profiles.resolve(name)) is not None
    )
    return SCENE_SYSTEM_TEMPLATE.format(
        characters=chars, setting=scene.setup.setting, context=scene.setup.context
    )


def assemble_record(
    scene: ExtractedScene,
    profiles: ProfileSet,
    run_id: str,
    holdout: bool,
    teacher_ids: dict[str, str],
    prompt_set_hash: str,
) -> ConversationRecord:
    source = SourceRef(
        kind="book_scene",
        detail={
            "work_id": scene.work_id,
            "chunk_idx": scene.setup.chunk_idx,
            "scene_idx": scene.setup.scene_idx,
        },
    )
    return ConversationRecord(
        system=scene_system_prompt(scene, profiles),
        messages=[
            Message(role="assistant", speaker=t.speaker, content=t.content, thought=t.thought)
            for t in scene.turns
        ],
        provenance=Provenance(
            record_id=make_record_id("b", run_id, source),
            lane="b",
            run_id=run_id,
            family=scene.work_id,
            holdout=holdout,
            teachers=teacher_ids,
            prompt_set_hash=prompt_set_hash,
            source=source,
        ),
    )
