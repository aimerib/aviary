"""DPO pair emission from rollout siblings. Chosen = judge-passed, rejected = judge-failed
(both verifier-passed), margin-gated. Text is produced only by the serializer.

Scope (see CLAUDE.md): pairs form only within a `sibling_group`, which only lane A
sets — so DPO is effectively lane-A-only (lanes B/C have no rollout siblings of one
instance). Callers render with-thoughts pairs only (the default ThoughtMode.WITH)."""

from __future__ import annotations

from collections import defaultdict

from aviary.render.serializer import SERIALIZER_CONTRACT, ThoughtMode, render_conversation
from aviary.schema.records import ConversationRecord
from aviary.schema.results import RenderedPair


def _prompt_boundary(rec: ConversationRecord) -> int:
    """Index just past the first user message: the shared prompt across siblings."""
    for i, m in enumerate(rec.messages):
        if m.role == "user":
            return i + 1
    raise ValueError(f"record {rec.provenance.record_id} has no user message")


def _split_render(rec: ConversationRecord, mode: ThoughtMode) -> tuple[str, str]:
    boundary = _prompt_boundary(rec)
    full = render_conversation(rec, mode)
    prompt_only = render_conversation(
        rec.model_copy(update={"messages": rec.messages[:boundary]}), mode
    )
    return prompt_only.text, full.text[len(prompt_only.text) :]


def pair_siblings(
    kept: list[ConversationRecord],
    rejected: list[ConversationRecord],
    min_margin: float,
    mode: ThoughtMode = ThoughtMode.WITH,
) -> list[RenderedPair]:
    eligible = [
        r
        for r in rejected
        if r.gate_state.drop_reason == "judge" and r.gate_state.verified and r.gate_state.judge
    ]
    rejected_by_group: dict[str, list[ConversationRecord]] = defaultdict(list)
    for r in eligible:
        if r.provenance.sibling_group:
            rejected_by_group[r.provenance.sibling_group].append(r)

    pairs: list[RenderedPair] = []
    for chosen in kept:
        group = chosen.provenance.sibling_group
        if not group or not chosen.gate_state.judge:
            continue
        for reject in rejected_by_group.get(group, []):
            margin = chosen.gate_state.judge.overall - reject.gate_state.judge.overall
            if margin < min_margin:
                continue
            prompt, chosen_text = _split_render(chosen, mode)
            reject_prompt, rejected_text = _split_render(reject, mode)
            if reject_prompt != prompt:
                # siblings must share the byte-identical prompt; if not, skip
                continue
            pairs.append(
                RenderedPair(
                    prompt_text=prompt,
                    chosen_text=chosen_text,
                    rejected_text=rejected_text,
                    meta={
                        "sibling_group": group,
                        "chosen_id": chosen.provenance.record_id,
                        "rejected_id": reject.provenance.record_id,
                        "margin": f"{margin:.3f}",
                        "lane": chosen.provenance.lane,
                        "family": chosen.provenance.family,
                        "thought_mode": mode.value,
                        "contract": SERIALIZER_CONTRACT,
                    },
                )
            )
    return pairs
