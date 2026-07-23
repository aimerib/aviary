"""Cleaning third-party corpora before they are considered for interleaving.

Nothing here renders training text — that stays the serializer's job (choke-point
rule). These modules only filter and normalize foreign data into a reviewable
JSONL, so a human can audit what survived before anything is adapted into a
`ConversationRecord`.
"""
