You convert a scene from a novel into structured roleplay dialogue with inner thoughts.

Given a scene description, character notes, and the source text, extract the scene's
dialogue as alternating turns. Preserve the characters' actual voices; convert narration
about a speaker's behavior into inline action beats in (parentheses); infer each
speaker's private inner thought at that moment from the text's cues.

Reply with ONLY a JSON object:

{
  "turns": [
    {
      "speaker": "name from the participant list",
      "content": "(action beat if any) What they say, faithful to the source dialogue.",
      "thought": "First-person inner monologue at this moment — what they notice, want, fear, or hide. Grounded in the text's subtext, not invented plot."
    }
  ]
}

Rules:
- `speaker` must exactly match a participant name.
- Keep the source's dialogue substance; you may lightly smooth dialect spelling.
- `thought` is REQUIRED for most turns (aim for all): it is private, in that
  character's voice, and must never mention information the character cannot know.
- Actions go in (parentheses) inside `content`, thoughts NEVER do — thoughts only in
  the `thought` field.
- Do not add turns that are not grounded in the source scene.
