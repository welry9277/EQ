from __future__ import annotations

import json

from eq_generation.query_types import QueryType

# Few-shot blocks for full_caption (entire caption set as bullets)
_KEY_DOG_SET = """- a dog barking repeatedly in the background
- a dog barking over distant outdoor ambience
- repeated barking from a dog outside
- a dog keeps barking in the background
- outdoor audio with a dog barking again and again"""

_KEY_RAIN_SET = """- rain falling on a metal surface with distant thunder
- a storm with rainfall hitting metal and thunder far away
- rain pattering on metal while thunder rumbles in the distance
- steady rain on a metal surface with faint thunder
- rainfall striking metal with distant thunder sounds"""

_KEY_COFFEE_SET = """- coffee machine brewing espresso with steam hissing
- espresso machine producing coffee with steam and brewing noises
- sound of a coffee maker brewing espresso with steam"""

FULL_CAPTION_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Full_caption: A dog is barking repeatedly outside, accompanied by distant outdoor background ambience.

Caption Set:
{_KEY_RAIN_SET}
Full_caption: Steady rain is falling and pattering on a metal surface during a storm, while faint thunder rumbles in the distance.

Caption Set:
{_KEY_COFFEE_SET}
Full_caption: An espresso machine is brewing coffee, producing distinct brewing noises and hissing steam.

Caption Set:
{{caption}}
Full_caption:"""

# Few-shot prefixes for key_phrase / statement / question / command / indirect:
# one reference caption per example. The real caption is appended via json.dumps (safe quoting).
_KEY_PHRASE_FEWSHOT = """Caption: "a dog barking repeatedly in the background"
Query: Repeated dog barking over distant outdoor ambience

Caption: "rain falling on a metal surface with distant thunder"
Query: Steady rain pattering on metal with distant thunder

Caption: "coffee machine brewing espresso with steam hissing"
Query: Espresso machine brewing with steam hissing

"""

_STATEMENT_FEWSHOT = """Caption: "a dog barking repeatedly in the background"
Statement: The audio captures a dog barking continuously in an outdoor environment.

Caption: "rain falling on a metal surface with distant thunder"
Statement: There is the sound of steady precipitation striking a metallic surface with thunder echoing far away.

Caption: "coffee machine brewing espresso with steam hissing"
Statement: An espresso maker is operating, producing brewing sounds and hissing steam.

"""

_COMMAND_FEWSHOT = """Caption: "a dog barking repeatedly in the background"
Command: Find audio of a dog barking repeatedly with outdoor background ambience.

Caption: "rain falling on a metal surface with distant thunder"
Command: Search for a recording of steady rain hitting a metal surface with distant thunder rumbles.

Caption: "coffee machine brewing espresso with steam hissing"
Command: Retrieve the sound of a coffee machine brewing espresso and hissing steam.

"""

_QUESTION_FEWSHOT = """Caption: "a dog barking repeatedly in the background"
Question: Does this audio contain the sound of a dog barking repeatedly outside?

Caption: "rain falling on a metal surface with distant thunder"
Question: Is there a recording of steady rain falling on metal accompanied by distant thunder?

Caption: "coffee machine brewing espresso with steam hissing"
Question: Are there sounds of an espresso machine brewing and emitting hissing steam?

"""

_INDIRECT_FEWSHOT = """Caption: "a dog barking repeatedly in the background"
Polite: I would appreciate it if you could find a clip where a dog barks repeatedly in an outdoor setting.

Caption: "rain falling on a metal surface with distant thunder"
Polite: Could you please locate an audio file featuring rain pattering on metal and distant thunder?

Caption: "coffee machine brewing espresso with steam hissing"
Polite: I was wondering if you might have the sound of an espresso maker brewing with steam hissing.

"""

SINGLE_REFERENCE_SUFFIX = {
    QueryType.KEY_PHRASE: "Query:",
    QueryType.STATEMENT: "Statement:",
    QueryType.QUESTION: "Question:",
    QueryType.COMMAND: "Command:",
    QueryType.INDIRECT: "Polite:",
}

SINGLE_REFERENCE_FEWSHOT = {
    QueryType.KEY_PHRASE: _KEY_PHRASE_FEWSHOT,
    QueryType.STATEMENT: _STATEMENT_FEWSHOT,
    QueryType.QUESTION: _QUESTION_FEWSHOT,
    QueryType.COMMAND: _COMMAND_FEWSHOT,
    QueryType.INDIRECT: _INDIRECT_FEWSHOT,
}

EQ_GPT_SYSTEM_PROMPT = """You are an expert data engineer tasked with generating faithful audio retrieval queries for an evaluation benchmark (EQ: Extended Query).
Return only the final query text, with no explanation, labels, quotes, or extra formatting.

Strict Rules:
1. Stay strictly faithful to the source. When given a single Caption line, use only that text; when given a Caption Set (multiple bullets), synthesize across all lines without inventing unsupported details.
2. Formats MUST match the target style:
- key_phrase: A concise noun phrase containing core audio elements. DO NOT use full sentences or periods.
- statement: An objective, declarative sentence describing the audio factually.
- question: A direct interrogative sentence asking if the audio elements exist.
- command: A direct imperative sentence ordering the retrieval system to find the audio.
- indirect: A polite, conversational request (e.g., "Could you please...", "I would appreciate...").
- full_caption: A single, highly detailed declarative sentence that perfectly synthesizes all unique details present across the entire Caption Set.
"""


def format_prompt(query_type: QueryType, caption: str) -> str:
    if query_type == QueryType.FULL_CAPTION:
        return FULL_CAPTION_PROMPT_TEMPLATE.format(caption=caption)
    prefix = SINGLE_REFERENCE_FEWSHOT[query_type]
    suffix_label = SINGLE_REFERENCE_SUFFIX[query_type]
    line = f"Caption: {json.dumps(caption)}\n{suffix_label}"
    return prefix + line


def get_system_prompt(query_type: QueryType, backend: str = "gpt") -> str:
    del query_type, backend
    return EQ_GPT_SYSTEM_PROMPT
