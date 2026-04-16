from __future__ import annotations

import json
import random

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


_SINGLE_DOG = "repeated barking from a dog outside"
_SINGLE_RAIN = "rain pattering on metal while thunder rumbles in the distance"
_SINGLE_COFFEE = "espresso machine producing coffee with steam and brewing noises"

# Few-shot prefixes for key_phrase / statement / question / command / indirect:
# one reference caption per example. The real caption is appended via json.dumps (safe quoting).
_KEY_PHRASE_FEWSHOT = f"""Source Caption: {_SINGLE_DOG}
Query: Repeated dog barking outside

Source Caption: {_SINGLE_RAIN}
Query: Rain pattering on metal with distant thunder

Source Caption: {_SINGLE_COFFEE}
Query: Coffee maker brewing espresso with steam

Source Caption: {{caption}}
Query:"""

_STATEMENT_FEWSHOT = f"""Source Caption: {_SINGLE_DOG}
Statement: The audio captures a dog barking repeatedly outside.

Source Caption: {_SINGLE_RAIN}
Statement: There is the sound of rain pattering on metal while thunder rumbles in the distance.

Source Caption: {_SINGLE_COFFEE}
Statement: An espresso maker is brewing coffee and producing steam sounds.

Source Caption: {{caption}}
Statement:"""

_COMMAND_FEWSHOT = f"""Source Caption: {_SINGLE_DOG}
Command: Find audio of repeated barking from a dog outside.

Source Caption: {_SINGLE_RAIN}
Command: Search for a recording of rain pattering on metal while thunder rumbles in the distance.

Source Caption: {_SINGLE_COFFEE}
Command: Retrieve the sound of a coffee maker brewing espresso with steam.

Source Caption: {{caption}}
Command:"""

_QUESTION_FEWSHOT = f"""Source Caption: {_SINGLE_DOG}
Question: Does this audio contain the sound of repeated barking from a dog outside?

Source Caption: {_SINGLE_RAIN}
Question: Is there a recording of rain pattering on metal while thunder rumbles in the distance?

Source Caption: {_SINGLE_COFFEE}
Question: Can you hear a coffee maker brewing espresso with steam?

Source Caption: {{caption}}
Question:"""

_INDIRECT_FEWSHOT = f"""Source Caption: {_SINGLE_DOG}
Polite: I would appreciate it if you could find a clip with repeated barking from a dog outside.

Source Caption: {_SINGLE_RAIN}
Polite: Could you please locate an audio file featuring rain pattering on metal while thunder rumbles in the distance?

Source Caption: {_SINGLE_COFFEE}
Polite: I was wondering if you might have the sound of a coffee maker brewing espresso with steam.

Source Caption: {{caption}}
Polite:"""


SINGLE_REFERENCE_SUFFIX = {
    QueryType.KEY_PHRASE: "Query:",
    QueryType.STATEMENT: "Statement:",
    QueryType.QUESTION: "Question:",
    QueryType.COMMAND: "Command:",
    QueryType.INDIRECT: "Indirect:",
}

SINGLE_REFERENCE_FEWSHOT = {
    QueryType.KEY_PHRASE: _KEY_PHRASE_FEWSHOT,
    QueryType.STATEMENT: _STATEMENT_FEWSHOT,
    QueryType.QUESTION: _QUESTION_FEWSHOT,
    QueryType.COMMAND: _COMMAND_FEWSHOT,
    QueryType.INDIRECT: _INDIRECT_FEWSHOT,
}

SYSTEM_PROMPTS = {
    QueryType.KEY_PHRASE: """You are an expert at optimizing audio captions for search engines. Analyze the given caption and compress it into a tight Key Phrase according to the following rules.

[Constraints]

1. Strictly NO complete sentences (subject + verb). You must use a Noun Phrase format.

2. Remove unnecessary prepositions and conjunctions (e.g., 'while', 'from a').

3. Convert long background descriptions containing verbs into concise modifiers.

4. Maintain the texture and action of the sound by combining the sound's subject with a present participle (-ing) instead of a regular verb.

5. Avoid using 'while' or 'and' to connect background sounds/environments. Instead, compress and connect them using prepositions like 'with', 'on', 'in', or adjectives like 'distant'.""",

    QueryType.STATEMENT: """You are an expert at objectively describing audio scenarios. Analyze the given caption and write a full, descriptive Statement according to the following rules.

[Constraints]

1. Reconstruct the description into a grammatically perfect sentence complete with a subject and a verb.

2. Soften stiff noun phrases (e.g., 'repeated barking') into verb-modifying structures (e.g., 'barking repeatedly').

3. When describing background sounds or simultaneous events, do not compress them. Use conjunctions like 'while' and 'and' to naturally connect two clauses.

4. Since this is a complete sentence, it must end with a period (.).""",

    QueryType.QUESTION: """You are an expert at generating questions to verify the presence of audio content. Analyze the given caption and create a Yes/No Question according to the following rules.

[Constraints]
1. The sentence must be a Yes/No question starting with an auxiliary or 'be' verb, such as 'Does', 'Is', or 'Can'.
2. Combine this with cognition or possession verbs that ask if the audio 'contains' or if one can 'hear' the sounds.
3. Do not severely compress or break apart the original sound description; use it fully as the object part of the question.
4. Slightly polish any overly mechanical expressions caused by the question format into natural wording (e.g., 'producing' -> 'brewing').
5. Since this is a complete question, it must end with a question mark (?).""",

    QueryType.COMMAND: """You are an expert at crafting precise instructions for search systems or agents. Analyze the given caption and generate a direct Command according to the following rules.

[Constraints]
1. Do not use a subject. Start the sentence with a strong Action Verb to directly instruct the system.
2. Include only search/extraction directives aimed at the system, removing any unnecessary modifiers or predicates.
3. Do not overly compress or excessively describe the original text. Group the description into one large noun phrase and provide it as the direct object of the verb.
4. This is a complete command sentence, so it must end with a period (.).""",

    QueryType.INDIRECT: """You are an expert in highly polite and conversational communication. Create an Indirect/Polite Request asking to find the sounds described in the caption according to the following rules.

[Constraints]
1. Do not use direct verbs like 'Find'. Use indirect expressions that politely ask for the other party's willingness or assistance.
2. You must start the sentence with a polite request phrase such as 'I would appreciate it if you could...', 'Could you please...', or 'I was wondering if you might...'.
3. Structure the rest of the sentence appropriately as a statement or a question, depending on the grammatical rules of your chosen polite introduction.""",
    
    
    
    
    QueryType.FULL_CAPTION: """You are an expert at combining fragmented audio captions into one vivid, unified scene. Analyze multiple captions and write a Full-caption according to the following rules.

[Constraints]

1. Gather all unique details scattered across each caption (e.g., 'storm', 'steady', 'pattering') without leaving any out.

2. Consolidate phrases that have different wording but the same meaning into a single, clean expression.

3. Do not just list fragmented noun phrases. Reconstruct them into a complete, vivid sentence (e.g., 'is barking', 'is falling', 'is brewing') as if a video is playing right before the eyes.

4. Do not stiffly glue the collected sound information (main sound, background, texture) together. Weave them into a fluent, single-breath sentence using connectors like 'accompanied by', 'while', 'during', or ', producing'."""
}

def format_prompt(query_type: QueryType, caption: str) -> str:
    # 안전한 따옴표 처리를 위해 json.dumps 사용
    safe_caption = json.dumps(caption)
    
    if query_type == QueryType.FULL_CAPTION:
        return FULL_CAPTION_PROMPT_TEMPLATE.format(caption=safe_caption)
    
    # 퓨샷 템플릿에 있는 {{caption}} 자리에 safe_caption을 쏙 집어넣음
    template = SINGLE_REFERENCE_FEWSHOT[query_type]
    return template.format(caption=safe_caption)


def get_system_prompt(query_type: QueryType, backend: str = "gpt") -> str:
    del backend

    return SYSTEM_PROMPTS[query_type]
