from eq_generation.query_types import QueryType

FULL_CAPTION_PROMPT_TEMPLATE = """Caption Set:
- a dog barking repeatedly in the background
- a dog barking over distant outdoor ambience
- repeated barking from a dog outside
- a dog keeps barking in the background
- outdoor audio with a dog barking again and again
Full_caption: dog barking repeatedly outdoors with background ambience

Caption Set:
- rain falling on a metal surface with distant thunder
- a storm with rainfall hitting metal and thunder far away
- rain pattering on metal while thunder rumbles in the distance
- steady rain on a metal surface with faint thunder
- rainfall striking metal with distant thunder sounds
Full_caption: rain falling on a metal surface with distant thunder

Caption Set:
{caption}
Query:"""

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

KEY_PHRASE_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Query: dog barking in the background

Caption Set:
{_KEY_RAIN_SET}
Query: rain on a metal surface with distant thunder

Caption Set:
{_KEY_COFFEE_SET}
Query: espresso machine brewing with steam hissing

Caption Set:
{{caption}}
Query:"""

STATEMENT_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Statement: Audio featuring a dog vocalizing with barks

Caption Set:
{_KEY_RAIN_SET}
Statement: Recording of precipitation hitting metal with storm sounds

Caption Set:
{_KEY_COFFEE_SET}
Statement: Sound of an espresso maker producing coffee with steaming noises

Caption Set:
{{caption}}
Statement:"""

COMMAND_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Command: Find clear dog barking recordings

Caption Set:
{_KEY_RAIN_SET}
Command: Find high-quality rain and thunder audio

Caption Set:
{_KEY_COFFEE_SET}
Command: Show me clear coffee machine brewing sounds with steam

Caption Set:
{{caption}}
Command:"""

QUESTION_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Question: Is there clear dog barking in this audio?

Caption Set:
{_KEY_RAIN_SET}
Question: Does this audio contain rain and distant thunder?

Caption Set:
{_KEY_COFFEE_SET}
Question: Is there an espresso machine brewing with steam hissing here?

Caption Set:
{{caption}}
Question:"""

INDIRECT_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Polite: I was wondering if you might happen to have a recording of a canine making some noise outside?

Caption Set:
{_KEY_RAIN_SET}
Polite: I'd quite like to listen to some precipitation hitting a roof, if at all possible.

Caption Set:
{_KEY_COFFEE_SET}
Polite: I need audio of a coffee machine brewing espresso with steam hissing.

Caption Set:
{{caption}}
Polite:"""

CAPTION_SET_PROMPT_TEMPLATES = {
    QueryType.KEY_PHRASE: KEY_PHRASE_PROMPT_TEMPLATE,
    QueryType.STATEMENT: STATEMENT_PROMPT_TEMPLATE,
    QueryType.QUESTION: QUESTION_PROMPT_TEMPLATE,
    QueryType.COMMAND: COMMAND_PROMPT_TEMPLATE,
    QueryType.INDIRECT: INDIRECT_PROMPT_TEMPLATE,
}

EQ_GPT_SYSTEM_PROMPT = """You generate faithful audio retrieval queries for EQ (Extended Query).
Return only the final query text, with no explanation, labels, quotes, or extra formatting.
Stay faithful to the source caption or caption set.
When the input is a Caption Set, synthesize supported details across all lines; do not invent content absent from the set.
Do not add sounds, events, speakers, objects, or context that are not supported by the source text.
Use the requested style only:
- key_phrase: short phrase, not a full sentence
- statement: declarative sentence
- question: question sentence
- command: direct retrieval command
- indirect: indirect request
- full_caption: one representative query summarizing the caption set
"""


def format_prompt(query_type: QueryType, caption: str) -> str:
    if query_type == QueryType.FULL_CAPTION:
        return FULL_CAPTION_PROMPT_TEMPLATE.format(caption=caption)
    return CAPTION_SET_PROMPT_TEMPLATES[query_type].format(caption=caption)


def get_system_prompt(query_type: QueryType, backend: str = "gpt") -> str:
    del query_type, backend
    return EQ_GPT_SYSTEM_PROMPT
