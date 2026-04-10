from eq_generation.query_types import QueryType

KEY_PHRASE_PROMPT_TEMPLATE = """Caption: "a dog barking repeatedly in the background"
Query: dog barking in the background

Caption: "rain falling on a metal surface with distant thunder"
Query: rain on a metal surface with distant thunder

Caption: "coffee machine brewing espresso with steam hissing"
Query: espresso machine brewing with steam hissing

Caption: "{caption}"
Query:"""

STATEMENT_PROMPT_TEMPLATE = '''Caption: "A dog barking repeatedly in the background"
Statement: Audio featuring a dog vocalizing with barks

Caption: "Rain falling on a metal surface with distant thunder"
Statement: Recording of precipitation hitting metal with storm sounds

Caption: "Coffee machine brewing espresso with steam hissing"
Statement: Sound of an espresso maker producing coffee with steaming noises

Caption: "{caption}"
Statement:'''

COMMAND_PROMPT_TEMPLATE = '''Caption: "A dog barking repeatedly in the background"
Command: Find clear dog barking recordings

Caption: "Rain falling on a metal surface with distant thunder"
Command: Find high-quality rain and thunder audio

Caption: "Coffee machine brewing espresso with steam hissing"
Command: Show me clear coffee machine brewing sounds with steam

Caption: "{caption}"
Command:'''

QUESTION_PROMPT_TEMPLATE = '''Caption: "A dog barking repeatedly in the background"
Question: Is there clear dog barking in this audio?

Caption: "Rain falling on a metal surface with distant thunder"
Question: Does this audio contain rain and distant thunder?

Caption: "Coffee machine brewing espresso with steam hissing"
Question: Is there an espresso machine brewing with steam hissing here?

Caption: "{caption}"
Question:'''

INDIRECT_PROMPT_TEMPLATE = """Caption: "A dog barking repeatedly in the background"
Polite: I was wondering if you might happen to have a recording of a canine making some noise outside?

Caption: "rain falling on a metal surface with distant thunder"
Polite: I'd quite like to listen to some precipitation hitting a roof, if at all possible.

Caption: "coffee machine brewing espresso with steam hissing"
Polite: I need audio of a coffee machine brewing espresso with steam hissing.

Caption: "{caption}"
Polite:"""

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

EQ_GPT_SYSTEM_PROMPT = """You generate faithful audio retrieval queries for EQ (Extended Query).
Return only the final query text, with no explanation, labels, quotes, or extra formatting.
Stay faithful to the source caption or caption set.
Do not add sounds, events, speakers, objects, or context that are not supported by the source text.
Use the requested style only:
- key_phrase: short phrase, not a full sentence
- statement: declarative sentence
- question: question sentence
- command: direct retrieval command
- indirect: indirect request
- full_caption: one representative query summarizing the caption set
"""


def get_prompt_template(query_type: QueryType) -> str:
    templates = {
        QueryType.KEY_PHRASE: KEY_PHRASE_PROMPT_TEMPLATE,
        QueryType.STATEMENT: STATEMENT_PROMPT_TEMPLATE,
        QueryType.QUESTION: QUESTION_PROMPT_TEMPLATE,
        QueryType.COMMAND: COMMAND_PROMPT_TEMPLATE,
        QueryType.INDIRECT: INDIRECT_PROMPT_TEMPLATE,
        QueryType.FULL_CAPTION: FULL_CAPTION_PROMPT_TEMPLATE,
    }
    return templates[query_type]


def format_prompt(query_type: QueryType, caption: str) -> str:
    return get_prompt_template(query_type).format(caption=caption)


def get_system_prompt(query_type: QueryType, backend: str = "gpt") -> str:
    del query_type, backend
    return EQ_GPT_SYSTEM_PROMPT
