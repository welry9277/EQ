from typing import Optional
from uiq_generation.query_types import QueryType

KEYWORD_PROMPT_TEMPLATE = """Caption: "A dog barking repeatedly in the background"
Query: canine, loud vocalizations, outdoor ambience, animal noise

Caption: "Rain falling on a metal surface with distant thunder"
Query: precipitation, storm, tin roof drops, weather sounds

Caption: "Coffee machine brewing espresso with steam hissing"
Query: cafe equipment, barista tools, liquid pouring, pressurized vapor

Caption: "{caption}"
Query:"""

KEYWORD_NEGATIVE_PROMPT_TEMPLATE = """Target: "A man is speaking while typing" | Negative: "A man speaking over bees buzzing"
Query: male voice, human talking, keyboard clicks, -insects, -buzzing

Target: "Crowd applauding" | Negative: "Rain falling on surface"
Query: audience cheering, clapping hands, concert hall, -weather, -precipitation

Target: "Coffee machine brewing espresso" | Negative: "People talking"
Query: cafe equipment, liquid pouring, barista tools, -human voices, -chatter

Target: "{target_caption}" | Negative: "{hard_negative_caption}"
Query:"""


IMPERATIVE_PROMPT_TEMPLATE = """Caption: "A dog barking repeatedly in the background"
Query: Retrieve audio featuring a loud hound outside.

Caption: "Rain falling on a metal surface with distant thunder"
Query: Play some heavy storm noises hitting a tin roof.

Caption: "Coffee machine brewing espresso with steam hissing"
Query: Find me the sound of a cafe appliance making a hot beverage.

Caption: "{caption}"
Query:"""

IMPERATIVE_NEGATIVE_PROMPT_TEMPLATE = """Target: "A man is speaking while typing" | Negative: "A man speaking over bees buzzing"
Query: Get me a male voice talking with keyboard clicks, and make absolutely sure there are no insect noises.

Target: "Crowd applauding" | Negative: "Rain falling on surface"
Query: Search for a cheering audience, but filter out any precipitation sounds.

Target: "Coffee machine brewing espresso" | Negative: "People talking"
Query: Find an espresso maker brewing, strictly without any background chatter.

Target: "{target_caption}" | Negative: "{hard_negative_caption}"
Query:"""


POLITE_PROMPT_TEMPLATE = """Caption: "A dog barking repeatedly in the background"
Query: I was wondering if you might happen to have a recording of a canine making some noise outside?

Caption: "Rain falling on a metal surface with distant thunder"
Query: I'd quite like to listen to some precipitation hitting a roof, if at all possible.

Caption: "Coffee machine brewing espresso with steam hissing"
Query: I don't suppose you have any ambient tracks of a barista appliance preparing a hot drink, would you?

Caption: "{caption}"
Query:"""

POLITE_NEGATIVE_PROMPT_TEMPLATE = """Target: "A man is speaking while typing" | Negative: "A man speaking over bees buzzing"
Query: I'd really appreciate a clip of a gentleman speaking, but ideally without any distracting insect buzzing, if that's alright.

Target: "Crowd applauding" | Negative: "Rain falling on surface"
Query: Perhaps something like an audience cheering would be nice, provided there aren't any weather sounds mixed in.

Target: "Coffee machine brewing espresso" | Negative: "People talking"
Query: Could you kindly play a cafe brewing sound, making sure to avoid any human conversations?

Target: "{target_caption}" | Negative: "{hard_negative_caption}"
Query:"""


QUESTION_PROMPT_TEMPLATE = """Caption: "A dog barking repeatedly in the background"
Query: Could you tell me if this track contains a hound vocalizing loudly?

Caption: "Rain falling on a metal surface with distant thunder"
Query: Does this recording feature a storm hitting a tin roof?

Caption: "Coffee machine brewing espresso with steam hissing"
Query: Are there any clear sounds of a barista tool releasing vapor in here?

Caption: "{caption}"
Query:"""

QUESTION_NEGATIVE_PROMPT_TEMPLATE = """Target: "A man is speaking while typing" | Negative: "A man speaking over bees buzzing"
Query: Are there any clips of a guy talking and typing that completely exclude bug noises?

Target: "Crowd applauding" | Negative: "Rain falling on surface"
Query: Do you have a track of people clapping that doesn't feature any rainfall?

Target: "Coffee machine brewing espresso" | Negative: "People talking"
Query: Is it possible to find a brewing appliance sound that doesn't have people chatting in the background?

Target: "{target_caption}" | Negative: "{hard_negative_caption}"
Query:"""

PARAPHRASE_PROMPT_TEMPLATE = """Caption: "A dog barking repeatedly in the background"
Query: Sound of a dog repeatedly vocalizing in an outdoor setting

Caption: "Rain falling on a metal surface with distant thunder"
Query: Audio of rainfall hitting a metallic surface accompanied by far-off thunder

Caption: "Coffee machine brewing espresso with steam hissing"
Query: Recording of an espresso maker operating with steaming noises

Caption: "{caption}"
Query:"""

PARAPHRASE_NEGATIVE_PROMPT_TEMPLATE = """Target: "A man is speaking while typing" | Negative: "A man speaking over bees buzzing"
Query: Audio of a male voice talking while typing on a keyboard, without any insect buzzing sounds

Target: "Crowd applauding" | Negative: "Rain falling on surface"
Query: Recording of an audience clapping and cheering, excluding any rain or weather noise

Target: "Coffee machine brewing espresso" | Negative: "People talking"
Query: Sound of an espresso machine preparing coffee with brewing noise, without background conversation

Target: "{target_caption}" | Negative: "{hard_negative_caption}"
Query:"""

KEY_PHRASE_PROMPT_TEMPLATE = """Caption: "a dog barking repeatedly in the background"
Query: dog barking in the background

Caption: "rain falling on a metal surface with distant thunder"
Query: rain on a metal surface with distant thunder

Caption: "coffee machine brewing espresso with steam hissing"
Query: espresso machine brewing with steam hissing

Caption: "{caption}"
Query:"""

STATEMENT_PROMPT_TEMPLATE = """Caption: "a dog barking repeatedly in the background"
Query: This audio contains a dog barking repeatedly in the background.

Caption: "rain falling on a metal surface with distant thunder"
Query: This recording features rain falling on a metal surface with distant thunder.

Caption: "coffee machine brewing espresso with steam hissing"
Query: This audio captures a coffee machine brewing espresso with steam hissing.

Caption: "{caption}"
Query:"""

COMMAND_PROMPT_TEMPLATE = """Caption: "a dog barking repeatedly in the background"
Query: Find audio of a dog barking repeatedly in the background.

Caption: "rain falling on a metal surface with distant thunder"
Query: Retrieve a recording of rain falling on a metal surface with distant thunder.

Caption: "coffee machine brewing espresso with steam hissing"
Query: Get me audio of a coffee machine brewing espresso with steam hissing.

Caption: "{caption}"
Query:"""

INDIRECT_PROMPT_TEMPLATE = """Caption: "a dog barking repeatedly in the background"
Query: I'm looking for audio of a dog barking repeatedly in the background.

Caption: "rain falling on a metal surface with distant thunder"
Query: I want a recording of rain falling on a metal surface with distant thunder.

Caption: "coffee machine brewing espresso with steam hissing"
Query: I need audio of a coffee machine brewing espresso with steam hissing.

Caption: "{caption}"
Query:"""

FULL_CAPTION_PROMPT_TEMPLATE = """Caption Set:
- a dog barking repeatedly in the background
- a dog barking over distant outdoor ambience
- repeated barking from a dog outside
- a dog keeps barking in the background
- outdoor audio with a dog barking again and again
Query: dog barking repeatedly outdoors with background ambience

Caption Set:
- rain falling on a metal surface with distant thunder
- a storm with rainfall hitting metal and thunder far away
- rain pattering on metal while thunder rumbles in the distance
- steady rain on a metal surface with faint thunder
- rainfall striking metal with distant thunder sounds
Query: rain falling on a metal surface with distant thunder

Caption Set:
{caption}
Query:"""

GPT_SYSTEM_PROMPT = """You generate natural language queries for audio retrieval.
Return only the query text, with no explanation or surrounding quotes.
CRITICAL INSTRUCTION: Do NOT simply recycle the exact words from the original caption. Use rich vocabulary, synonyms, and realistic human search phrasing to describe the acoustic content."""

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
- full_caption: one representative query summarizing the caption set"""

LLAMA_SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You generate natural language queries for audio retrieval.
Return only the query text, with no explanation or surrounding quotes.
CRITICAL INSTRUCTION: Do NOT simply recycle the exact words from the original caption. Use rich vocabulary, synonyms, and realistic human search phrasing to describe the acoustic content.
<</SYS>>

"""

EQ_LLAMA_SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You generate faithful audio retrieval queries for EQ (Extended Query).
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
<</SYS>>

"""


def get_prompt_template(query_type: QueryType) -> str:
    templates = {
        QueryType.KEYWORD: KEYWORD_PROMPT_TEMPLATE,
        QueryType.KEYWORD_NEGATIVE: KEYWORD_NEGATIVE_PROMPT_TEMPLATE,
        QueryType.IMPERATIVE: IMPERATIVE_PROMPT_TEMPLATE,
        QueryType.IMPERATIVE_NEGATIVE: IMPERATIVE_NEGATIVE_PROMPT_TEMPLATE,
        QueryType.POLITE: POLITE_PROMPT_TEMPLATE,
        QueryType.POLITE_NEGATIVE: POLITE_NEGATIVE_PROMPT_TEMPLATE,
        QueryType.QUESTION: QUESTION_PROMPT_TEMPLATE,
        QueryType.QUESTION_NEGATIVE: QUESTION_NEGATIVE_PROMPT_TEMPLATE,
        QueryType.PARAPHRASE: PARAPHRASE_PROMPT_TEMPLATE,
        QueryType.PARAPHRASE_NEGATIVE: PARAPHRASE_NEGATIVE_PROMPT_TEMPLATE,
        QueryType.KEY_PHRASE: KEY_PHRASE_PROMPT_TEMPLATE,
        QueryType.STATEMENT: STATEMENT_PROMPT_TEMPLATE,
        QueryType.COMMAND: COMMAND_PROMPT_TEMPLATE,
        QueryType.INDIRECT: INDIRECT_PROMPT_TEMPLATE,
        QueryType.FULL_CAPTION: FULL_CAPTION_PROMPT_TEMPLATE,
    }
    return templates[query_type]

def format_prompt(
    query_type: QueryType,
    caption: str,
    hard_negative_caption: Optional[str] = None,
) -> str:
    template = get_prompt_template(query_type)
    if "NEGATIVE" in query_type.name:
        if not hard_negative_caption:
            raise ValueError("Negative queries require a hard_negative_caption.")
        return template.format(
            target_caption=caption,
            hard_negative_caption=hard_negative_caption,
        )
    return template.format(caption=caption)


def is_eq_query_type(query_type: QueryType) -> bool:
    return query_type in {
        QueryType.KEY_PHRASE,
        QueryType.STATEMENT,
        QueryType.QUESTION,
        QueryType.COMMAND,
        QueryType.INDIRECT,
        QueryType.FULL_CAPTION,
    }


def get_system_prompt(query_type: QueryType, backend: str = "gpt") -> str:
    backend = backend.lower().strip()
    if is_eq_query_type(query_type):
        return EQ_GPT_SYSTEM_PROMPT if backend == "gpt" else EQ_LLAMA_SYSTEM_PROMPT
    return GPT_SYSTEM_PROMPT if backend == "gpt" else LLAMA_SYSTEM_PROMPT
