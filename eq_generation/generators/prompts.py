from eq_generation.query_types import QueryType

# 1. 원본 캡션 세트 (동일하게 유지)
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



# 2. 고도화된 Few-Shot 템플릿
# 각 쿼리 유형별로 '디테일'은 잃지 않으면서 '어투'만 완벽하게 바꾸도록 수정함.

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

KEY_PHRASE_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Query: Repeated dog barking over distant outdoor ambience

Caption Set:
{_KEY_RAIN_SET}
Query: Steady rain pattering on metal with distant thunder

Caption Set:
{_KEY_COFFEE_SET}
Query: Espresso machine brewing with steam hissing

Caption Set:
{{caption}}
Query:"""

STATEMENT_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Statement: The audio captures a dog barking continuously in an outdoor environment.

Caption Set:
{_KEY_RAIN_SET}
Statement: There is the sound of steady precipitation striking a metallic surface with thunder echoing far away.

Caption Set:
{_KEY_COFFEE_SET}
Statement: An espresso maker is operating, producing brewing sounds and hissing steam.

Caption Set:
{{caption}}
Statement:"""

COMMAND_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Command: Find audio of a dog barking repeatedly with outdoor background ambience.

Caption Set:
{_KEY_RAIN_SET}
Command: Search for a recording of steady rain hitting a metal surface with distant thunder rumbles.

Caption Set:
{_KEY_COFFEE_SET}
Command: Retrieve the sound of a coffee machine brewing espresso and hissing steam.

Caption Set:
{{caption}}
Command:"""

QUESTION_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Question: Does this audio contain the sound of a dog barking repeatedly outside?

Caption Set:
{_KEY_RAIN_SET}
Question: Is there a recording of steady rain falling on metal accompanied by distant thunder?

Caption Set:
{_KEY_COFFEE_SET}
Question: Are there sounds of an espresso machine brewing and emitting hissing steam?

Caption Set:
{{caption}}
Question:"""

INDIRECT_PROMPT_TEMPLATE = f"""Caption Set:
{_KEY_DOG_SET}
Polite: I would appreciate it if you could find a clip where a dog barks repeatedly in an outdoor setting.

Caption Set:
{_KEY_RAIN_SET}
Polite: Could you please locate an audio file featuring rain pattering on metal and distant thunder?

Caption Set:
{_KEY_COFFEE_SET}
Polite: I was wondering if you might have the sound of an espresso maker brewing with steam hissing.

Caption Set:
{{caption}}
Polite:"""




CAPTION_SET_PROMPT_TEMPLATES = {
    QueryType.KEY_PHRASE: KEY_PHRASE_PROMPT_TEMPLATE,
    QueryType.STATEMENT: STATEMENT_PROMPT_TEMPLATE,
    QueryType.QUESTION: QUESTION_PROMPT_TEMPLATE,
    QueryType.COMMAND: COMMAND_PROMPT_TEMPLATE,
    QueryType.INDIRECT: INDIRECT_PROMPT_TEMPLATE,
    QueryType.FULL_CAPTION: FULL_CAPTION_PROMPT_TEMPLATE,
}

# 3. System Prompt 강화 (제약 조건 명확화)
EQ_GPT_SYSTEM_PROMPT = """You are an expert data engineer tasked with generating faithful audio retrieval queries for an evaluation benchmark (EQ: Extended Query).
Return only the final query text, with no explanation, labels, quotes, or extra formatting.

Strict Rules:
1. Stay strictly faithful to the source Caption Set. Synthesize the core details (events, environments, attributes) without inventing new, unsupported information.
2. Formats MUST match the target style:
- key_phrase: A concise noun phrase containing core audio elements. DO NOT use full sentences or periods.
- statement: An objective, declarative sentence describing the audio factually.
- question: A direct interrogative sentence asking if the audio elements exist.
- command: A direct imperative sentence ordering the retrieval system to find the audio.
- indirect: A polite, conversational request (e.g., "Could you please...", "I would appreciate...").
- full_caption: A single, highly detailed declarative sentence that perfectly synthesizes all unique details present across the entire Caption Set.
"""


def format_prompt(query_type: QueryType, caption: str) -> str:
    return CAPTION_SET_PROMPT_TEMPLATES[query_type].format(caption=caption)


def get_system_prompt(query_type: QueryType, backend: str = "gpt") -> str:
    del query_type, backend
    return EQ_GPT_SYSTEM_PROMPT
