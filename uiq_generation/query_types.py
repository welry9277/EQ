from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class QueryType(Enum):
    KEYWORD = "keyword"
    IMPERATIVE = "imperative"
    POLITE = "polite"
    QUESTION = "question"
    PARAPHRASE = "paraphrase"

    # EQ variants
    KEY_PHRASE = "key_phrase"
    STATEMENT = "statement"
    COMMAND = "command"
    INDIRECT = "indirect"
    FULL_CAPTION = "full_caption"
    
    # Negative variants
    KEYWORD_NEGATIVE = "keyword_negative"
    IMPERATIVE_NEGATIVE = "imperative_negative"
    POLITE_NEGATIVE = "polite_negative"
    QUESTION_NEGATIVE = "question_negative"
    PARAPHRASE_NEGATIVE = "paraphrase_negative"

    @classmethod
    def from_string(cls, value: str):
        for member in cls:
            if member.value == value.lower():
                return member
        raise ValueError(f"Unknown QueryType: {value}")

@dataclass
class QueryResult:
    audio_id: str
    dataset: str
    dataset_slug: str
    query_type: QueryType
    generated_query: str
    original_captions: list[str]
    vgg: Optional[dict] = None
    metadata: Optional[dict] = field(default_factory=dict)
    source_model: str = "gpt-5.1"
    regen_model: str = "gpt-5.1"

    def to_dict(self) -> dict:
        result = {
            "audio_id": self.audio_id,
            "dataset": self.dataset,
            "dataset_slug": self.dataset_slug,
            "query_type": self.query_type.value,
            "generated_query": self.generated_query,
            "original_captions": self.original_captions,
            "metadata": self.metadata,
            "source_model": self.source_model,
            "regen_model": self.regen_model,
        }
        if self.vgg is not None:
            result["vgg"] = self.vgg
        return result
