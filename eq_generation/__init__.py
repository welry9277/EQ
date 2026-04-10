"""EQ (Extended Query) generation package."""

from eq_generation.data import load_audiocaps, load_config
from eq_generation.query_types import QueryResult, QueryType
from eq_generation.generators import BaseEQGenerator, EQGenerator, GPTEQGenerator

try:
    from eq_generation.semantic_mapping import (
        DEFAULT_MODEL_NAME,
        DEFAULT_TEMPLATES,
        SemanticMapper,
        assign_captions,
        build_category_prototypes,
        load_json_list,
        save_assignments,
    )
except ModuleNotFoundError:
    DEFAULT_MODEL_NAME = None
    DEFAULT_TEMPLATES = None
    SemanticMapper = None
    assign_captions = None
    build_category_prototypes = None
    load_json_list = None
    save_assignments = None

__all__ = [
    "load_audiocaps",
    "load_config",
    "QueryType",
    "QueryResult",
    "BaseEQGenerator",
    "GPTEQGenerator",
    "EQGenerator",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TEMPLATES",
    "SemanticMapper",
    "build_category_prototypes",
    "assign_captions",
    "load_json_list",
    "save_assignments",
]
