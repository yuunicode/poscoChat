# config/enums.py

from enum import Enum

class ChunkingType(Enum):
    NARRATIVE = "narrative"
    TABLE = "table"

class ChunkingStrategy(Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"
    ROW = "row"

