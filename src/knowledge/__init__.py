"""
Knowledge Representation System

This package provides the components for the adaptive, compressed knowledge representation system:
- CompressedContextPack: Handles compression and expansion of knowledge contexts
- EventTriggerSystem: Controls when updates and expansions occur
- DynamicContextGraph: Manages relationships between context packs
- LLMContextLinker: Uses language models to enhance relationship analysis
- StorageHierarchy: Manages hierarchical storage of context packs
- AdaptiveKnowledgeSystem: Integrates all components for a complete knowledge system
"""

from .context_pack import CompressedContextPack
from .event_trigger import EventTriggerSystem
from .context_graph import DynamicContextGraph
from .adaptive_knowledge_system import StorageHierarchy, AdaptiveKnowledgeSystem

# Import LLMContextLinker if available
try:
    from .llm_context_linker import LLMContextLinker
    __all__ = [
        'CompressedContextPack',
        'EventTriggerSystem',
        'DynamicContextGraph',
        'LLMContextLinker',
        'StorageHierarchy',
        'AdaptiveKnowledgeSystem'
    ]
except ImportError:
    # LLMContextLinker is optional
    __all__ = [
        'CompressedContextPack',
        'EventTriggerSystem',
        'DynamicContextGraph',
        'StorageHierarchy',
        'AdaptiveKnowledgeSystem'
    ]
