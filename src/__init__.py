"""
Adaptive Compressed World Model Framework

A framework for efficient and dynamic management of agents, context, and world information,
focusing on compression, event-triggered updates, and adaptive context linking.

Key components:
- Knowledge: Knowledge representation system using compression and event-triggered updates
- Simulation: Multi-agent simulation environment
- Monitoring: Intelligence of Agents (IoA) monitoring system
- Utils: Utility functions for visualization and other common operations
"""

from .knowledge import (
    CompressedContextPack,
    EventTriggerSystem,
    DynamicContextGraph,
    StorageHierarchy,
    AdaptiveKnowledgeSystem
)

__version__ = '0.1.0'
