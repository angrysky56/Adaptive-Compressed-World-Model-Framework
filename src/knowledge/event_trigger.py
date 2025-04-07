"""
Event-Triggered Update System

This module provides the functionality for determining when knowledge updates should be triggered
based on the significance of changes and adaptive thresholds.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import json


class EventTriggerSystem:
    """
    Controls when updates and expansions of the knowledge model occur.
    
    The EventTriggerSystem monitors changes in data and uses adaptive thresholds to determine 
    when updates are necessary, optimizing the balance between keeping information current
    and minimizing computational load.
    """
    
    def __init__(self, 
                 initial_threshold: float = 0.3, 
                 adaptation_rate: float = 0.05,
                 min_threshold: float = 0.1,
                 max_threshold: float = 0.8,
                 history_window: int = 50):
        """
        Initialize the event trigger system with threshold parameters.
        
        Args:
            initial_threshold: Starting threshold for triggering updates (0-1)
            adaptation_rate: Rate at which thresholds adapt based on history
            min_threshold: Minimum allowed threshold value
            max_threshold: Maximum allowed threshold value
            history_window: Number of events to consider for threshold adaptation
        """
        self.thresholds = {}  # Context-specific thresholds
        self.default_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.history_window = history_window
        self.event_history = []
        
        # Track stats for each context
        self.context_stats = {}
        
    def set_threshold(self, context_id: str, threshold: float) -> None:
        """
        Set a specific threshold for a context.
        
        Args:
            context_id: ID of the context
            threshold: Threshold value (0-1)
        """
        # Ensure threshold is within bounds
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))
        self.thresholds[context_id] = threshold
        
    def get_threshold(self, context_id: str) -> float:
        """
        Get the threshold for a specific context or return the default.
        
        Args:
            context_id: ID of the context
            
        Returns:
            The threshold value for the context
        """
        return self.thresholds.get(context_id, self.default_threshold)
    
    def should_trigger_update(self, context_id: str, change_magnitude: float) -> bool:
        """
        Determine if an update should be triggered based on threshold.
        
        Args:
            context_id: ID of the context
            change_magnitude: Magnitude of the change (0-1)
            
        Returns:
            True if an update should be triggered, False otherwise
        """
        threshold = self.get_threshold(context_id)
        
        # If this is a new context, initialize its stats
        if context_id not in self.context_stats:
            self.context_stats[context_id] = {
                "update_count": 0,
                "last_update_time": None,
                "total_changes": 0,
                "change_history": []
            }
            
        # Track this change in the context's history
        self.context_stats[context_id]["total_changes"] += 1
        self.context_stats[context_id]["change_history"].append(change_magnitude)
        
        # Keep history within window
        if len(self.context_stats[context_id]["change_history"]) > self.history_window:
            self.context_stats[context_id]["change_history"].pop(0)
            
        # Check against threshold
        return change_magnitude > threshold
    
    def record_event(self, context_id: str, change_magnitude: float, was_triggered: bool) -> None:
        """
        Record an event to adapt thresholds over time.
        
        Args:
            context_id: ID of the context
            change_magnitude: Magnitude of the change (0-1)
            was_triggered: Whether an update was triggered
        """
        # Record the event
        event = {
            "context_id": context_id,
            "change_magnitude": change_magnitude,
            "was_triggered": was_triggered,
            "timestamp": time.time(),
            "threshold": self.get_threshold(context_id)
        }
        
        self.event_history.append(event)
        
        # Update context stats if an update was triggered
        if was_triggered:
            if context_id in self.context_stats:
                self.context_stats[context_id]["update_count"] += 1
                self.context_stats[context_id]["last_update_time"] = time.time()
                
        # Keep event history within a reasonable size
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
            
        # Adapt threshold based on recent history
        self._adapt_threshold(context_id)
    
    def _adapt_threshold(self, context_id: str) -> None:
        """
        Adapt the threshold based on recent event history.
        
        This implements an adaptive threshold that adjusts based on:
        1. The frequency of updates
        2. The distribution of change magnitudes
        3. Time-based patterns
        
        Args:
            context_id: ID of the context to adapt threshold for
        """
        # Get recent events for this context
        recent_events = [e for e in self.event_history[-self.history_window:] 
                        if e["context_id"] == context_id]
        
        if len(recent_events) < 5:
            return  # Not enough data to adapt
            
        # Calculate the average change magnitude
        avg_magnitude = sum(e["change_magnitude"] for e in recent_events) / len(recent_events)
        
        # Calculate the standard deviation of change magnitudes
        magnitudes = [e["change_magnitude"] for e in recent_events]
        std_magnitude = np.std(magnitudes) if len(magnitudes) > 1 else 0
        
        # Count triggered and non-triggered events
        triggered_count = sum(1 for e in recent_events if e["was_triggered"])
        non_triggered_count = len(recent_events) - triggered_count
        
        # Calculate the update frequency
        update_frequency = triggered_count / len(recent_events)
        
        # Target frequency - we want to aim for about 20-30% of changes to trigger updates
        target_frequency = 0.25
        
        # Calculate time-based patterns
        current_time = time.time()
        time_diffs = [current_time - e["timestamp"] for e in recent_events]
        avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Get current threshold
        current_threshold = self.get_threshold(context_id)
        
        # Adjust threshold based on update frequency
        if update_frequency > target_frequency:
            # Too many updates, increase threshold
            frequency_adjustment = self.adaptation_rate * 0.5
        elif update_frequency < target_frequency:
            # Too few updates, decrease threshold
            frequency_adjustment = -self.adaptation_rate * 0.5
        else:
            frequency_adjustment = 0
            
        # Adjust threshold based on magnitude distribution
        # If there's high variance, set threshold closer to mean to catch outliers
        if std_magnitude > 0.1:
            magnitude_adjustment = self.adaptation_rate * (avg_magnitude - current_threshold) * 0.5
        else:
            magnitude_adjustment = 0
            
        # Combine adjustments
        total_adjustment = frequency_adjustment + magnitude_adjustment
        
        # Apply adjustment
        new_threshold = current_threshold + total_adjustment
        
        # Ensure threshold stays within bounds
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        
        # Update the threshold
        self.set_threshold(context_id, new_threshold)
        
    def get_context_stats(self, context_id: str = None) -> Dict:
        """
        Get statistics for a specific context or all contexts.
        
        Args:
            context_id: Optional ID of the context to get stats for. If None, return all stats.
            
        Returns:
            Dict containing context statistics
        """
        if context_id:
            if context_id in self.context_stats:
                stats = self.context_stats[context_id].copy()
                stats["current_threshold"] = self.get_threshold(context_id)
                return stats
            return {"error": "Context not found"}
        
        # Return stats for all contexts
        all_stats = {}
        for ctx_id, stats in self.context_stats.items():
            ctx_stats = stats.copy()
            ctx_stats["current_threshold"] = self.get_threshold(ctx_id)
            all_stats[ctx_id] = ctx_stats
            
        return all_stats
    
    def analyze_trigger_patterns(self, context_id: str = None) -> Dict:
        """
        Analyze patterns in the event triggers for a context or all contexts.
        
        This provides insights into how the system is performing and can help
        diagnose issues with threshold settings.
        
        Args:
            context_id: Optional ID of the context to analyze. If None, analyze all contexts.
            
        Returns:
            Dict containing analysis results
        """
        if not self.event_history:
            return {"error": "No event history available"}
            
        # Filter events if context_id is provided
        events = self.event_history
        if context_id:
            events = [e for e in events if e["context_id"] == context_id]
            if not events:
                return {"error": f"No events found for context {context_id}"}
                
        # Count triggered vs. non-triggered events
        triggered = [e for e in events if e["was_triggered"]]
        non_triggered = [e for e in events if not e["was_triggered"]]
        
        # Calculate average change magnitudes
        avg_triggered_magnitude = sum(e["change_magnitude"] for e in triggered) / len(triggered) if triggered else 0
        avg_non_triggered_magnitude = sum(e["change_magnitude"] for e in non_triggered) / len(non_triggered) if non_triggered else 0
        
        # Calculate time-based metrics
        if len(events) > 1:
            timestamps = [e["timestamp"] for e in events]
            sorted_timestamps = sorted(timestamps)
            time_diffs = [sorted_timestamps[i+1] - sorted_timestamps[i] for i in range(len(sorted_timestamps)-1)]
            avg_time_between_events = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            
            # Calculate time between triggered events
            triggered_timestamps = [e["timestamp"] for e in triggered]
            sorted_triggered = sorted(triggered_timestamps)
            triggered_diffs = [sorted_triggered[i+1] - sorted_triggered[i] for i in range(len(sorted_triggered)-1)]
            avg_time_between_updates = sum(triggered_diffs) / len(triggered_diffs) if triggered_diffs else 0
        else:
            avg_time_between_events = 0
            avg_time_between_updates = 0
            
        # Analyze threshold adaptation
        if len(events) > 1:
            thresholds = [e["threshold"] for e in events]
            threshold_changes = [thresholds[i+1] - thresholds[i] for i in range(len(thresholds)-1)]
            avg_threshold_change = sum(abs(tc) for tc in threshold_changes) / len(threshold_changes) if threshold_changes else 0
            current_threshold = thresholds[-1]
            threshold_trend = "increasing" if sum(threshold_changes) > 0 else "decreasing" if sum(threshold_changes) < 0 else "stable"
        else:
            avg_threshold_change = 0
            current_threshold = events[0]["threshold"] if events else 0
            threshold_trend = "unknown"
            
        # Compile the analysis
        analysis = {
            "total_events": len(events),
            "triggered_events": len(triggered),
            "non_triggered_events": len(non_triggered),
            "trigger_rate": len(triggered) / len(events) if events else 0,
            "avg_triggered_magnitude": avg_triggered_magnitude,
            "avg_non_triggered_magnitude": avg_non_triggered_magnitude,
            "avg_time_between_events": avg_time_between_events,
            "avg_time_between_updates": avg_time_between_updates,
            "current_threshold": current_threshold,
            "avg_threshold_change": avg_threshold_change,
            "threshold_trend": threshold_trend
        }
        
        return analysis
    
    def save_to_json(self, filepath: str) -> bool:
        """
        Save the event trigger system state to a JSON file.
        
        Args:
            filepath: Path to the file to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for serialization
            state = {
                "default_threshold": self.default_threshold,
                "adaptation_rate": self.adaptation_rate,
                "min_threshold": self.min_threshold,
                "max_threshold": self.max_threshold,
                "history_window": self.history_window,
                "thresholds": self.thresholds,
                "context_stats": self.context_stats,
                # Only save the most recent events to keep file size reasonable
                "event_history": self.event_history[-100:]
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving event trigger system to {filepath}: {e}")
            return False
    
    def load_from_json(self, filepath: str) -> bool:
        """
        Load the event trigger system state from a JSON file.
        
        Args:
            filepath: Path to the file to load from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Restore state
            self.default_threshold = state.get("default_threshold", self.default_threshold)
            self.adaptation_rate = state.get("adaptation_rate", self.adaptation_rate)
            self.min_threshold = state.get("min_threshold", self.min_threshold)
            self.max_threshold = state.get("max_threshold", self.max_threshold)
            self.history_window = state.get("history_window", self.history_window)
            self.thresholds = state.get("thresholds", {})
            self.context_stats = state.get("context_stats", {})
            self.event_history = state.get("event_history", [])
            
            return True
        except Exception as e:
            print(f"Error loading event trigger system from {filepath}: {e}")
            return False
