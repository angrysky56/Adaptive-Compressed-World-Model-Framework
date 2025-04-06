#!/usr/bin/env python3
"""
Tests for the Adaptive Knowledge System.
"""

import asyncio
import unittest
import sys
import os
import numpy as np

# Add the parent directory to the Python path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge.adaptive_knowledge_system import AdaptiveKnowledgeSystem, CompressedContextPack


class TestCompressedContextPack(unittest.TestCase):
    """Test the CompressedContextPack class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.compressor = CompressedContextPack()
        
    def test_compression(self):
        """Test that text can be compressed into a context pack."""
        test_text = "This is a test text for compression."
        critical_entities = ["test", "compression"]
        
        compressed_pack = self.compressor.compress(test_text, critical_entities)
        
        # Check that the compressed pack has the expected structure
        self.assertIn("id", compressed_pack)
        self.assertIn("embedding", compressed_pack)
        self.assertIn("summary", compressed_pack)
        self.assertIn("critical_entities", compressed_pack)
        
        # Check that critical entities are preserved
        for entity in critical_entities:
            self.assertIn(entity, compressed_pack["critical_entities"])
        
    def test_expansion(self):
        """Test that a compressed pack can be expanded."""
        test_text = "This is a test text for expansion."
        compressed_pack = self.compressor.compress(test_text)
        
        expanded_text = self.compressor.expand(compressed_pack)
        
        # In the current implementation, expand returns the summary
        self.assertEqual(expanded_text, compressed_pack["summary"])


class TestAdaptiveKnowledgeSystem(unittest.TestCase):
    """Test the AdaptiveKnowledgeSystem class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.knowledge_system = AdaptiveKnowledgeSystem()
        
    async def test_add_knowledge(self):
        """Test adding knowledge to the system."""
        test_text = "This is a test knowledge entry."
        context_id = await self.knowledge_system.add_knowledge(test_text)
        
        # Check that a valid ID was returned
        self.assertIsNotNone(context_id)
        self.assertTrue(isinstance(context_id, str))
        
    async def test_query_knowledge(self):
        """Test querying knowledge from the system."""
        # Add some knowledge
        await self.knowledge_system.add_knowledge(
            "Artificial intelligence is the simulation of human intelligence by machines."
        )
        
        # Query for related knowledge
        results = await self.knowledge_system.query_knowledge("What is AI?")
        
        # Check that results were returned
        self.assertTrue(len(results) > 0)
        
    async def test_update_knowledge(self):
        """Test updating existing knowledge."""
        # Add initial knowledge
        context_id = await self.knowledge_system.add_knowledge(
            "Initial version of knowledge."
        )
        
        # Update the knowledge
        updated = await self.knowledge_system.update_knowledge(
            context_id,
            "Updated version of knowledge with significant changes."
        )
        
        # Check that the update was performed
        self.assertTrue(updated)
        
        # Retrieve and verify the updated knowledge
        expanded = await self.knowledge_system.expand_knowledge(context_id)
        self.assertIn("Updated version", expanded["expanded_content"])


def run_tests():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    # For async tests, we need to run them in an event loop
    loop = asyncio.get_event_loop()
    
    # Create a test suite
    suite = unittest.TestSuite()
    suite.addTest(TestCompressedContextPack('test_compression'))
    suite.addTest(TestCompressedContextPack('test_expansion'))
    
    # Add async tests
    async_tests = [
        TestAdaptiveKnowledgeSystem('test_add_knowledge'),
        TestAdaptiveKnowledgeSystem('test_query_knowledge'),
        TestAdaptiveKnowledgeSystem('test_update_knowledge')
    ]
    
    # Run async tests
    for test in async_tests:
        loop.run_until_complete(getattr(test, test._testMethodName)())
        
    # Run regular tests
    unittest.TextTestRunner().run(suite)
