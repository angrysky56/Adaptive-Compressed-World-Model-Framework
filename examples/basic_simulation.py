#!/usr/bin/env python3
"""
Basic simulation example using the Adaptive Compressed World Model Framework.

This example creates a simulation with multiple agents, each with their own
compressed world model, and demonstrates how the IoA monitor tracks their
knowledge convergence.
"""

import asyncio
import random
import os
import sys

# Add the parent directory to the Python path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge.adaptive_knowledge_system import AdaptiveKnowledgeSystem
from src.simulation.simulation_environment import SimulationEnvironment, Agent
from src.monitoring.ioa_monitor import IoAMonitor


async def run_basic_simulation(num_agents=5, num_resources=20, time_steps=50):
    """
    Run a basic simulation with the specified number of agents and resources.
    
    Args:
        num_agents: Number of agents to create
        num_resources: Number of initial resources
        time_steps: Number of simulation steps to run
    """
    print(f"Starting simulation with {num_agents} agents and {num_resources} resources")
    
    # Create simulation environment
    sim = SimulationEnvironment(size=(100, 100))
    
    # Create and add agents
    for i in range(num_agents):
        position = (
            random.uniform(0, sim.state.size[0]),
            random.uniform(0, sim.state.size[1])
        )
        
        agent = Agent(f"agent_{i}", position, perception_radius=15.0)
        await sim.add_agent(agent)
        print(f"Added Agent {i} at position {position}")
    
    # Add initial resources
    await sim._add_random_resources(num_resources)
    print(f"Added {num_resources} resources to the environment")
    
    # Create the IoA Monitor
    monitor = IoAMonitor(sim)
    await monitor.initialize()
    print("Initialized IoA Monitoring System")
    
    # Run the simulation with monitoring
    print("\nRunning simulation...")
    for step in range(time_steps):
        print(f"\nTime step: {step+1}/{time_steps}")
        await sim.step()
        await monitor.observe_time_step()
        
    # Generate and print insights
    print("\n----- IoA Monitor Insights -----")
    insights = await monitor.generate_insights()
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Visualize the results
    print("\nGenerating visualizations...")
    monitor.visualize_agent_convergence()
    monitor.visualize_agent_interactions()
    monitor.visualize_knowledge_graph()
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    asyncio.run(run_basic_simulation())
