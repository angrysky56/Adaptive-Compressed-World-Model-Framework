#!/usr/bin/env python3
"""
Example of running the Intelligence of Agents (IoA) monitor to analyze
simulation data and generate insights.

This script demonstrates how to use the IoA monitoring system to analyze
agent behaviors, knowledge convergence, and emergent patterns in a
multi-agent simulation.
"""

import asyncio
import random
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the Python path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge.adaptive_knowledge_system import AdaptiveKnowledgeSystem
from src.simulation.simulation_environment import SimulationEnvironment, Agent
from src.monitoring.ioa_monitor import IoAMonitor
from src.utils.visualization import (
    plot_agent_positions,
    plot_knowledge_graph,
    plot_convergence_heatmap,
    plot_convergence_over_time
)


async def run_monitor_analysis(simulation_data_path=None):
    """
    Run analysis using the IoA monitoring system.
    
    If simulation_data_path is provided, load simulation data from a file.
    Otherwise, run a new simulation to generate data.
    
    Args:
        simulation_data_path: Optional path to saved simulation data
    """
    if simulation_data_path and os.path.exists(simulation_data_path):
        # Load simulation data from file
        print(f"Loading simulation data from {simulation_data_path}")
        with open(simulation_data_path, 'r') as f:
            simulation_data = json.load(f)
            
        # Recreate simulation environment
        sim = SimulationEnvironment(size=tuple(simulation_data["environment_size"]))
        
        # Add agents based on saved data
        for agent_data in simulation_data["agents"]:
            agent = Agent(
                agent_data["id"],
                tuple(agent_data["position"]),
                perception_radius=agent_data["perception_radius"]
            )
            await sim.add_agent(agent)
            
        # Add resources based on saved data
        for resource_data in simulation_data["resources"]:
            await sim.add_resource(
                resource_data["id"],
                tuple(resource_data["position"]),
                resource_data["properties"]
            )
            
        # Set simulation time step
        sim.time_step = simulation_data["time_step"]
        sim.state.time_step = simulation_data["time_step"]
        
        print(f"Loaded simulation with {len(sim.agents)} agents at time step {sim.time_step}")
    else:
        # Create a new simulation
        print("Creating new simulation environment")
        sim = SimulationEnvironment(size=(100, 100))
        
        # Create and add agents
        for i in range(5):
            position = (
                random.uniform(0, sim.state.size[0]),
                random.uniform(0, sim.state.size[1])
            )
            
            agent = Agent(f"agent_{i}", position, perception_radius=15.0)
            await sim.add_agent(agent)
            
        # Add resources
        await sim._add_random_resources(20)
        
        # Run the simulation for some steps
        print("Running simulation for 30 steps")
        for _ in range(30):
            await sim.step()
            
        # Save simulation data if requested
        if simulation_data_path:
            simulation_data = {
                "time_step": sim.time_step,
                "environment_size": sim.state.size,
                "agents": [
                    {
                        "id": agent_id,
                        "position": agent.position,
                        "perception_radius": agent.perception_radius
                    }
                    for agent_id, agent in sim.agents.items()
                ],
                "resources": [
                    {
                        "id": resource_id,
                        "position": properties["position"],
                        "properties": properties
                    }
                    for resource_id, properties in sim.state.resources.items()
                ]
            }
            
            os.makedirs(os.path.dirname(simulation_data_path), exist_ok=True)
            with open(simulation_data_path, 'w') as f:
                json.dump(simulation_data, f, indent=2)
                
            print(f"Saved simulation data to {simulation_data_path}")
    
    # Create the IoA Monitor
    print("Initializing IoA Monitor")
    monitor = IoAMonitor(sim)
    await monitor.initialize()
    
    # Run observation for the current state
    await monitor.observe_time_step()
    
    # Analyze the simulation
    print("Analyzing simulation state")
    await monitor.analyze_simulation_state()
    
    # Generate insights
    print("\n----- IoA Monitor Insights -----")
    insights = await monitor.generate_insights()
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
        
    # Advanced analysis: agent behavior patterns
    print("\n----- Agent Behavior Analysis -----")
    agent_behaviors = {}
    for agent_id, metrics in monitor.agent_metrics.items():
        if "behavior_analysis" in metrics and metrics["behavior_analysis"]:
            latest_behavior = metrics["behavior_analysis"][-1]
            agent_behaviors[agent_id] = latest_behavior["behavior"]
            print(f"Agent {agent_id}: {latest_behavior['behavior']} (average movement: {latest_behavior['avg_movement']:.2f})")
    
    # Analyze convergence
    if monitor.convergence_history:
        print("\n----- Knowledge Convergence Analysis -----")
        latest_convergence = monitor.convergence_history[-1]
        print(f"Average convergence at time step {sim.time_step}: {latest_convergence['average_convergence']:.4f}")
        
        # Extract similarity matrix
        similarity_matrix = np.array(latest_convergence["similarity_matrix"])
        
        # Plot convergence heatmap
        agent_ids = list(sim.agents.keys())
        plot_convergence_heatmap(similarity_matrix, agent_ids)
        
        # Plot convergence over time if we have history
        if len(monitor.convergence_history) > 1:
            time_steps = [entry["time_step"] for entry in monitor.convergence_history]
            convergence_values = [entry["average_convergence"] for entry in monitor.convergence_history]
            plot_convergence_over_time(time_steps, convergence_values)
    
    # Visualize agent positions
    agent_positions = {agent_id: agent.position for agent_id, agent in sim.agents.items()}
    perception_radii = {agent_id: agent.perception_radius for agent_id, agent in sim.agents.items()}
    resource_positions = {r_id: props["position"] for r_id, props in sim.state.resources.items()}
    
    plot_agent_positions(
        agent_positions,
        perception_radii,
        resource_positions,
        sim.state.size,
        f"Simulation State at Time Step {sim.time_step}"
    )
    
    # Visualize knowledge graph
    plot_knowledge_graph(monitor.knowledge_graph)
    
    print("\nAnalysis complete!")
    
    return monitor


if __name__ == "__main__":
    # Check if a simulation data file was specified
    if len(sys.argv) > 1:
        simulation_data_path = sys.argv[1]
    else:
        # Default path in the data directory
        simulation_data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "simulation_data.json"
        )
    
    # Run the analysis
    asyncio.run(run_monitor_analysis(simulation_data_path))
