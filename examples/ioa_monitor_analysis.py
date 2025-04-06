#!/usr/bin/env python3
"""
Example of running the Intelligence of Agents (IoA) monitor to analyze
a simulation and extract insights about agent knowledge convergence.

This script demonstrates how to use the IoA monitor as a separate tool
to analyze a running or completed simulation.
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

from src.simulation.simulation_environment import SimulationEnvironment, Agent
from src.monitoring.ioa_monitor import IoAMonitor


async def analyze_simulation():
    """Run a simulation and perform detailed analysis with the IoA monitor."""
    
    # Create and run a simulation
    simulation = await create_and_run_simulation()
    
    # Create an IoA monitor
    monitor = IoAMonitor(simulation)
    await monitor.initialize()
    
    print("\nRunning IoA analysis...")
    
    # Analyze the complete simulation history
    for time_step in range(simulation.time_step):
        await monitor.observe_time_step()
    
    # Generate meta-analysis report
    await generate_meta_analysis_report(monitor)
    
    # Visualize the analysis results
    visualize_analysis_results(monitor)
    
    return monitor


async def create_and_run_simulation(num_agents=10, num_resources=30, time_steps=100):
    """Create and run a simulation with the specified parameters."""
    print(f"Creating simulation with {num_agents} agents and {num_resources} resources")
    
    # Create simulation environment
    sim = SimulationEnvironment(size=(200, 200))
    
    # Create and add agents with different types
    agent_types = ["explorer", "gatherer", "observer"]
    
    for i in range(num_agents):
        position = (
            random.uniform(0, sim.state.size[0]),
            random.uniform(0, sim.state.size[1])
        )
        
        # Assign a type to influence behavior
        agent_type = random.choice(agent_types)
        
        # Vary perception radius by type
        if agent_type == "explorer":
            perception_radius = 25.0  # Explorers see further
        elif agent_type == "gatherer":
            perception_radius = 10.0  # Gatherers focus on nearby resources
        else:  # observer
            perception_radius = 20.0  # Observers have medium range
            
        agent = Agent(f"agent_{i}", position, perception_radius=perception_radius)
        agent.state = agent_type  # Set initial state based on type
        
        await sim.add_agent(agent)
        print(f"Added {agent_type.capitalize()} Agent {i} at position {position}")
    
    # Add clustered resources to create interesting patterns
    await add_clustered_resources(sim, num_resources)
    
    # Run the simulation
    print(f"\nRunning simulation for {time_steps} time steps...")
    for _ in range(time_steps):
        await sim.step()
        
        # Periodically add new resources
        if _ % 20 == 0 and _ > 0:
            await add_clustered_resources(sim, 5)
            print(f"Added new resources at time step {_}")
    
    print(f"Simulation completed after {time_steps} time steps")
    return sim


async def add_clustered_resources(simulation, count):
    """Add resources in clusters to create interesting patterns."""
    # Create a few cluster centers
    cluster_centers = []
    for _ in range(random.randint(1, 3)):
        center = (
            random.uniform(0, simulation.state.size[0]),
            random.uniform(0, simulation.state.size[1])
        )
        cluster_centers.append(center)
    
    # Add resources around cluster centers
    for i in range(count):
        # Choose a random cluster center
        center = random.choice(cluster_centers)
        
        # Add noise to create a cluster
        position = (
            center[0] + random.gauss(0, 10),  # Cluster with standard deviation of 10
            center[1] + random.gauss(0, 10)
        )
        
        # Ensure position is within bounds
        position = (
            max(0, min(position[0], simulation.state.size[0])),
            max(0, min(position[1], simulation.state.size[1]))
        )
        
        # Create resource with random properties
        resource_id = f"resource_{simulation.time_step}_{i}"
        properties = {
            "value": random.uniform(1.0, 10.0),
            "type": "resource",
            "subtype": random.choice(["food", "water", "material"]),
            "quantity": random.randint(1, 5)
        }
        
        await simulation.add_resource(resource_id, position, properties)


async def generate_meta_analysis_report(monitor):
    """Generate and save a detailed meta-analysis report."""
    print("\nGenerating meta-analysis report...")
    
    # Generate insights
    insights = await monitor.generate_insights()
    
    # Create report dictionary
    report = {
        "simulation_summary": {
            "num_agents": len(monitor.simulation.agents),
            "num_resources": len(monitor.simulation.state.resources),
            "time_steps": monitor.simulation.time_step,
            "environment_size": monitor.simulation.state.size
        },
        "convergence_analysis": {
            "final_convergence": monitor.convergence_history[-1]["average_convergence"] if monitor.convergence_history else None,
            "convergence_trend": calculate_convergence_trend(monitor.convergence_history),
            "time_to_significant_convergence": time_to_convergence(monitor.convergence_history, threshold=0.6)
        },
        "agent_behavior": {
            "behavior_distribution": count_agent_behaviors(monitor),
            "most_active_agents": identify_most_active_agents(monitor),
            "agent_clusters": identify_agent_clusters(monitor)
        },
        "insights": insights
    }
    
    # Save report to file
    os.makedirs("../data", exist_ok=True)
    with open("../data/ioa_meta_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary to console
    print("\n----- IoA Meta-Analysis Report -----")
    print(f"Number of agents: {report['simulation_summary']['num_agents']}")
    print(f"Number of resources: {report['simulation_summary']['num_resources']}")
    print(f"Final convergence: {report['convergence_analysis']['final_convergence']:.4f}")
    
    print("\nInsights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    return report


def calculate_convergence_trend(convergence_history):
    """Calculate the trend in convergence over time."""
    if not convergence_history or len(convergence_history) < 2:
        return None
        
    # Extract time steps and convergence values
    time_steps = []
    convergence_values = []
    
    for entry in convergence_history:
        time_steps.append(entry["time_step"])
        convergence_values.append(entry["average_convergence"])
    
    # Calculate trend using linear regression
    if len(time_steps) >= 2:
        slope, intercept = np.polyfit(time_steps, convergence_values, 1)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "direction": "increasing" if slope > 0 else "decreasing",
            "magnitude": abs(slope)
        }
    
    return None


def time_to_convergence(convergence_history, threshold=0.6):
    """Calculate the time steps until convergence reaches a threshold."""
    if not convergence_history:
        return None
        
    for entry in convergence_history:
        if entry["average_convergence"] >= threshold:
            return entry["time_step"]
    
    return None  # Threshold never reached


def count_agent_behaviors(monitor):
    """Count the different behaviors exhibited by agents."""
    behavior_counts = {}
    
    for agent_id, metrics in monitor.agent_metrics.items():
        if "behavior_analysis" in metrics and metrics["behavior_analysis"]:
            latest_behavior = metrics["behavior_analysis"][-1]["behavior"]
            
            if latest_behavior in behavior_counts:
                behavior_counts[latest_behavior] += 1
            else:
                behavior_counts[latest_behavior] = 1
    
    return behavior_counts


def identify_most_active_agents(monitor):
    """Identify the most active agents based on movement and interactions."""
    agent_activity = {}
    
    for agent_id, metrics in monitor.agent_metrics.items():
        # Calculate average movement
        total_distance = 0.0
        if len(metrics.get("position_history", [])) >= 2:
            positions = metrics["position_history"]
            for i in range(1, len(positions)):
                pos1 = positions[i-1]
                pos2 = positions[i]
                distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                total_distance += distance
            
            avg_movement = total_distance / (len(positions) - 1)
        else:
            avg_movement = 0.0
        
        # Count interactions (simplified)
        num_interactions = len(metrics.get("communication_events", []))
        
        # Calculate overall activity score
        activity_score = avg_movement + (num_interactions * 2)  # Weight interactions higher
        
        agent_activity[agent_id] = {
            "avg_movement": avg_movement,
            "num_interactions": num_interactions,
            "activity_score": activity_score
        }
    
    # Sort by activity score
    sorted_agents = sorted(agent_activity.items(), key=lambda x: x[1]["activity_score"], reverse=True)
    
    # Return top 3 or all if fewer
    return dict(sorted_agents[:min(3, len(sorted_agents))])


def identify_agent_clusters(monitor):
    """Identify clusters of agents based on their positions."""
    # Get final positions for each agent
    final_positions = {}
    
    for agent_id, metrics in monitor.agent_metrics.items():
        if metrics.get("position_history"):
            final_positions[agent_id] = metrics["position_history"][-1]
    
    # Convert to numpy array for clustering
    if len(final_positions) >= 3:  # Need at least 3 points for meaningful clustering
        agents = list(final_positions.keys())
        positions = np.array([final_positions[agent_id] for agent_id in agents])
        
        # Use a simple distance-based clustering
        # In a real implementation, you would use a more sophisticated algorithm
        clusters = []
        cluster_threshold = 20.0  # Maximum distance between agents in same cluster
        
        for i, agent_id in enumerate(agents):
            position = positions[i]
            
            # Check if agent belongs to any existing cluster
            cluster_found = False
            for cluster in clusters:
                for cluster_agent_id in cluster:
                    cluster_agent_pos = final_positions[cluster_agent_id]
                    distance = np.sqrt(
                        (position[0] - cluster_agent_pos[0])**2 + 
                        (position[1] - cluster_agent_pos[1])**2
                    )
                    
                    if distance <= cluster_threshold:
                        cluster.append(agent_id)
                        cluster_found = True
                        break
                        
                if cluster_found:
                    break
            
            # If not in any cluster, create a new one
            if not cluster_found:
                clusters.append([agent_id])
        
        return clusters
    
    return []


def visualize_analysis_results(monitor):
    """Visualize the analysis results from the IoA monitor."""
    print("\nVisualizing analysis results...")
    
    # Visualize agent convergence over time
    visualize_convergence_over_time(monitor)
    
    # Visualize final agent knowledge similarity
    visualize_knowledge_similarity(monitor)
    
    # Visualize agent interaction network
    monitor.visualize_agent_interactions()
    
    # Visualize knowledge graph
    monitor.visualize_knowledge_graph()


def visualize_convergence_over_time(monitor):
    """Visualize the convergence of agent knowledge over time."""
    if not monitor.convergence_history:
        print("No convergence data available")
        return
        
    # Extract data for plotting
    time_steps = [entry["time_step"] for entry in monitor.convergence_history]
    convergence_values = [entry["average_convergence"] for entry in monitor.convergence_history]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, convergence_values, marker='o', linestyle='-', color='blue')
    plt.title('Agent Knowledge Convergence Over Time')
    plt.xlabel('Simulation Time Step')
    plt.ylabel('Average Convergence (0-1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if len(time_steps) >= 2:
        z = np.polyfit(time_steps, convergence_values, 1)
        p = np.poly1d(z)
        plt.plot(time_steps, p(time_steps), "r--", alpha=0.8, label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_knowledge_similarity(monitor):
    """Visualize the final knowledge similarity between agents."""
    if not monitor.convergence_history:
        print("No convergence data available")
        return
        
    # Get the final similarity matrix
    final_similarity = monitor.convergence_history[-1]["similarity_matrix"]
    
    # Get agent IDs
    agent_ids = list(monitor.simulation.agents.keys())
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(final_similarity, cmap='Blues', vmin=0, vmax=1)
    
    # Add labels
    plt.xticks(range(len(agent_ids)), agent_ids, rotation=45)
    plt.yticks(range(len(agent_ids)), agent_ids)
    
    # Add colorbar
    plt.colorbar(label='Similarity')
    
    # Add values in cells
    for i in range(len(agent_ids)):
        for j in range(len(agent_ids)):
            plt.text(j, i, f"{final_similarity[i][j]:.2f}",
                    ha="center", va="center", 
                    color="white" if final_similarity[i][j] > 0.5 else "black")
    
    plt.title('Final Knowledge Similarity Between Agents')
    plt.tight_layout()
    plt.show()


async def main():
    """Main function to run the IoA analysis."""
    monitor = await analyze_simulation()
    
    # Example of accessing the IoA meta-knowledge
    meta_knowledge = await monitor.knowledge_system.expand_knowledge(monitor.simulation_context_id)
    print("\nIoA Meta-Knowledge:")
    print(meta_knowledge["expanded_content"])


if __name__ == "__main__":
    asyncio.run(main())
