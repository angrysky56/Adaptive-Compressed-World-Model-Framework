"""
Intelligence of Agents (IoA) Monitoring System

This system monitors the multi-agent simulation and knowledge representation system,
tracking agent behaviors, knowledge evolution, and emergent patterns.
It provides meta-analysis of agent behaviors and knowledge convergence.

Key Features:
1. Monitoring agent knowledge evolution
2. Tracking convergence of agent world models
3. Identifying emergent patterns in agent behavior
4. Measuring knowledge transfer efficiency
5. Visualizing agent interactions and knowledge graphs
"""

import numpy as np
import asyncio
import time
import uuid
from typing import Dict, List, Tuple, Any, Optional, Set
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats
import json

# Importing from the other modules
from adaptive_knowledge_system import AdaptiveKnowledgeSystem, CompressedContextPack
from simulation_environment import SimulationEnvironment, Agent, EnvironmentState


class IoAMonitor:
    """Monitors and analyzes a multi-agent simulation environment"""
    
    def __init__(self, simulation: SimulationEnvironment):
        """Initialize the IoA monitoring system"""
        self.simulation = simulation
        self.knowledge_system = AdaptiveKnowledgeSystem()
        self.observations = []
        self.agent_metrics = {}
        self.convergence_history = []
        self.knowledge_graph = nx.DiGraph()
        self.agent_interaction_graph = nx.Graph()
        
    async def initialize(self) -> None:
        """Initialize the monitoring system"""
        # Create a meta-knowledge entry for the simulation
        sim_description = (
            f"Multi-agent simulation with {len(self.simulation.agents)} agents in an "
            f"environment of size {self.simulation.state.size}. "
            f"Initial resources: {len(self.simulation.state.resources)}."
        )
        
        self.simulation_context_id = await self.knowledge_system.add_knowledge(
            sim_description,
            ["simulation", "multi-agent", "meta-analysis"]
        )
        
        # Initialize metrics for each agent
        for agent_id in self.simulation.agents:
            self.agent_metrics[agent_id] = {
                "world_model_updates": 0,
                "actions_taken": [],
                "position_history": [],
                "communication_events": [],
                "resource_interactions": []
            }
            
            # Add to knowledge graph
            self.knowledge_graph.add_node(
                f"agent:{agent_id}", 
                type="agent",
                creation_time=time.time()
            )
            
        # Initialize agent interaction graph
        for agent_id in self.simulation.agents:
            self.agent_interaction_graph.add_node(agent_id)
            
    async def observe_time_step(self) -> None:
        """Observe and record one time step of the simulation"""
        time_step = self.simulation.time_step
        
        # Collect observations from this time step
        observation = {
            "time_step": time_step,
            "agent_positions": {},
            "resource_positions": {},
            "agent_actions": {},
            "world_model_updates": set(),
            "communication_events": []
        }
        
        # Record agent positions and actions
        for agent_id, agent in self.simulation.agents.items():
            observation["agent_positions"][agent_id] = agent.position
            
            # Store position history for the agent
            self.agent_metrics[agent_id]["position_history"].append(agent.position)
            
            # We would need to capture actions from the simulation
            # This would be implemented in a more complete version
            
        # Record resource positions
        for resource_id, properties in self.simulation.state.resources.items():
            observation["resource_positions"][resource_id] = properties["position"]
            
        # Store the observation
        self.observations.append(observation)
        
        # If it's time for analysis (every 10 time steps)
        if time_step % 10 == 0:
            await self.analyze_simulation_state()
            
    async def analyze_simulation_state(self) -> None:
        """Analyze the current state of the simulation"""
        time_step = self.simulation.time_step
        
        # 1. Analyze agent knowledge convergence
        await self.analyze_knowledge_convergence()
        
        # 2. Analyze agent behaviors
        self.analyze_agent_behaviors()
        
        # 3. Update the IoA's meta-knowledge
        await self.update_meta_knowledge()
        
        # 4. Update interaction graphs
        self.update_interaction_graphs()
        
        print(f"IoA Monitor: Completed analysis at time step {time_step}")
        
    async def analyze_knowledge_convergence(self) -> None:
        """Analyze the convergence of agent world models"""
        # For each pair of agents, compare their world models
        agents = list(self.simulation.agents.values())
        num_agents = len(agents)
        
        if num_agents < 2:
            return  # Not enough agents to analyze convergence
            
        # Create a similarity matrix
        similarity_matrix = np.zeros((num_agents, num_agents))
        
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                agent1 = agents[i]
                agent2 = agents[j]
                
                # Get the expanded world models
                world_model1 = await agent1.knowledge_system.expand_knowledge(agent1.world_model_id)
                world_model2 = await agent2.knowledge_system.expand_knowledge(agent2.world_model_id)
                
                # Calculate similarity using embedding comparison
                # This is a simplified approach; in reality, we would use more sophisticated methods
                query_pack1 = await agent1.knowledge_system.query_knowledge(
                    world_model2["expanded_content"], max_results=1
                )
                
                if query_pack1:
                    similarity = query_pack1[0]["relevance_score"]
                else:
                    similarity = 0.0
                    
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Matrix is symmetric
                
                # Add interaction edge to knowledge graph if similarity is above threshold
                if similarity > 0.6:
                    self.knowledge_graph.add_edge(
                        f"agent:{agent1.id}",
                        f"agent:{agent2.id}",
                        weight=similarity,
                        type="world_model_similarity",
                        time_step=self.simulation.time_step
                    )
                    
        # Calculate average convergence
        off_diag_indices = np.where(~np.eye(num_agents, dtype=bool))
        average_convergence = np.mean(similarity_matrix[off_diag_indices])
        
        # Record convergence history
        self.convergence_history.append({
            "time_step": self.simulation.time_step,
            "average_convergence": average_convergence,
            "similarity_matrix": similarity_matrix.tolist()
        })
        
        print(f"Agent knowledge convergence at time step {self.simulation.time_step}: {average_convergence:.4f}")
        
    def analyze_agent_behaviors(self) -> None:
        """Analyze patterns in agent behaviors"""
        for agent_id, agent in self.simulation.agents.items():
            metrics = self.agent_metrics[agent_id]
            
            # Calculate movement patterns
            if len(metrics["position_history"]) >= 5:
                recent_positions = metrics["position_history"][-5:]
                
                # Calculate average movement distance
                total_distance = 0.0
                for i in range(1, len(recent_positions)):
                    pos1 = recent_positions[i-1]
                    pos2 = recent_positions[i]
                    distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                    total_distance += distance
                    
                avg_movement = total_distance / (len(recent_positions) - 1)
                
                # Determine if the agent is exploring or staying in one area
                if avg_movement < 2.0:
                    behavior = "stationary"
                elif avg_movement < 5.0:
                    behavior = "local_exploration"
                else:
                    behavior = "wide_exploration"
                    
                # Store the behavior analysis
                if "behavior_analysis" not in metrics:
                    metrics["behavior_analysis"] = []
                    
                metrics["behavior_analysis"].append({
                    "time_step": self.simulation.time_step,
                    "avg_movement": avg_movement,
                    "behavior": behavior
                })
                
                # Add behavior node to knowledge graph
                behavior_node_id = f"behavior:{agent_id}:{self.simulation.time_step}"
                self.knowledge_graph.add_node(
                    behavior_node_id,
                    type="behavior",
                    agent_id=agent_id,
                    time_step=self.simulation.time_step,
                    behavior=behavior,
                    avg_movement=avg_movement
                )
                
                # Link to agent
                self.knowledge_graph.add_edge(
                    f"agent:{agent_id}",
                    behavior_node_id,
                    type="exhibits_behavior"
                )
                
    async def update_meta_knowledge(self) -> None:
        """Update the IoA's meta-knowledge about the simulation"""
        time_step = self.simulation.time_step
        
        # Prepare meta-knowledge update
        meta_knowledge = f"Simulation analysis at time step {time_step}:\n"
        
        # Add convergence information
        if self.convergence_history:
            latest_convergence = self.convergence_history[-1]
            meta_knowledge += f"Agent knowledge convergence: {latest_convergence['average_convergence']:.4f}\n"
            
        # Add behavior information
        behavior_counts = {"stationary": 0, "local_exploration": 0, "wide_exploration": 0}
        
        for agent_id, metrics in self.agent_metrics.items():
            if "behavior_analysis" in metrics and metrics["behavior_analysis"]:
                latest_behavior = metrics["behavior_analysis"][-1]["behavior"]
                behavior_counts[latest_behavior] += 1
                
        meta_knowledge += "Agent behaviors:\n"
        for behavior, count in behavior_counts.items():
            meta_knowledge += f"- {behavior}: {count} agents\n"
            
        # Add resource information
        resource_count = len(self.simulation.state.resources)
        meta_knowledge += f"Current resources in environment: {resource_count}\n"
        
        # Update the knowledge system
        await self.knowledge_system.update_knowledge(
            self.simulation_context_id,
            meta_knowledge
        )
        
    def update_interaction_graphs(self) -> None:
        """Update graphs tracking agent interactions and knowledge flow"""
        # For simplicity, we'll update based on proximity
        # In a full implementation, we would track actual interactions
        
        # For each pair of agents, check proximity
        agents = list(self.simulation.agents.values())
        
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                agent1 = agents[i]
                agent2 = agents[j]
                
                # Calculate distance
                distance = np.sqrt(
                    (agent1.position[0] - agent2.position[0])**2 + 
                    (agent1.position[1] - agent2.position[1])**2
                )
                
                # If agents are close, consider it an interaction
                if distance <= (agent1.perception_radius + agent2.perception_radius) / 2:
                    # Update interaction weight
                    if self.agent_interaction_graph.has_edge(agent1.id, agent2.id):
                        self.agent_interaction_graph[agent1.id][agent2.id]["weight"] += 1
                    else:
                        self.agent_interaction_graph.add_edge(
                            agent1.id, 
                            agent2.id, 
                            weight=1, 
                            first_interaction_time=self.simulation.time_step
                        )
                        
    def visualize_agent_convergence(self) -> None:
        """Visualize the convergence of agent knowledge over time"""
        if not self.convergence_history:
            print("No convergence data available yet")
            return
            
        # Extract data for plotting
        time_steps = [entry["time_step"] for entry in self.convergence_history]
        convergence_values = [entry["average_convergence"] for entry in self.convergence_history]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, convergence_values, marker='o', linestyle='-')
        plt.title('Agent Knowledge Convergence Over Time')
        plt.xlabel('Simulation Time Step')
        plt.ylabel('Average Convergence (0-1)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    def visualize_agent_interactions(self) -> None:
        """Visualize the network of agent interactions"""
        if not self.agent_interaction_graph.edges:
            print("No agent interactions recorded yet")
            return
            
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Get position data for nodes
        pos = {}
        for agent_id in self.agent_interaction_graph.nodes:
            agent = self.simulation.agents.get(agent_id)
            if agent:
                pos[agent_id] = agent.position
                
        # Get edge weights for line thickness
        edge_weights = [data["weight"] * 0.5 for _, _, data in self.agent_interaction_graph.edges(data=True)]
        
        # Draw the graph
        nx.draw_networkx(
            self.agent_interaction_graph,
            pos=pos,
            with_labels=True,
            node_color='skyblue',
            node_size=800,
            font_size=10,
            width=edge_weights,
            edge_color='gray',
            alpha=0.7
        )
        
        plt.title('Agent Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def visualize_knowledge_graph(self) -> None:
        """Visualize the knowledge graph"""
        if not self.knowledge_graph.nodes:
            print("Knowledge graph is empty")
            return
            
        # Create the plot
        plt.figure(figsize=(14, 12))
        
        # Prepare node colors based on type
        node_colors = []
        for node in self.knowledge_graph.nodes:
            node_type = self.knowledge_graph.nodes[node].get("type", "unknown")
            if node_type == "agent":
                node_colors.append("skyblue")
            elif node_type == "behavior":
                node_colors.append("lightgreen")
            else:
                node_colors.append("lightgray")
                
        # Prepare edge colors based on type
        edge_colors = []
        for _, _, data in self.knowledge_graph.edges(data=True):
            edge_type = data.get("type", "unknown")
            if edge_type == "world_model_similarity":
                edge_colors.append("red")
            elif edge_type == "exhibits_behavior":
                edge_colors.append("green")
            else:
                edge_colors.append("gray")
                
        # Use a spring layout since we don't have physical positions for all nodes
        pos = nx.spring_layout(self.knowledge_graph, seed=42)
        
        # Draw the graph
        nx.draw_networkx(
            self.knowledge_graph,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=600,
            font_size=8,
            alpha=0.7
        )
        
        plt.title('IoA Knowledge Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    async def generate_insights(self) -> List[str]:
        """Generate insights from the simulation data"""
        insights = []
        
        # Only generate insights if we have enough data
        if len(self.observations) < 10:
            return ["Not enough data to generate insights yet"]
            
        # 1. Convergence trend
        if len(self.convergence_history) >= 3:
            recent_convergence = [entry["average_convergence"] for entry in self.convergence_history[-3:]]
            convergence_trend = np.polyfit(range(len(recent_convergence)), recent_convergence, 1)[0]
            
            if convergence_trend > 0.01:
                insights.append(
                    "Agent knowledge is converging rapidly, indicating effective information sharing."
                )
            elif convergence_trend > 0:
                insights.append(
                    "Agent knowledge is slowly converging over time."
                )
            elif convergence_trend < -0.01:
                insights.append(
                    "Agent knowledge divergence detected - agents may be specializing in different areas."
                )
                
        # 2. Agent behavior patterns
        behavior_counts = {"stationary": 0, "local_exploration": 0, "wide_exploration": 0}
        
        for agent_id, metrics in self.agent_metrics.items():
            if "behavior_analysis" in metrics and metrics["behavior_analysis"]:
                latest_behavior = metrics["behavior_analysis"][-1]["behavior"]
                behavior_counts[latest_behavior] += 1
                
        dominant_behavior = max(behavior_counts.items(), key=lambda x: x[1])
        
        if dominant_behavior[1] > len(self.simulation.agents) * 0.6:
            insights.append(
                f"Agents are predominantly exhibiting {dominant_behavior[0]} behavior ({dominant_behavior[1]} out of {len(self.simulation.agents)} agents)."
            )
            
        # 3. Resource distribution analysis
        if self.observations:
            latest_obs = self.observations[-1]
            resource_positions = list(latest_obs["resource_positions"].values())
            
            if resource_positions:
                # Calculate clustering of resources
                if len(resource_positions) >= 5:
                    resource_positions_array = np.array(resource_positions)
                    
                    # Use K-means to check for resource clusters
                    kmeans = KMeans(n_clusters=min(3, len(resource_positions)), random_state=42)
                    kmeans.fit(resource_positions_array)
                    
                    # Calculate Davies-Bouldin Index for cluster quality
                    cluster_distances = []
                    for i in range(kmeans.n_clusters):
                        cluster_i = resource_positions_array[kmeans.labels_ == i]
                        if len(cluster_i) > 0:
                            centroid_i = kmeans.cluster_centers_[i]
                            avg_distance = np.mean([
                                np.sqrt(np.sum((p - centroid_i) ** 2)) for p in cluster_i
                            ])
                            cluster_distances.append(avg_distance)
                    
                    if cluster_distances and np.mean(cluster_distances) < 20:
                        insights.append(
                            "Resources appear to be clustered in specific regions of the environment."
                        )
                    else:
                        insights.append(
                            "Resources are relatively evenly distributed throughout the environment."
                        )
                        
        # 4. Agent interaction analysis
        if self.agent_interaction_graph.edges:
            interaction_density = len(self.agent_interaction_graph.edges) / (len(self.agent_interaction_graph.nodes) * (len(self.agent_interaction_graph.nodes) - 1) / 2)
            
            if interaction_density > 0.7:
                insights.append(
                    "High level of agent interactions observed, indicating a well-connected social network."
                )
            elif interaction_density < 0.3:
                insights.append(
                    "Low level of agent interactions observed, agents may be operating independently."
                )
                
            # Check for central agents
            centrality = nx.degree_centrality(self.agent_interaction_graph)
            central_agents = [agent_id for agent_id, cent in centrality.items() if cent > 0.6]
            
            if central_agents:
                insights.append(
                    f"Agents {', '.join(central_agents)} are central in the interaction network, suggesting they are key information hubs."
                )
                
        return insights


async def main():
    # Create a simulation environment
    sim = SimulationEnvironment(size=(100, 100))
    
    # Create and add agents
    for i in range(5):
        position = (
            random.uniform(0, sim.state.size[0]),
            random.uniform(0, sim.state.size[1])
        )
        
        agent = Agent(f"agent_{i}", position, perception_radius=15.0)
        await sim.add_agent(agent)
        
    # Add initial resources
    await sim._add_random_resources(20)
    
    # Create the IoA Monitor
    monitor = IoAMonitor(sim)
    await monitor.initialize()
    
    # Run the simulation with monitoring
    for _ in range(50):
        await sim.step()
        await monitor.observe_time_step()
        
    # Generate and print insights
    insights = await monitor.generate_insights()
    print("\nIoA Monitor Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
        
    # Visualize the results
    monitor.visualize_agent_convergence()
    monitor.visualize_agent_interactions()
    monitor.visualize_knowledge_graph()

if __name__ == "__main__":
    asyncio.run(main())
