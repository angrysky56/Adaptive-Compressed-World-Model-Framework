"""
Multi-Agent Simulation Environment

This simulation environment implements a world where multiple agents interact,
each with its own compressed world model. The environment supports:
1. Agent creation with compressed world models
2. Environment state updates
3. Agent perception and action
4. Event-triggered updates to agent world models
5. Agent communication and knowledge sharing
"""

import numpy as np
import uuid
import asyncio
import time
import random
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# Link to the Knowledge Representation System
from adaptive_knowledge_system import AdaptiveKnowledgeSystem, CompressedContextPack

@dataclass
class EnvironmentState:
    """Represents the current state of the simulation environment"""
    entities: Dict[str, Dict] = field(default_factory=dict)
    resources: Dict[str, Dict] = field(default_factory=dict)
    time_step: int = 0
    size: Tuple[int, int] = (100, 100)
    
    def add_entity(self, entity_id: str, properties: Dict) -> None:
        """Add an entity to the environment"""
        self.entities[entity_id] = properties
        
    def update_entity(self, entity_id: str, properties: Dict) -> None:
        """Update an entity's properties"""
        if entity_id in self.entities:
            self.entities[entity_id].update(properties)
            
    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the environment"""
        if entity_id in self.entities:
            del self.entities[entity_id]
            
    def add_resource(self, resource_id: str, properties: Dict) -> None:
        """Add a resource to the environment"""
        self.resources[resource_id] = properties
        
    def update_resource(self, resource_id: str, properties: Dict) -> None:
        """Update a resource's properties"""
        if resource_id in self.resources:
            self.resources[resource_id].update(properties)
            
    def remove_resource(self, resource_id: str) -> None:
        """Remove a resource from the environment"""
        if resource_id in self.resources:
            del self.resources[resource_id]
            
    def get_entities_in_range(self, position: Tuple[float, float], radius: float) -> Dict[str, Dict]:
        """Get all entities within a radius of the specified position"""
        in_range = {}
        for entity_id, properties in self.entities.items():
            if "position" in properties:
                entity_pos = properties["position"]
                distance = np.sqrt((entity_pos[0] - position[0])**2 + (entity_pos[1] - position[1])**2)
                if distance <= radius:
                    in_range[entity_id] = properties
        return in_range
        
    def get_resources_in_range(self, position: Tuple[float, float], radius: float) -> Dict[str, Dict]:
        """Get all resources within a radius of the specified position"""
        in_range = {}
        for resource_id, properties in self.resources.items():
            if "position" in properties:
                resource_pos = properties["position"]
                distance = np.sqrt((resource_pos[0] - position[0])**2 + (resource_pos[1] - position[1])**2)
                if distance <= radius:
                    in_range[resource_id] = properties
        return in_range


class Agent:
    """Represents an agent with a compressed world model"""
    
    def __init__(self, agent_id: str, initial_position: Tuple[float, float], 
                 perception_radius: float = 10.0):
        """Initialize an agent with a compressed world model"""
        self.id = agent_id
        self.position = initial_position
        self.perception_radius = perception_radius
        self.knowledge_system = AdaptiveKnowledgeSystem()
        self.world_model_id = None  # Will store the ID of the agent's world model in the knowledge system
        self.last_perception_time = time.time()
        self.memory = []  # Store recent observations
        self.goals = []  # Agent's current goals
        self.state = "exploring"  # Current agent state
        
    async def initialize_world_model(self) -> None:
        """Initialize the agent's world model"""
        initial_model = f"Agent {self.id} world model. Initial position: {self.position}. No other entities known."
        self.world_model_id = await self.knowledge_system.add_knowledge(
            initial_model, 
            ["agent", f"agent_{self.id}", "position", "world_model"]
        )
        
    async def perceive(self, env_state: EnvironmentState) -> Dict:
        """Perceive the environment and update the agent's world model if necessary"""
        # Get entities and resources within perception radius
        nearby_entities = env_state.get_entities_in_range(self.position, self.perception_radius)
        nearby_resources = env_state.get_resources_in_range(self.position, self.perception_radius)
        
        # Exclude self from nearby entities
        if self.id in nearby_entities:
            del nearby_entities[self.id]
            
        # Create perception data
        perception = {
            "time_step": env_state.time_step,
            "position": self.position,
            "nearby_entities": nearby_entities,
            "nearby_resources": nearby_resources,
            "timestamp": time.time()
        }
        
        # Store in memory
        self.memory.append(perception)
        if len(self.memory) > 10:  # Keep only recent perceptions
            self.memory.pop(0)
            
        # Check if world model should be updated
        should_update = await self._should_update_world_model(perception)
        
        if should_update:
            await self._update_world_model(perception)
            
        return perception
    
    async def _should_update_world_model(self, perception: Dict) -> bool:
        """Determine if the world model should be updated based on perception"""
        # Always update if this is the first perception
        if len(self.memory) <= 1:
            return True
            
        # Calculate time since last update
        time_since_update = time.time() - self.last_perception_time
        
        # If it's been a while, update regardless
        if time_since_update > 10.0:  # 10 seconds
            return True
            
        # Check for significant changes in perception
        if len(self.memory) >= 2:
            previous_perception = self.memory[-2]
            
            # Check if position has changed significantly
            pos_change = np.sqrt(
                (perception["position"][0] - previous_perception["position"][0])**2 +
                (perception["position"][1] - previous_perception["position"][1])**2
            )
            
            if pos_change > 5.0:  # Significant movement
                return True
                
            # Check if nearby entities have changed
            prev_entities = set(previous_perception["nearby_entities"].keys())
            curr_entities = set(perception["nearby_entities"].keys())
            
            if prev_entities != curr_entities:
                return True
                
            # Check if nearby resources have changed
            prev_resources = set(previous_perception["nearby_resources"].keys())
            curr_resources = set(perception["nearby_resources"].keys())
            
            if prev_resources != curr_resources:
                return True
                
        return False
    
    async def _update_world_model(self, perception: Dict) -> None:
        """Update the agent's world model with new perception data"""
        # Create a text representation of the perception
        perception_text = self._perception_to_text(perception)
        
        # Update the world model
        updated = await self.knowledge_system.update_knowledge(self.world_model_id, perception_text)
        
        if updated:
            self.last_perception_time = time.time()
            print(f"Agent {self.id} updated world model at time step {perception['time_step']}")
    
    def _perception_to_text(self, perception: Dict) -> str:
        """Convert perception data to text format for the knowledge system"""
        text = f"Agent {self.id} world model update at time step {perception['time_step']}.\n"
        text += f"Current position: {perception['position']}.\n"
        
        # Add nearby entities
        text += f"Nearby entities ({len(perception['nearby_entities'])}):\n"
        for entity_id, properties in perception['nearby_entities'].items():
            text += f"- Entity {entity_id}: {properties}\n"
            
        # Add nearby resources
        text += f"Nearby resources ({len(perception['nearby_resources'])}):\n"
        for resource_id, properties in perception['nearby_resources'].items():
            text += f"- Resource {resource_id}: {properties}\n"
            
        return text
    
    async def decide_action(self) -> Dict:
        """Decide on an action based on the agent's world model and goals"""
        # Retrieve and expand the world model
        expanded_model = await self.knowledge_system.expand_knowledge(self.world_model_id)
        
        # Simple decision logic based on current state
        if self.state == "exploring":
            # Move randomly
            dx = random.uniform(-2.0, 2.0)
            dy = random.uniform(-2.0, 2.0)
            
            return {
                "action": "move",
                "parameters": {
                    "dx": dx,
                    "dy": dy
                }
            }
        elif self.state == "gathering":
            # Find nearest resource in memory
            nearest_resource = None
            nearest_distance = float('inf')
            
            for perception in self.memory:
                for resource_id, properties in perception["nearby_resources"].items():
                    resource_pos = properties["position"]
                    distance = np.sqrt(
                        (resource_pos[0] - self.position[0])**2 + 
                        (resource_pos[1] - self.position[1])**2
                    )
                    
                    if distance < nearest_distance:
                        nearest_resource = (resource_id, properties)
                        nearest_distance = distance
                        
            if nearest_resource:
                resource_pos = nearest_resource[1]["position"]
                direction = (
                    resource_pos[0] - self.position[0],
                    resource_pos[1] - self.position[1]
                )
                
                # Normalize and scale
                magnitude = np.sqrt(direction[0]**2 + direction[1]**2)
                if magnitude > 0:
                    direction = (
                        direction[0] / magnitude * 2.0,
                        direction[1] / magnitude * 2.0
                    )
                    
                return {
                    "action": "move",
                    "parameters": {
                        "dx": direction[0],
                        "dy": direction[1]
                    }
                }
            
            # If no resource found, go back to exploring
            self.state = "exploring"
            return await self.decide_action()
        
        # Default action
        return {
            "action": "wait",
            "parameters": {}
        }
    
    async def execute_action(self, action: Dict, env_state: EnvironmentState) -> bool:
        """Execute the chosen action in the environment"""
        if action["action"] == "move":
            # Update position based on movement parameters
            dx = action["parameters"].get("dx", 0.0)
            dy = action["parameters"].get("dy", 0.0)
            
            # Calculate new position
            new_x = self.position[0] + dx
            new_y = self.position[1] + dy
            
            # Ensure we stay within environment boundaries
            new_x = max(0, min(new_x, env_state.size[0]))
            new_y = max(0, min(new_y, env_state.size[1]))
            
            # Update position
            self.position = (new_x, new_y)
            
            # Update entity in environment
            env_state.update_entity(self.id, {"position": self.position})
            
            return True
        elif action["action"] == "gather":
            # Attempt to gather a resource
            resource_id = action["parameters"].get("resource_id")
            
            if resource_id in env_state.resources:
                resource = env_state.resources[resource_id]
                resource_pos = resource["position"]
                
                # Check if we're close enough to the resource
                distance = np.sqrt(
                    (resource_pos[0] - self.position[0])**2 + 
                    (resource_pos[1] - self.position[1])**2
                )
                
                if distance <= 1.0:  # Close enough to gather
                    # Remove the resource from the environment
                    env_state.remove_resource(resource_id)
                    
                    # Add to agent's inventory (not implemented here)
                    return True
            
            return False
        elif action["action"] == "communicate":
            # Send information to another agent
            target_id = action["parameters"].get("target_id")
            message = action["parameters"].get("message", "")
            
            if target_id in env_state.entities and target_id != self.id:
                # In a real implementation, we would pass the message to the target agent
                # For now, just log the communication
                print(f"Agent {self.id} sends message to Agent {target_id}: {message}")
                return True
                
            return False
        elif action["action"] == "wait":
            # Do nothing
            return True
            
        return False
    
    async def communicate(self, target_agent: 'Agent', message: str) -> bool:
        """Share information with another agent"""
        # Create a context pack from the message
        message_context_id = await self.knowledge_system.add_knowledge(
            f"Message from Agent {self.id}: {message}",
            ["message", f"agent_{self.id}", "communication"]
        )
        
        # Get the expanded context
        expanded_message = await self.knowledge_system.expand_knowledge(message_context_id)
        
        # Add the knowledge to the target agent's knowledge system
        await target_agent.knowledge_system.add_knowledge(
            expanded_message["expanded_content"],
            [f"received_from_agent_{self.id}", "communication"]
        )
        
        print(f"Agent {self.id} shared knowledge with Agent {target_agent.id}")
        return True


class SimulationEnvironment:
    """Manages the multi-agent simulation environment"""
    
    def __init__(self, size: Tuple[int, int] = (100, 100)):
        """Initialize the simulation environment"""
        self.state = EnvironmentState(size=size)
        self.agents = {}
        self.running = False
        self.time_step = 0
        self.max_time_steps = 1000
        self.step_delay = 0.1  # Seconds between time steps
        
    async def add_agent(self, agent: Agent) -> None:
        """Add an agent to the simulation"""
        self.agents[agent.id] = agent
        
        # Add the agent to the environment state
        self.state.add_entity(agent.id, {
            "type": "agent",
            "position": agent.position,
            "perception_radius": agent.perception_radius
        })
        
        # Initialize the agent's world model
        await agent.initialize_world_model()
        
    async def add_resource(self, resource_id: str, position: Tuple[float, float], properties: Dict = None) -> None:
        """Add a resource to the environment"""
        if properties is None:
            properties = {}
            
        properties["position"] = position
        properties["type"] = "resource"
        
        self.state.add_resource(resource_id, properties)
        
    async def run_simulation(self) -> None:
        """Run the simulation until completion"""
        self.running = True
        self.time_step = 0
        
        while self.running and self.time_step < self.max_time_steps:
            await self.step()
            await asyncio.sleep(self.step_delay)
            
        print(f"Simulation completed after {self.time_step} time steps")
        
    async def step(self) -> None:
        """Advance the simulation by one time step"""
        self.time_step += 1
        self.state.time_step = self.time_step
        
        print(f"Time step: {self.time_step}")
        
        # Process each agent
        for agent_id, agent in self.agents.items():
            # Agent perceives the environment
            perception = await agent.perceive(self.state)
            
            # Agent decides on an action
            action = await agent.decide_action()
            
            # Agent executes the action
            await agent.execute_action(action, self.state)
            
        # Optional: Add dynamic events or changes to the environment
        if self.time_step % 50 == 0:
            await self._add_random_resources(5)
            
    async def _add_random_resources(self, count: int) -> None:
        """Add random resources to the environment"""
        for i in range(count):
            resource_id = f"resource_{uuid.uuid4()}"
            position = (
                random.uniform(0, self.state.size[0]),
                random.uniform(0, self.state.size[1])
            )
            
            properties = {
                "value": random.uniform(1.0, 10.0),
                "type": "resource",
                "subtype": random.choice(["food", "water", "material"])
            }
            
            await self.add_resource(resource_id, position, properties)
            
    def stop_simulation(self) -> None:
        """Stop the running simulation"""
        self.running = False
        
    def visualize(self) -> None:
        """Visualize the current state of the simulation"""
        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set limits
        ax.set_xlim(0, self.state.size[0])
        ax.set_ylim(0, self.state.size[1])
        
        # Plot agents
        for agent_id, agent in self.agents.items():
            ax.plot(agent.position[0], agent.position[1], 'bo', markersize=10, label=f"Agent {agent_id}")
            
            # Draw perception radius
            perception_circle = plt.Circle(
                agent.position, 
                agent.perception_radius, 
                fill=False, 
                linestyle='--', 
                color='blue', 
                alpha=0.5
            )
            ax.add_patch(perception_circle)
            
        # Plot resources
        for resource_id, properties in self.state.resources.items():
            position = properties["position"]
            subtype = properties.get("subtype", "generic")
            
            color = 'green'
            if subtype == "water":
                color = 'blue'
            elif subtype == "material":
                color = 'brown'
                
            ax.plot(position[0], position[1], 'o', color=color, markersize=6)
            
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Simulation State at Time Step {self.time_step}')
        
        # Add a legend
        ax.legend()
        
        # Show the plot
        plt.tight_layout()
        plt.show()


# Example usage
async def run_example_simulation():
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
    
    # Run the simulation for a few steps
    for _ in range(10):
        await sim.step()
        
    # Visualize the state
    sim.visualize()

if __name__ == "__main__":
    asyncio.run(run_example_simulation())