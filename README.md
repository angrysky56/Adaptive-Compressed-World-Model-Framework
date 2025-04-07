# Adaptive Compressed World Model Framework

This repository implements an Adaptive Compressed World Model Framework that efficiently manages complex, dynamic information using compression techniques, event-triggered updates, and adaptive context linking.

 CAUTION: AI generated code and I am not a coder.

## Overview

The Adaptive Compressed World Model Framework provides an efficient, scalable, and flexible solution for managing world models in multi-agent systems. It addresses the challenges of real-time processing, context retention, and efficient resource use in complex environments.

### Key Components (Work in Progress)

#### Knowledge Representation System

- **Compressed Context Packs**: Compact, semantically rich representations of world states
- **Event-Triggered Updates**: Mechanisms that update context only when relevant changes occur
- **Dynamic Context Linking**: Context packs are interconnected for efficient on-demand retrieval
- **Hierarchical Storage**: Efficient caching and long-term storage system

#### GUI Interface

- **Next.js Frontend**: Modern, responsive, type-safe user interface
- **Flask Backend API**: RESTful API for interacting with the knowledge system
- **Interactive Visualization**: Graph-based knowledge visualization
- **LLM Enhancement**: Ollama LLM integration for improved knowledge linking

![image](https://github.com/user-attachments/assets/4e490f66-102b-4e75-a84d-47ae8e218967)

![image](https://github.com/user-attachments/assets/183d76d5-803d-4d3a-95c4-c8dd4fe07230)

Graph is still rough
![image](https://github.com/user-attachments/assets/a394d63e-744f-4305-a595-c853a234a828)

![image](https://github.com/user-attachments/assets/c9a9d27c-6f7f-4f79-b933-af179b22c163)


## Repository Structure

```md
Adaptive-Compressed-World-Model-Framework/
├── data/               # Sample data and simulation results
├── docs/               # Documentation
│   ├── theory.md       # Theoretical foundations
│   ├── architecture.md # System architecture
│   └── development.md  # Development guidelines
├── examples/           # Usage examples
├── gui/                # Web interface
│   ├── app.py          # Flask backend
│   ├── frontend/       # React (CRA) frontend (legacy)
│   ├── nextjs-frontend/# Next.js frontend (recommended)
│   └── requirements.txt# Backend dependencies
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── knowledge/      # Knowledge representation system
│   ├── simulation/     # Multi-agent simulation environment
│   ├── monitoring/     # IoA monitoring system
│   └── utils/          # Utility functions
└── tests/              # Test cases
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+ (for web interface)

### Installation

```bash
# Clone this repository
git clone https://github.com/angrysky56/Adaptive-Compressed-World-Model-Framework.git
cd Adaptive-Compressed-World-Model-Framework

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install and run everything
./start_nextjs.sh
```

### Web Interface

[localhost:3000](http://localhost:3000)

### Development Setup

You don't need to do any of this unless you want to do development and is probably mostly broken since I went to next.js:

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install --use-pep517 -e .
```

#### Backend (Flask)

```bash
cd gui
pip install -r requirements.txt
python app.py
```

This will start the Flask backend server on port 5000.

#### Frontend (Next.js - Recommended)

```bash
cd gui/nextjs-frontend
npm install
npm run dev
```

This will start the Next.js development server on port 3000. The application will be available at [http://localhost:3000](http://localhost:3000).

#### Legacy Frontend (Create React App)

```bash
cd gui/frontend
npm install
npm start
```

This will start the Create React App development server on port 3000. The application will be available at [http://localhost:3000](http://localhost:3000).

### Running a Basic Example

```bash
python examples/knowledge_system_demo.py
```

This demonstrates the core functionality of the knowledge system, including compression, event-triggered updates, and dynamic context linking.

## Key Features and Concepts (Got sidetracked lol, currently not available, but the original idea)

The framework is built on the hypothesis that higher-order intelligences naturally converge in their thinking patterns and representations - a concept supported by research like Fu, Liu, and Tegmark's "Do Two AI Scientists Agree?" (2025) which demonstrates that independently trained AI models often converge toward similar theoretical frameworks when given sufficient data.

### Multi-Agent Simulation

- **Agent Model**: Agents with perception, decision-making, and action capabilities
- **Environment State**: Managed representation of the world, entities, and resources
- **Interaction System**: Framework for agent-environment and agent-agent interactions

#### Intelligence of Agents (IoA) Monitor

- **Convergence Analysis**: Tracking how agent knowledge models evolve and converge
- **Behavior Pattern Detection**: Identifying emergent behaviors and strategies
- **Meta-Learning**: Learning from the collective intelligence of the agent population

### Optimal Representations vs. Objective Truth

Following insights from Gödel's incompleteness theorems, the framework acknowledges that no representation system can capture all aspects of reality perfectly. Instead, it focuses on developing optimal representations - compressed yet semantically rich encodings that effectively predict consequences of actions.

### Adaptive Compression

The framework's core innovation is combining:

1. **Compression techniques** that reduce memory and storage requirements
2. **Event-triggered updates** that minimize computational load
3. **Dynamic context linking** that enables efficient retrieval and expansion

### Convergence of Knowledge

When multiple agents interact within an environment, each developing their own compressed world model, the framework allows studying how their representations converge, providing insights into:

1. Fundamental properties of the environment that shape understanding
2. Effects of different initialization conditions and learning parameters
3. Emergence of shared, optimal representations

## Applications

The adaptive, compressed world model framework is suitable for a wide range of applications:

- **Conversational AI & Virtual Assistants**: Context management with memory retention
- **Autonomous Agents & Robotics**: Efficient world modeling in dynamic environments
- **Multi-Agent Systems**: Collaborative intelligence with minimal communication overhead
- **Virtual Reality (VR) and Augmented Reality (AR)**: Efficient environment management
- **IoT and Edge Computing**: Managing large-scale sensor networks with minimal data transmission
- **Healthcare Monitoring**: Event-triggered updates for patient data analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by research on convergent thinking in AI systems
- Based on concepts from "Do Two AI Scientists Agree?" by Fu, Liu & Tegmark (2025)
