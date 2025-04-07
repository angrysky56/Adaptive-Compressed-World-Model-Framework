# Adaptive Compressed World Model Framework GUI

This is a web-based graphical user interface for interacting with the Adaptive Compressed World Model Framework. The GUI enables you to visualize, manage, and interact with the knowledge system through an intuitive interface.

## Features

- **Knowledge Graph Visualization**: Interactive visualization of the knowledge graph with details about contexts and relationships.
- **Add Knowledge**: Add new knowledge to the system with custom critical entities.
- **Query Knowledge**: Search for relevant knowledge with semantic understanding.
- **Explore Relationships**: View and understand the relationships between knowledge contexts.
- **Community Analysis**: Analyze clusters of related knowledge with LLM-assisted insights.
- **Knowledge Gap Detection**: Identify gaps, sparse areas, and isolated concepts in your knowledge graph.
- **LLM Enhancement**: Use language models to enhance context linking and relationship analysis.

## Prerequisites

- Python 3.7+
- Node.js and npm
- Ollama (optional, for LLM enhancement)

## Setup and Running

1. Make sure the main ACWMF framework is installed and working.
2. Navigate to the GUI directory.
3. Run the setup script:

```bash
./setup_and_run.sh
```

This script will:
- Create a virtual environment (if it doesn't exist)
- Install required Python dependencies
- Install required npm dependencies
- Build the React frontend
- Start the Flask server

Once the server is running, you can access the GUI at: http://localhost:5000

## Manual Setup

If the automatic setup doesn't work, you can follow these steps:

### Backend Setup

```bash
# Create and activate a virtual environment
python -m venv ../venv
source ../venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the Flask server
python app.py
```

### Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Build the frontend
npm run build
```

## Using LLM Enhancement

For LLM enhancement to work, you need to have Ollama installed and running with the necessary models. Follow these steps:

1. Install Ollama by following the instructions at [ollama.ai](https://ollama.ai).
2. Start the Ollama server:

```bash
ollama serve
```

3. Pull the models you want to use:

```bash
ollama pull mistral-nemo:latest
ollama pull all-minilm:latest
ollama pull nomic-embed-text:latest
```

4. In the GUI, go to Settings and enable LLM Enhancement.
5. Select the appropriate model and reinitialize the system.

## Troubleshooting

- **Frontend not loading**: Make sure the frontend is built correctly by navigating to the `frontend` directory and running `npm run build`.
- **Backend errors**: Check the terminal where the Flask server is running for error messages.
- **LLM Enhancement not working**: Ensure that Ollama is installed and running, and that you have the required models.

## Architecture

- **Backend**: Flask server that interfaces with the ACWMF framework.
- **Frontend**: React application with Material-UI components for a modern, responsive interface.
- **API**: RESTful API for communication between the frontend and backend.

## License

This project is licensed under the terms of the LICENSE file in the root directory.
