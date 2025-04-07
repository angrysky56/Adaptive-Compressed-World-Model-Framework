"""
ACWMF Web GUI Server

This module provides a Flask-based web server for interacting with the
Adaptive Compressed World Model Framework through a browser interface.
"""

import os
import json
import uuid
import time
import asyncio
from flask import Flask, request, jsonify, send_from_directory, current_app
from flask_cors import CORS

# Add the parent directory to the Python path to import the package
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.knowledge import AdaptiveKnowledgeSystem, LLMContextLinker
from src.knowledge.context_graph import DynamicContextGraph

# Initialize the Flask app
app = Flask(__name__, static_folder="frontend/build")
CORS(app)  # Enable Cross-Origin Resource Sharing

# Define allowed upload file extensions
ALLOWED_EXTENSIONS = {'txt', 'md', 'csv', 'json', 'xml', 'html'}

def allowed_file(filename):
    """Check if the filename has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the knowledge system
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_gui")
os.makedirs(data_dir, exist_ok=True)

# Global knowledge system instance
knowledge_system = None

# Create a dictionary to store graph visualizations by ID
graph_visualizations = {}

# Helper function to run async functions in the sync Flask context
def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine)
    loop.close()
    return result

# Initialize the knowledge system (with or without LLM)
def init_knowledge_system(use_llm=True, ollama_model="mistral-nemo:latest"):
    global knowledge_system
    knowledge_system = AdaptiveKnowledgeSystem(
        storage_dir=data_dir,
        use_llm=use_llm,
        ollama_model=ollama_model
    )
    return knowledge_system


# Routes for the API
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the status of the knowledge system"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "status": "not_initialized",
            "message": "Knowledge system not initialized"
        })
    
    # Get system stats
    stats = run_async(knowledge_system.generate_knowledge_summary())
    
    # Get LLM status
    llm_enabled = knowledge_system.use_llm
    llm_available = (hasattr(knowledge_system.context_graph, 'llm_linker') and 
                    knowledge_system.context_graph.llm_linker is not None)
    
    return jsonify({
        "status": "initialized",
        "llm_enabled": llm_enabled,
        "llm_available": llm_available,
        "stats": stats
    })


@app.route('/api/init', methods=['POST'])
def initialize_knowledge():
    """Initialize the knowledge system"""
    data = request.json
    use_llm = data.get('use_llm', True)
    ollama_model = data.get('ollama_model', "mistral-nemo:latest")
    
    try:
        init_knowledge_system(use_llm, ollama_model)
        return jsonify({
            "success": True,
            "message": "Knowledge system initialized successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error initializing knowledge system: {str(e)}"
        }), 500


@app.route('/api/add_knowledge', methods=['POST'])
def add_knowledge():
    """Add knowledge to the system from text input"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    data = request.json
    text = data.get('text', '')
    critical_entities = data.get('critical_entities', [])
    source = data.get('source', 'manual_input')
    
    if not text:
        return jsonify({
            "success": False,
            "message": "Text is required"
        }), 400
    
    try:
        # Store source information
        metadata = {"source": source}
        
        context_id = run_async(knowledge_system.add_knowledge(text, critical_entities, metadata=metadata))
        return jsonify({
            "success": True,
            "context_id": context_id,
            "message": "Knowledge added successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error adding knowledge: {str(e)}"
        }), 500


@app.route('/api/upload_file', methods=['POST'])
def upload_file():
    """Upload and process a file to add knowledge"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "message": "No file part in the request"
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "success": False,
            "message": "No file selected"
        }), 400
    
    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Get critical entities from form data
    critical_entities = request.form.get('entities', '').split(',')
    critical_entities = [entity.strip() for entity in critical_entities if entity.strip()]
    
    # Process the file based on type
    try:
        file_content = file.read().decode('utf-8')
        
        if not file_content:
            return jsonify({
                "success": False,
                "message": "Empty file content"
            }), 400
        
        # Store source information
        metadata = {
            "source": file.filename, 
            "file_type": file.content_type,
            "upload_time": time.time()
        }
        
        # Use entity extraction to identify key entities if none provided
        if not critical_entities:
            try:
                from src.knowledge.entity_extraction import extract_entities, filter_extracted_entities
                extracted_entities = extract_entities(file_content)
                critical_entities = filter_extracted_entities(extracted_entities)[:15]  # Limit to 15 entities
            except ImportError:
                pass  # Continue without extracted entities
        
        context_id = run_async(knowledge_system.add_knowledge(
            file_content, 
            critical_entities, 
            metadata=metadata
        ))
        
        return jsonify({
            "success": True,
            "context_id": context_id,
            "entities_extracted": critical_entities,
            "message": f"File '{file.filename}' processed and added successfully"
        })
    except UnicodeDecodeError:
        return jsonify({
            "success": False,
            "message": "File is not a text file or contains unsupported encoding"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error processing file: {str(e)}"
        }), 500


@app.route('/api/query_knowledge', methods=['POST'])
def query_knowledge():
    """Query the knowledge system"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    data = request.json
    query_text = data.get('query_text', '')
    max_results = data.get('max_results', 5)
    include_explanations = data.get('include_explanations', False)
    
    if not query_text:
        return jsonify({
            "success": False,
            "message": "Query text is required"
        }), 400
    
    try:
        results = run_async(knowledge_system.query_knowledge(
            query_text,
            max_results=max_results,
            include_explanations=include_explanations
        ))
        
        # Convert numpy arrays to lists for JSON serialization
        for result in results:
            if "embedding" in result and hasattr(result["embedding"], "tolist"):
                result["embedding"] = result["embedding"].tolist()
        
        return jsonify({
            "success": True,
            "results": results,
            "message": f"Found {len(results)} results"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error querying knowledge: {str(e)}"
        }), 500


@app.route('/api/expand_knowledge/<context_id>', methods=['GET'])
def expand_knowledge(context_id):
    """Expand a compressed context"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    try:
        expanded = run_async(knowledge_system.expand_knowledge(context_id))
        
        # Check if expansion was successful
        if "error" in expanded:
            return jsonify({
                "success": False,
                "message": expanded["error"]
            }), 404
        
        return jsonify({
            "success": True,
            "expanded": expanded
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error expanding knowledge: {str(e)}"
        }), 500


@app.route('/api/get_related_contexts/<context_id>', methods=['GET'])
def get_related_contexts(context_id):
    """Get related contexts for a given context"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    include_explanations = request.args.get('include_explanations', 'false').lower() == 'true'
    include_indirect = request.args.get('include_indirect', 'false').lower() == 'true'
    max_depth = int(request.args.get('max_depth', 2))
    
    try:
        related = knowledge_system.context_graph.get_related_contexts(
            context_id,
            include_explanations=include_explanations,
            include_indirect=include_indirect,
            max_depth=max_depth
        )
        
        return jsonify({
            "success": True,
            "related_contexts": related
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting related contexts: {str(e)}"
        }), 500


@app.route('/api/enhance_links', methods=['POST'])
def enhance_links():
    """Enhance knowledge connections using LLM analysis"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    data = request.json
    min_similarity = data.get('min_similarity', 0.5)
    max_suggestions = data.get('max_suggestions', 5)
    
    try:
        results = run_async(knowledge_system.enhance_knowledge_links(
            min_similarity=min_similarity,
            max_suggestions=max_suggestions
        ))
        
        return jsonify({
            "success": True,
            "results": results,
            "message": f"Found {len(results)} potential links"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error enhancing links: {str(e)}"
        }), 500


@app.route('/api/visualize_graph', methods=['GET'])
def visualize_graph():
    """Visualize the knowledge graph"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    # Generate a unique ID for this visualization
    viz_id = str(uuid.uuid4())
    output_file = os.path.join(data_dir, f"knowledge_graph_{viz_id}.png")
    
    try:
        result = run_async(knowledge_system.visualize_knowledge_graph(
            output_file=output_file
        ))
        
        if result:
            # Store visualization info for later retrieval
            graph_visualizations[viz_id] = {
                "path": output_file,
                "time": time.time()
            }
            
            return jsonify({
                "success": True,
                "viz_id": viz_id,
                "message": "Graph visualization created successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to create graph visualization"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error visualizing graph: {str(e)}"
        }), 500


@app.route('/api/graph_image/<viz_id>', methods=['GET'])
def get_graph_image(viz_id):
    """Get a graph visualization image"""
    if viz_id not in graph_visualizations:
        return jsonify({
            "success": False,
            "message": "Visualization not found"
        }), 404
    
    viz_info = graph_visualizations[viz_id]
    try:
        return send_from_directory(
            os.path.dirname(viz_info["path"]),
            os.path.basename(viz_info["path"])
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error serving visualization: {str(e)}"
        }), 500


@app.route('/api/graph_data', methods=['GET'])
def get_graph_data():
    """Get the knowledge graph data for visualization in the frontend"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    try:
        # Extract graph data in a format suitable for visualization libraries
        nodes = []
        edges = []
        
        # Extract node data
        for node_id, node_data in knowledge_system.context_graph.graph.nodes(data=True):
            # Prepare a clean node object with only necessary data
            node = {
                "id": node_id,
                "label": node_data.get("summary", "")[:50] + "...",
                "title": node_data.get("summary", ""),
                "entities": node_data.get("critical_entities", []),
                "creation_time": node_data.get("creation_time", 0),
                "access_count": node_data.get("access_count", 0)
            }
            nodes.append(node)
        
        # Extract edge data
        for source, target, edge_data in knowledge_system.context_graph.graph.edges(data=True):
            # Prepare a clean edge object
            edge = {
                "from": source,
                "to": target,
                "label": edge_data.get("type", "related"),
                "title": f"Relevance: {edge_data.get('weight', 0):.2f}",
                "weight": edge_data.get("weight", 0),
                "type": edge_data.get("type", "related"),
                "shared_entities": edge_data.get("shared_entities", [])
            }
            
            # Add explanation if available
            if "relationship_info" in edge_data and "explanation" in edge_data["relationship_info"]:
                edge["explanation"] = edge_data["relationship_info"]["explanation"]
                
            edges.append(edge)
        
        return jsonify({
            "success": True,
            "nodes": nodes,
            "edges": edges
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting graph data: {str(e)}"
        }), 500


@app.route('/api/find_communities', methods=['GET'])
def find_communities():
    """Find communities in the knowledge graph"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    try:
        communities = knowledge_system.context_graph.find_communities()
        
        # Convert to a list for easier handling in the frontend
        community_list = []
        for community_id, members in communities.items():
            # Get summary information about each community
            member_data = []
            for member_id in members:
                if knowledge_system.context_graph.graph.has_node(member_id):
                    node_data = knowledge_system.context_graph.graph.nodes[member_id]
                    member_data.append({
                        "id": member_id,
                        "summary": node_data.get("summary", "No summary available"),
                        "entities": node_data.get("critical_entities", [])
                    })
            
            # Generate LLM explanation if available
            explanation = {}
            if (hasattr(knowledge_system.context_graph, 'llm_linker') and 
                knowledge_system.context_graph.llm_linker is not None):
                explanation = knowledge_system.context_graph.llm_linker.explain_context_cluster(
                    knowledge_system.context_graph,
                    members
                )
                
            community_list.append({
                "id": community_id,
                "members": member_data,
                "size": len(members),
                "theme": explanation.get("theme", f"Community {community_id}"),
                "summary": explanation.get("summary", ""),
                "key_concepts": explanation.get("key_concepts", [])
            })
        
        return jsonify({
            "success": True,
            "communities": community_list
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error finding communities: {str(e)}"
        }), 500


@app.route('/api/find_gaps', methods=['GET'])
def find_gaps():
    """Find knowledge gaps in the system"""
    global knowledge_system
    if knowledge_system is None:
        return jsonify({
            "success": False,
            "message": "Knowledge system not initialized"
        }), 400
    
    try:
        gaps = knowledge_system.context_graph.find_knowledge_gaps()
        return jsonify({
            "success": True,
            "gaps": gaps
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error finding knowledge gaps: {str(e)}"
        }), 500


# Serve the React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


# Main entry point
if __name__ == "__main__":
    # Initialize knowledge system with default settings
    init_knowledge_system(use_llm=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
