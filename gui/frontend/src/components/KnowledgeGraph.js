import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, Typography, Paper, CircularProgress, Alert,
  Button, Drawer, Divider, Chip, Card, CardContent,
  IconButton, FormControlLabel, Switch
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';
import SettingsIcon from '@mui/icons-material/Settings';
import { Network } from 'vis-network/standalone';
import axios from 'axios';

const KnowledgeGraph = ({ showNotification }) => {
  // Graph data and display state
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Detail panel state
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetails, setNodeDetails] = useState(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [nodeRelationships, setNodeRelationships] = useState([]);
  
  // Visualization options
  const [options, setOptions] = useState({
    showRelationshipLabels: true,
    physicsEnabled: true
  });
  
  // Refs
  const networkRef = useRef(null);
  const containerRef = useRef(null);
  
  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);
  
  // Effect to create and update the network visualization
  useEffect(() => {
    if (loading || !containerRef.current) return;
    
    // Create the network visualization
    const data = {
      nodes: nodes.map(node => ({
        id: node.id,
        label: node.label,
        title: node.title,
        color: {
          background: getNodeColor(node),
          border: '#2B7CE9',
          highlight: {
            background: '#D2E5FF',
            border: '#2B7CE9'
          }
        },
        font: { size: 14 },
        size: getNodeSize(node)
      })),
      edges: edges.map(edge => ({
        id: `${edge.from}-${edge.to}`,
        from: edge.from,
        to: edge.to,
        label: options.showRelationshipLabels ? edge.label : '',
        title: edge.title,
        width: edge.weight * 3,
        arrows: {
          to: {
            enabled: edge.label.includes('is_subtopic_of') || edge.label.includes('extends'),
            scaleFactor: 0.5
          }
        },
        color: getEdgeColor(edge)
      }))
    };
    
    // Define network options
    const networkOptions = {
      nodes: {
        shape: 'dot',
        borderWidth: 1,
        shadow: true
      },
      edges: {
        smooth: {
          type: 'continuous',
          forceDirection: 'none'
        },
        shadow: true,
        font: {
          size: 12,
          align: 'middle'
        }
      },
      physics: {
        enabled: options.physicsEnabled,
        barnesHut: {
          gravitationalConstant: -2000,
          centralGravity: 0.3,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.09
        },
        stabilization: {
          iterations: 200
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true
      }
    };
    
    // Initialize the network
    networkRef.current = new Network(containerRef.current, data, networkOptions);
    
    // Add event listeners
    networkRef.current.on('click', function (params) {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        handleNodeClick(nodeId);
      }
    });
    
    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nodes, edges, options, loading, getNodeColor, getNodeSize, getEdgeColor, handleNodeClick]);
  
  // Function to determine node color based on type or attributes
  const getNodeColor = useCallback((node) => {
    // Base color for nodes
    return '#97C2FC';
  }, []);
  
  // Function to determine node size based on importance
  const getNodeSize = useCallback((node) => {
    // Base size with slight variation based on number of entities
    const baseSize = 20;
    const entitiesBonus = node.entities ? node.entities.length * 2 : 0;
    return baseSize + entitiesBonus;
  }, []);
  
  // Function to determine edge color based on type
  const getEdgeColor = useCallback((edge) => {
    // Color mapping for different relationship types
    const typeColorMap = {
      'is_subtopic_of': '#FF9999',
      'is_supertopic_of': '#99FF99',
      'extends': '#9999FF',
      'uses': '#FFCC99',
      'is_related_to': '#CCCCCC',
      'is_strongly_related': '#99CCFF',
      'shares_concepts': '#FFFF99',
      'similarity': '#CCCCCC',
      'suggested_by_llm': '#FF99FF'
    };
    
    return typeColorMap[edge.type] || '#CCCCCC';
  }, []);
  
  // Function to fetch graph data from API
  const fetchGraphData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.get('/api/graph_data');
      if (response.data.success) {
        setNodes(response.data.nodes);
        setEdges(response.data.edges);
      } else {
        setError(response.data.message || 'Failed to load graph data');
      }
    } catch (error) {
      console.error('Error loading graph data:', error);
      setError(error.response?.data?.message || error.message || 'Failed to load graph data');
      showNotification('Error loading graph data', 'error');
    } finally {
      setLoading(false);
    }
  }, [showNotification]);
  
  // Function to handle node click
  const handleNodeClick = useCallback(async (nodeId) => {
    try {
      // Get the selected node data
      const node = nodes.find(n => n.id === nodeId);
      setSelectedNode(node);
      
      // Fetch node details from API
      const detailsResponse = await axios.get(`/api/expand_knowledge/${nodeId}`);
      if (detailsResponse.data.success) {
        setNodeDetails(detailsResponse.data.expanded);
      }
      
      // Fetch related contexts
      const relatedResponse = await axios.get(`/api/get_related_contexts/${nodeId}?include_explanations=true`);
      if (relatedResponse.data.success) {
        setNodeRelationships(relatedResponse.data.related_contexts);
      }
      
      // Open the details panel
      setDetailsOpen(true);
    } catch (error) {
      console.error('Error loading node details:', error);
      showNotification('Error loading node details', 'error');
    }
  }, [nodes, showNotification]);
  
  // Toggle physics simulation
  const togglePhysics = () => {
    setOptions(prev => {
      const newOptions = { ...prev, physicsEnabled: !prev.physicsEnabled };
      if (networkRef.current) {
        networkRef.current.setOptions({ physics: { enabled: newOptions.physicsEnabled } });
      }
      return newOptions;
    });
  };
  
  // Toggle relationship labels
  const toggleRelationshipLabels = () => {
    setOptions(prev => ({ ...prev, showRelationshipLabels: !prev.showRelationshipLabels }));
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Knowledge Graph
      </Typography>
      
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Button 
            variant="outlined" 
            startIcon={<RefreshIcon />} 
            onClick={fetchGraphData}
            disabled={loading}
            sx={{ mr: 1 }}
          >
            Refresh
          </Button>
          <Button 
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => {
              // Show visualization options as a notification
              showNotification(
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Visualization Options
                  </Typography>
                  <FormControlLabel
                    control={<Switch checked={options.physicsEnabled} onChange={togglePhysics} />}
                    label="Physics Simulation"
                  />
                  <FormControlLabel
                    control={<Switch checked={options.showRelationshipLabels} onChange={toggleRelationshipLabels} />}
                    label="Show Relationship Labels"
                  />
                </Box>,
                'info'
              );
            }}
          >
            Options
          </Button>
        </Box>
        
        <Typography variant="body2" color="text.secondary">
          {nodes.length} nodes, {edges.length} relationships
        </Typography>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Paper 
        className="graph-container" 
        elevation={2}
        sx={{ 
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {loading ? (
          <CircularProgress />
        ) : (
          <Box 
            ref={containerRef} 
            sx={{ 
              width: '100%', 
              height: '100%'
            }}
          />
        )}
      </Paper>
      
      <Drawer
        anchor="right"
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
      >
        <Box
          sx={{
            width: 400,
            p: 2,
            height: '100%',
            overflow: 'auto'
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Context Details</Typography>
            <IconButton onClick={() => setDetailsOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
          
          {selectedNode && (
            <>
              <Typography variant="subtitle1" gutterBottom>
                {selectedNode.label}
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                {selectedNode.entities && selectedNode.entities.map(entity => (
                  <Chip 
                    key={entity} 
                    label={entity} 
                    size="small" 
                    sx={{ mr: 0.5, mb: 0.5 }} 
                  />
                ))}
              </Box>
              
              <Divider sx={{ mb: 2 }} />
              
              {nodeDetails ? (
                <>
                  <Typography variant="subtitle2" gutterBottom>
                    Content
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: '#f9f9f9' }}>
                    <Typography variant="body2">
                      {nodeDetails.expanded_content}
                    </Typography>
                  </Paper>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Metadata
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Creation:</strong> {new Date(nodeDetails.creation_time * 1000).toLocaleString()}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Last Access:</strong> {new Date(nodeDetails.last_accessed * 1000).toLocaleString()}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Access Count:</strong> {nodeDetails.access_count}
                    </Typography>
                  </Box>
                </>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                  <CircularProgress size={24} />
                </Box>
              )}
              
              <Divider sx={{ mb: 2 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                Relationships ({nodeRelationships.length})
              </Typography>
              
              {nodeRelationships.length > 0 ? (
                nodeRelationships.map(relation => (
                  <Card key={relation.id} sx={{ mb: 1.5 }}>
                    <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="body2" fontWeight="bold">
                          {nodes.find(n => n.id === relation.id)?.label || 'Unknown Context'}
                        </Typography>
                        <Chip 
                          label={`${(relation.relevance * 100).toFixed(0)}%`} 
                          size="small" 
                          color={relation.relevance > 0.7 ? "success" : relation.relevance > 0.4 ? "primary" : "default"}
                        />
                      </Box>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', mb: 0.5 }}>
                        <Chip 
                          label={relation.relationship_type} 
                          size="small" 
                          variant="outlined"
                          sx={{ mr: 0.5, mb: 0.5 }}
                        />
                        {relation.shared_entities && relation.shared_entities.map(entity => (
                          <Chip 
                            key={entity} 
                            label={entity} 
                            size="small" 
                            variant="outlined"
                            sx={{ mr: 0.5, mb: 0.5 }} 
                          />
                        ))}
                      </Box>
                      
                      {relation.explanation && (
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.85rem' }}>
                          {relation.explanation}
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No relationships found.
                </Typography>
              )}
            </>
          )}
        </Box>
      </Drawer>
    </Box>
  );
};

export default KnowledgeGraph;
