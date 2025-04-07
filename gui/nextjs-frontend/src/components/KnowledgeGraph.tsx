'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Box, Typography, Paper, CircularProgress, Alert,
  Button, Drawer, Divider, Chip, Card, CardContent,
  IconButton, FormControlLabel, Switch
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';
import SettingsIcon from '@mui/icons-material/Settings';
import { getGraphData, expandKnowledge, getRelatedContexts } from '@/lib/api';

/**
 * Interface for graph node data
 */
interface Node {
  id: string;
  label: string;
  title?: string;
  entities?: string[];
  creation_time?: number;
  access_count?: number;
  [key: string]: any;
}

/**
 * Interface for graph edge data
 */
interface Edge {
  from: string;
  to: string;
  label: string;
  title?: string;
  weight: number;
  type: string;
  shared_entities?: string[];
  explanation?: string;
  [key: string]: any;
}

/**
 * Interface for relationship data
 */
interface Relationship {
  id: string;
  relevance: number;
  relationship_type: string;
  shared_entities?: string[];
  explanation?: string;
  [key: string]: any;
}

/**
 * Interface for component props
 */
interface KnowledgeGraphProps {
  showNotification?: (message: string | React.ReactNode, severity?: 'success' | 'info' | 'warning' | 'error') => void;
}

/**
 * KnowledgeGraph component
 * Displays the knowledge graph visualization
 * 
 * @param showNotification - Function to show notifications
 * @returns The knowledge graph component
 */
const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ showNotification }) => {
  // Graph data and display state
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Detail panel state
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [nodeDetails, setNodeDetails] = useState<any | null>(null);
  const [detailsOpen, setDetailsOpen] = useState<boolean>(false);
  const [nodeRelationships, setNodeRelationships] = useState<Relationship[]>([]);
  
  // Visualization options
  const [options, setOptions] = useState({
    showRelationshipLabels: true,
    physicsEnabled: true
  });
  
  // Refs
  const networkRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  /**
   * Function to determine node color based on type or attributes
   */
  const getNodeColor = useCallback((node: Node) => {
    // Base color for nodes
    return '#97C2FC';
  }, []);
  
  /**
   * Function to determine node size based on importance
   */
  const getNodeSize = useCallback((node: Node) => {
    // Base size with slight variation based on number of entities
    const baseSize = 20;
    const entitiesBonus = node.entities ? node.entities.length * 2 : 0;
    return baseSize + entitiesBonus;
  }, []);
  
  /**
   * Function to determine edge color based on type
   */
  const getEdgeColor = useCallback((edge: Edge) => {
    // Color mapping for different relationship types
    const typeColorMap: {[key: string]: string} = {
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
  
  /**
   * Function to fetch graph data from API
   */
  const fetchGraphData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await getGraphData();
      if (response.success) {
        setNodes(response.nodes);
        setEdges(response.edges);
      } else {
        setError(response.message || 'Failed to load graph data');
      }
    } catch (error) {
      console.error('Error loading graph data:', error);
      setError('Failed to load graph data');
      if (showNotification) {
        showNotification('Error loading graph data', 'error');
      }
    } finally {
      setLoading(false);
    }
  };
  
  /**
   * Function to handle node click
   */
  const handleNodeClick = useCallback(async (nodeId: string) => {
    try {
      // Get the selected node data
      const node = nodes.find(n => n.id === nodeId);
      if (!node) return;
      
      setSelectedNode(node);
      
      // Fetch node details from API
      const detailsResponse = await expandKnowledge(nodeId);
      if (detailsResponse.success) {
        setNodeDetails(detailsResponse.expanded);
      }
      
      // Fetch related contexts
      const relatedResponse = await getRelatedContexts(nodeId, true);
      if (relatedResponse.success) {
        setNodeRelationships(relatedResponse.related_contexts);
      }
      
      // Open the details panel
      setDetailsOpen(true);
    } catch (error) {
      console.error('Error loading node details:', error);
      if (showNotification) {
        showNotification('Error loading node details', 'error');
      }
    }
  }, [nodes, showNotification]);
  
  /**
   * Toggle physics simulation
   */
  const togglePhysics = () => {
    setOptions(prev => {
      const newOptions = { ...prev, physicsEnabled: !prev.physicsEnabled };
      if (networkRef.current) {
        networkRef.current.setOptions({ physics: { enabled: newOptions.physicsEnabled } });
      }
      return newOptions;
    });
  };
  
  /**
   * Toggle relationship labels
   */
  const toggleRelationshipLabels = () => {
    setOptions(prev => ({ ...prev, showRelationshipLabels: !prev.showRelationshipLabels }));
  };
  
  // Fetch graph data on component mount
  useEffect(() => {
    fetchGraphData();
  }, []);
  
  // Effect to create and update the network visualization
  useEffect(() => {
    if (loading || !containerRef.current || typeof window === 'undefined') return;
    
    // Dynamically import vis-network
    import('vis-network/standalone').then(({ Network }) => {
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
      if (networkRef.current) {
        networkRef.current.destroy();
      }
      
      networkRef.current = new Network(containerRef.current, data, networkOptions);
      
      // Add event listeners
      networkRef.current.on('click', function (params: any) {
        if (params.nodes.length > 0) {
          const nodeId = params.nodes[0];
          handleNodeClick(nodeId);
        }
      });
    });
    
    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nodes, edges, options, loading, handleNodeClick, getNodeColor, getNodeSize, getEdgeColor]);
  
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
              if (showNotification) {
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
              }
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
        elevation={2}
        sx={{ 
          position: 'relative',
          height: '70vh',
          width: '100%',
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
