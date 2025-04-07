'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, Typography, Paper, Button, CircularProgress,
  Card, CardContent, Divider, Chip, Alert, Slider,
  Grid, Tooltip
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import InfoIcon from '@mui/icons-material/Info';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { getGraphData, enhanceLinks, RelationshipResult } from '@/lib/api';
import { useAppContext } from '@/contexts/AppContext';

/**
 * Interface for relationship data
 */
interface Relationship {
  id: string;
  sourceId: string;
  targetId: string;
  sourceLabel: string;
  targetLabel: string;
  type: string;
  weight: number;
  sharedEntities?: string[];
  explanation?: string | null;
}

/**
 * Interface for component props
 */
interface RelationshipsProps {
  showNotification?: (message: string, severity?: 'success' | 'info' | 'warning' | 'error') => void;
}

/**
 * Relationships component
 * Displays and manages knowledge context relationships
 * 
 * @param showNotification - Function to show notifications
 * @returns The relationships component
 */
const Relationships: React.FC<RelationshipsProps> = ({ showNotification }) => {
  const { llmEnabled, llmAvailable } = useAppContext();
  const [loading, setLoading] = useState<boolean>(false);
  const [enhancing, setEnhancing] = useState<boolean>(false);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(0.5);
  const [maxSuggestions, setMaxSuggestions] = useState<number>(5);
  const [enhancementResults, setEnhancementResults] = useState<RelationshipResult[] | null>(null);
  
  /**
   * Fetch graph data from API
   */
  const fetchGraphData = useCallback(async () => {
    try {
      setLoading(true);
      
      const response = await getGraphData();
      if (response.success) {
        // Transform edges to a more useful format
        const relationships = response.edges.map(edge => {
          const sourceNode = response.nodes.find(node => node.id === edge.from);
          const targetNode = response.nodes.find(node => node.id === edge.to);
          
          return {
            id: `${edge.from}-${edge.to}`,
            sourceId: edge.from,
            targetId: edge.to,
            sourceLabel: sourceNode ? sourceNode.label : 'Unknown',
            targetLabel: targetNode ? targetNode.label : 'Unknown',
            type: edge.type,
            weight: edge.weight,
            sharedEntities: edge.shared_entities || [],
            explanation: edge.explanation || null
          };
        });
        
        // Sort by weight (descending)
        relationships.sort((a, b) => b.weight - a.weight);
        
        setRelationships(relationships);
      } else {
        if (showNotification) {
          showNotification('Failed to load graph data', 'error');
        }
      }
    } catch (error) {
      console.error('Error loading graph data:', error);
      if (showNotification) {
        showNotification('Failed to load graph data', 'error');
      }
    } finally {
      setLoading(false);
    }
  }, [showNotification]);
  
  // Fetch current graph data on component mount
  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);
  
  /**
   * Enhance relationships using LLM
   */
  const enhanceRelationships = useCallback(async () => {
    if (!llmEnabled || !llmAvailable) {
      if (showNotification) {
        showNotification('LLM enhancement is not available', 'error');
      }
      return;
    }
    
    try {
      setEnhancing(true);
      setEnhancementResults(null);
      
      const response = await enhanceLinks(similarityThreshold, maxSuggestions);
      
      setEnhancementResults(response);
      if (showNotification) {
        showNotification(`Found ${response.length} potential links`, 'success');
      }
      
      // Reload the graph data to show new relationships
      await fetchGraphData();
    } catch (error) {
      console.error('Error enhancing relationships:', error);
      if (showNotification) {
        showNotification('Enhancement failed', 'error');
      }
    } finally {
      setEnhancing(false);
    }
  }, [llmEnabled, llmAvailable, similarityThreshold, maxSuggestions, showNotification, fetchGraphData]);
  
  /**
   * Format the relevance score as a percentage
   */
  const formatRelevance = (score: number) => {
    return `${Math.round(score * 100)}%`;
  };
  
  /**
   * Get color based on relationship type
   */
  const getRelationshipColor = (type: string) => {
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
    
    return typeColorMap[type] || '#CCCCCC';
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Knowledge Relationships
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">
                Current Relationships ({relationships.length})
              </Typography>
              
              <Button 
                variant="outlined" 
                startIcon={<RefreshIcon />} 
                onClick={fetchGraphData}
                disabled={loading}
              >
                Refresh
              </Button>
            </Box>
            
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
              </Box>
            ) : relationships.length === 0 ? (
              <Alert severity="info">
                No relationships found. Add more knowledge or use LLM enhancement to discover relationships.
              </Alert>
            ) : (
              <Box>
                {relationships.map((relationship) => (
                  <Card key={relationship.id} sx={{ mb: 2 }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box>
                          <Typography variant="subtitle1" fontWeight="medium">
                            {relationship.sourceLabel} → {relationship.targetLabel}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', mt: 1 }}>
                            <Chip 
                              label={relationship.type}
                              size="small"
                              sx={{ 
                                mr: 1, 
                                mb: 1,
                                bgcolor: getRelationshipColor(relationship.type),
                                color: '#000000'
                              }}
                            />
                            <Chip 
                              label={`Strength: ${formatRelevance(relationship.weight)}`}
                              size="small"
                              variant="outlined"
                              sx={{ mr: 1, mb: 1 }}
                            />
                          </Box>
                        </Box>
                        
                        {relationship.explanation && (
                          <Tooltip title="This relationship has a detailed explanation">
                            <InfoIcon color="primary" />
                          </Tooltip>
                        )}
                      </Box>
                      
                      {relationship.sharedEntities && relationship.sharedEntities.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            Shared Concepts:
                          </Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                            {relationship.sharedEntities.map((entity) => (
                              <Chip
                                key={entity}
                                label={entity}
                                size="small"
                                variant="outlined"
                                sx={{ mr: 0.5, mb: 0.5 }}
                              />
                            ))}
                          </Box>
                        </Box>
                      )}
                      
                      {relationship.explanation && (
                        <Box sx={{ mt: 2 }}>
                          <Divider sx={{ mb: 1 }} />
                          <Typography variant="body2" color="text.secondary">
                            {relationship.explanation}
                          </Typography>
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              LLM-Enhanced Link Discovery
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Use LLM to discover potential missing relationships between contexts based on semantic understanding.
            </Typography>
            
            <Box sx={{ mb: 3 }}>
              <Typography gutterBottom>Similarity Threshold: {similarityThreshold}</Typography>
              <Slider
                value={similarityThreshold}
                onChange={(_, newValue) => setSimilarityThreshold(newValue as number)}
                min={0.3}
                max={0.9}
                step={0.05}
                marks
                disabled={enhancing || !llmEnabled || !llmAvailable}
              />
              
              <Typography gutterBottom sx={{ mt: 2 }}>Maximum Suggestions: {maxSuggestions}</Typography>
              <Slider
                value={maxSuggestions}
                onChange={(_, newValue) => setMaxSuggestions(newValue as number)}
                min={1}
                max={10}
                step={1}
                marks
                disabled={enhancing || !llmEnabled || !llmAvailable}
              />
            </Box>
            
            <Button
              fullWidth
              variant="contained"
              color="primary"
              startIcon={enhancing ? <CircularProgress size={20} color="inherit" /> : <AutoFixHighIcon />}
              onClick={enhanceRelationships}
              disabled={enhancing || !llmEnabled || !llmAvailable}
              sx={{ mb: 2 }}
            >
              {enhancing ? 'Analyzing...' : 'Discover Missing Links'}
            </Button>
            
            {!llmEnabled && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                LLM enhancement is disabled. Enable it in Settings.
              </Alert>
            )}
            
            {llmEnabled && !llmAvailable && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                LLM enhancement is enabled but not available. Check your Ollama connection.
              </Alert>
            )}
            
            {enhancementResults && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Enhancement Results
                </Typography>
                
                {enhancementResults.length === 0 ? (
                  <Alert severity="info">
                    No new relationships discovered. Try adjusting the similarity threshold.
                  </Alert>
                ) : (
                  <Box>
                    {enhancementResults.map((result, index) => (
                      <Card key={index} sx={{ mb: 2 }}>
                        <CardContent>
                          {result.from_id && result.to_id ? (
                            <>
                              <Typography variant="body2" fontWeight="medium" gutterBottom>
                                {result.from_summary} → {result.to_summary}
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                                <Chip 
                                  label={result.relationship_type || 'is_related_to'} 
                                  size="small"
                                  sx={{ 
                                    bgcolor: getRelationshipColor(result.relationship_type || 'is_related_to'),
                                    color: '#000000'
                                  }}
                                />
                                <Chip 
                                  label={`Similarity: ${formatRelevance(result.similarity)}`}
                                  size="small"
                                  variant="outlined"
                                />
                              </Box>
                              
                              {result.explanation && (
                                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                  {result.explanation}
                                </Typography>
                              )}
                            </>
                          ) : (
                            <Alert severity="error">
                              Failed to create relationship
                            </Alert>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                )}
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Relationships;
