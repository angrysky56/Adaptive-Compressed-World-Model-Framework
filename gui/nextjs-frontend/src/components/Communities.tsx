'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, Typography, Paper, Button, CircularProgress,
  Card, CardContent, Divider, Chip, Alert, Accordion,
  AccordionSummary, AccordionDetails, Grid
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RefreshIcon from '@mui/icons-material/Refresh';
import GroupWorkIcon from '@mui/icons-material/GroupWork';
import SearchIcon from '@mui/icons-material/Search';
import { findCommunities, findGaps, Community } from '@/lib/api';
import { useAppContext } from '@/contexts/AppContext';

/**
 * Interface for knowledge gap information
 */
interface KnowledgeGap {
  type: string;
  description: string;
  entities?: Record<string, number>;
  counts?: Record<string, number>;
  components?: string[][];
  contexts?: string[];
  [key: string]: any;
}

/**
 * Interface for component props
 */
interface CommunitiesProps {
  showNotification?: (message: string, severity?: 'success' | 'info' | 'warning' | 'error') => void;
}

/**
 * Communities component
 * Displays knowledge communities and gap analysis
 * 
 * @param showNotification - Function to show notifications
 * @returns The communities component
 */
const Communities: React.FC<CommunitiesProps> = ({ showNotification }) => {
  const { llmEnabled } = useAppContext();
  const [loading, setLoading] = useState<boolean>(false);
  const [communities, setCommunities] = useState<Community[]>([]);
  const [gapsLoading, setGapsLoading] = useState<boolean>(false);
  const [knowledgeGaps, setKnowledgeGaps] = useState<KnowledgeGap[]>([]);
  
  /**
   * Fetch communities from API
   */
  const fetchCommunities = useCallback(async () => {
    try {
      setLoading(true);
      
      const response = await findCommunities();
      setCommunities(response);
    } catch (error) {
      console.error('Error loading communities:', error);
      if (showNotification) {
        showNotification('Failed to load communities', 'error');
      }
    } finally {
      setLoading(false);
    }
  }, [showNotification]);
  
  // Fetch communities on component mount
  useEffect(() => {
    fetchCommunities();
  }, [fetchCommunities]);
  
  /**
   * Fetch knowledge gaps from API
   */
  const findKnowledgeGaps = async () => {
    try {
      setGapsLoading(true);
      
      const response = await findGaps();
      if (response.success) {
        setKnowledgeGaps(response.gaps);
      } else {
        if (showNotification) {
          showNotification('Failed to find knowledge gaps', 'error');
        }
      }
    } catch (error) {
      console.error('Error finding knowledge gaps:', error);
      if (showNotification) {
        showNotification('Failed to find knowledge gaps', 'error');
      }
    } finally {
      setGapsLoading(false);
    }
  };
  
  /**
   * Get color for community header
   */
  const getCommunityColor = (index: number) => {
    const colors = [
      '#C8E6C9', // Light Green
      '#BBDEFB', // Light Blue
      '#F8BBD0', // Light Pink
      '#D1C4E9', // Light Purple
      '#FFE0B2', // Light Orange
      '#B2DFDB', // Light Teal
      '#F0F4C3', // Light Lime
      '#FFCCBC', // Light Deep Orange
      '#CFD8DC'  // Light Blue Grey
    ];
    
    return colors[index % colors.length];
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Knowledge Communities
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">
                Context Communities ({communities.length})
              </Typography>
              
              <Button 
                variant="outlined" 
                startIcon={<RefreshIcon />} 
                onClick={fetchCommunities}
                disabled={loading}
              >
                Refresh
              </Button>
            </Box>
            
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
              </Box>
            ) : communities.length === 0 ? (
              <Alert severity="info">
                No communities found. Add more knowledge to the system to enable community detection.
              </Alert>
            ) : (
              <Box>
                {communities.map((community, index) => (
                  <Card 
                    key={community.id} 
                    sx={{ 
                      mb: 3,
                      overflow: 'visible'
                    }}
                  >
                    <CardContent sx={{ 
                      bgcolor: getCommunityColor(index),
                      py: 2
                    }}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <GroupWorkIcon sx={{ mr: 1 }} />
                        <Typography variant="h6">
                          {community.theme} ({community.size} contexts)
                        </Typography>
                      </Box>
                      
                      {community.key_concepts && community.key_concepts.length > 0 && (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', mt: 1 }}>
                          {community.key_concepts.map((concept) => (
                            <Chip
                              key={concept}
                              label={concept}
                              size="small"
                              sx={{ mr: 0.5, mb: 0.5, backgroundColor: 'rgba(255, 255, 255, 0.7)' }}
                            />
                          ))}
                        </Box>
                      )}
                    </CardContent>
                    
                    <CardContent>
                      {community.summary && (
                        <>
                          <Typography variant="subtitle2" gutterBottom>
                            Community Summary:
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {community.summary}
                          </Typography>
                          <Divider sx={{ mb: 2 }} />
                        </>
                      )}
                      
                      <Typography variant="subtitle2" gutterBottom>
                        Members ({community.members.length}):
                      </Typography>
                      
                      <Box>
                        {community.members.map((member) => (
                          <Card 
                            key={member.id} 
                            variant="outlined" 
                            sx={{ mb: 1.5 }}
                          >
                            <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
                              <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                                {member.summary}
                              </Typography>
                              
                              {member.entities && member.entities.length > 0 && (
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', mt: 1 }}>
                                  {member.entities.map((entity) => (
                                    <Chip
                                      key={entity}
                                      label={entity}
                                      size="small"
                                      variant="outlined"
                                      sx={{ mr: 0.5, mb: 0.5 }}
                                    />
                                  ))}
                                </Box>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </Box>
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
              Knowledge Gaps Analysis
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Identify gaps, sparse areas, and isolated concepts in your knowledge graph.
            </Typography>
            
            <Button
              fullWidth
              variant="contained"
              color="primary"
              startIcon={gapsLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
              onClick={findKnowledgeGaps}
              disabled={gapsLoading}
              sx={{ mb: 3 }}
            >
              {gapsLoading ? 'Analyzing...' : 'Find Knowledge Gaps'}
            </Button>
            
            {knowledgeGaps.length > 0 ? (
              <Box>
                {knowledgeGaps.map((gap, index) => (
                  <Accordion key={index} sx={{ mb: 1 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle2">
                        {gap.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {gap.description}
                      </Typography>
                      
                      {gap.type === 'isolated_concepts' && gap.entities && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Isolated Concepts:
                          </Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                            {Object.keys(gap.entities).slice(0, 10).map((entity) => (
                              <Chip
                                key={entity}
                                label={entity}
                                size="small"
                                sx={{ mr: 0.5, mb: 0.5 }}
                              />
                            ))}
                            {Object.keys(gap.entities).length > 10 && (
                              <Typography variant="body2" color="text.secondary">
                                ...and {Object.keys(gap.entities).length - 10} more
                              </Typography>
                            )}
                          </Box>
                        </Box>
                      )}
                      
                      {gap.type === 'relationship_imbalance' && gap.counts && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Relationship Type Distribution:
                          </Typography>
                          {Object.entries(gap.counts).map(([type, count]) => (
                            <Box key={type} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                              <Typography variant="body2">{type}:</Typography>
                              <Typography variant="body2">{count}</Typography>
                            </Box>
                          ))}
                        </Box>
                      )}
                      
                      {gap.type === 'disconnected_components' && gap.components && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Disconnected Components:
                          </Typography>
                          {gap.components.map((component, idx) => (
                            <Typography key={idx} variant="body2" sx={{ mb: 0.5 }}>
                              Component {idx + 1}: {component.length} contexts
                            </Typography>
                          ))}
                        </Box>
                      )}
                      
                      {gap.type === 'bridge_contexts' && gap.contexts && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Bridge Contexts:
                          </Typography>
                          <Typography variant="body2">
                            {gap.contexts.length} contexts act as bridges
                          </Typography>
                        </Box>
                      )}
                    </AccordionDetails>
                  </Accordion>
                ))}
              </Box>
            ) : (
              <Alert severity="info" sx={{ mb: 2 }}>
                Run the analysis to find knowledge gaps.
              </Alert>
            )}
            
            {!llmEnabled && (
              <Alert severity="info" sx={{ mt: 3 }}>
                Enable LLM enhancement in Settings for more detailed community insights.
              </Alert>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Communities;
