import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Box, Paper, Grid, Card, CardContent, Button, Chip } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import StorageIcon from '@mui/icons-material/Storage';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import SearchIcon from '@mui/icons-material/Search';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import AccountTreeIcon from '@mui/icons-material/AccountTree';

const HomePage = ({ llmEnabled, llmAvailable }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/status');
      if (response.data.stats) {
        setStats(response.data.stats);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <Box>
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>
          Knowledge System Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Explore and interact with your adaptive compressed knowledge system
        </Typography>
        
        {llmEnabled && (
          <Box sx={{ mt: 2 }}>
            <Chip 
              icon={<SmartToyIcon />}
              label={llmAvailable ? "LLM Enhancement Active" : "LLM Enhancement Unavailable"}
              color={llmAvailable ? "success" : "warning"}
              variant="outlined"
            />
          </Box>
        )}
      </Box>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <BubbleChartIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Knowledge Graph
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Visualize your knowledge as an interactive graph with connected contexts and relationships.
              </Typography>
              <Button variant="contained" onClick={() => navigate('/graph')}>
                View Graph
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <AddCircleOutlineIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Add Knowledge
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Add new knowledge to the system with custom critical entities and automatic compression.
              </Typography>
              <Button variant="contained" onClick={() => navigate('/add')}>
                Add Knowledge
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <SearchIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Query Knowledge
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Search for relevant knowledge across all contexts with semantic understanding.
              </Typography>
              <Button variant="contained" onClick={() => navigate('/query')}>
                Query System
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <AccountTreeIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Explore Relationships
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Discover and understand relationships between contexts, including LLM-generated explanations.
              </Typography>
              <Button variant="contained" onClick={() => navigate('/relationships')}>
                View Relationships
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <StorageIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Community Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Analyze clusters of related knowledge and identify themes and key concepts.
              </Typography>
              <Button variant="contained" onClick={() => navigate('/communities')}>
                View Communities
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <SmartToyIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                System Settings
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Configure system settings, including LLM enhancement options and model selection.
              </Typography>
              <Button variant="contained" onClick={() => navigate('/settings')}>
                Settings
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {loading ? (
        <Paper sx={{ p: 3, mt: 4, textAlign: 'center' }}>
          <Typography variant="body1">Loading system statistics...</Typography>
        </Paper>
      ) : stats && (
        <Paper sx={{ p: 3, mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            System Statistics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={4}>
              <Typography variant="body2" color="text.secondary">
                Total Contexts
              </Typography>
              <Typography variant="h5">
                {stats.total_contexts}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Typography variant="body2" color="text.secondary">
                Average Compression Ratio
              </Typography>
              <Typography variant="h5">
                {stats.avg_compression_ratio ? (stats.avg_compression_ratio * 100).toFixed(1) + '%' : 'N/A'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Typography variant="body2" color="text.secondary">
                Relationships
              </Typography>
              <Typography variant="h5">
                {stats.graph_stats?.edge_count || 0}
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Box>
  );
};

export default HomePage;
