import React, { useState } from 'react';
import { 
  Box, Typography, Paper, TextField, Button, 
  CircularProgress, Card, CardContent, Divider,
  Chip, Grid, Alert, FormControlLabel, Switch
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import axios from 'axios';

const QueryKnowledge = ({ showNotification }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [includeExplanations, setIncludeExplanations] = useState(true);
  
  // Handle query input change
  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };
  
  // Submit the query to the API
  const handleSubmit = async () => {
    if (!query.trim()) {
      showNotification('Please enter a search query', 'error');
      return;
    }
    
    try {
      setLoading(true);
      setResults([]);
      
      const response = await axios.post('/api/query_knowledge', {
        query_text: query,
        max_results: 10,
        include_explanations: includeExplanations
      });
      
      if (response.data.success) {
        setResults(response.data.results);
        showNotification(`Found ${response.data.results.length} results`, 'success');
      } else {
        showNotification(response.data.message || 'Query failed', 'error');
      }
      
      setHasSearched(true);
    } catch (error) {
      console.error('Error querying knowledge:', error);
      showNotification(error.response?.data?.message || error.message || 'Query failed', 'error');
      setHasSearched(true);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };
  
  // Format the relevance score as a percentage
  const formatRelevance = (score) => {
    return `${Math.round(score * 100)}%`;
  };
  
  // Determine color based on relevance score
  const getRelevanceColor = (score) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.5) return 'primary';
    if (score >= 0.3) return 'secondary';
    return 'default';
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Query Knowledge
      </Typography>
      
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Search Query
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Enter a search query to find relevant knowledge in the system. The system uses semantic understanding to find the most relevant contexts.
        </Typography>
        
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Enter your query..."
          value={query}
          onChange={handleQueryChange}
          onKeyPress={handleKeyPress}
          disabled={loading}
          sx={{ mb: 2 }}
        />
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={includeExplanations}
                onChange={(e) => setIncludeExplanations(e.target.checked)}
                color="primary"
              />
            }
            label="Include explanations"
          />
          
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            disabled={loading || !query.trim()}
            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
          >
            {loading ? 'Searching...' : 'Search'}
          </Button>
        </Box>
      </Paper>
      
      {hasSearched && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Search Results ({results.length})
          </Typography>
          
          {results.length === 0 ? (
            <Alert severity="info">
              No results found for "{query}". Try a different query or add more knowledge to the system.
            </Alert>
          ) : (
            <Grid container spacing={2}>
              {results.map((result, index) => (
                <Grid item xs={12} key={result.id}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                        <Typography variant="h6">
                          Result {index + 1}
                        </Typography>
                        <Chip 
                          label={`Relevance: ${formatRelevance(result.relevance_score)}`}
                          color={getRelevanceColor(result.relevance_score)}
                        />
                      </Box>
                      
                      <Typography variant="body1" sx={{ mb: 2 }}>
                        {result.summary}
                      </Typography>
                      
                      {result.critical_entities && (
                        <Box sx={{ mb: 2 }}>
                          {result.critical_entities.map((entity) => (
                            <Chip
                              key={entity}
                              label={entity}
                              size="small"
                              variant="outlined"
                              sx={{ m: 0.5 }}
                            />
                          ))}
                        </Box>
                      )}
                      
                      {result.explanation && (
                        <>
                          <Divider sx={{ my: 2 }} />
                          <Typography variant="subtitle2" gutterBottom>
                            Explanation of Relevance
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {result.explanation}
                          </Typography>
                        </>
                      )}
                      
                      <Box sx={{ mt: 2, textAlign: 'right' }}>
                        <Typography variant="body2" color="text.secondary">
                          Context ID: {result.id}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </Box>
      )}
    </Box>
  );
};

export default QueryKnowledge;
