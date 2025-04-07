import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, Typography, Paper, FormControl, FormControlLabel,
  Switch, Button, CircularProgress, Alert, Divider,
  Select, MenuItem, InputLabel, Card, CardContent
} from '@mui/material';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import axios from 'axios';

const SettingsPage = ({ showNotification, llmEnabled, setLlmEnabled, initializeSystem }) => {
  const [loading, setLoading] = useState(false);
  const [ollamaModel, setOllamaModel] = useState('mistral-nemo:latest');
  const [availableModels, setAvailableModels] = useState([]);
  const [ollamaStatus, setOllamaStatus] = useState('checking');
  
  // Check if Ollama is available and get models
  const checkOllamaStatus = useCallback(async () => {
    try {
      setOllamaStatus('checking');
      
      // This is a simple check - in a real implementation, you might
      // want to have a dedicated endpoint to check Ollama status
      const response = await axios.get('/api/status');
      
      if (response.data.llm_available) {
        setOllamaStatus('available');
        
        // In a real implementation, you'd get the available models from the API
        // For now, we'll use a static list based on the example logs
        setAvailableModels([
          'all-minilm:latest',
          'qwen2.5-coder:1.5b',
          'deepseek-r1:latest',
          'mistral-nemo:latest',
          'qwen2.5-coder:14b',
          'gemma3:12b',
          'exaone-deep:32b',
          'nomic-embed-text:latest'
        ]);
      } else {
        setOllamaStatus('unavailable');
      }
    } catch (error) {
      console.error('Error checking Ollama status:', error);
      setOllamaStatus('error');
    }
  }, []);
  
  useEffect(() => {
    // Check Ollama status
    checkOllamaStatus();
  }, [checkOllamaStatus]);
  
  // Handle LLM settings update
  const handleReinitialize = useCallback(async () => {
    setLoading(true);
    try {
      await initializeSystem(llmEnabled, ollamaModel);
      showNotification('System reinitialized successfully', 'success');
      await checkOllamaStatus();
    } catch (error) {
      console.error('Error reinitializing system:', error);
      showNotification('Error reinitializing system', 'error');
    } finally {
      setLoading(false);
    }
  }, [initializeSystem, llmEnabled, ollamaModel, showNotification, checkOllamaStatus]);
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Settings
      </Typography>
      
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          LLM Enhancement Settings
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Configure LLM enhancement options for improved relationship analysis and context linking.
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <FormControlLabel
            control={
              <Switch
                checked={llmEnabled}
                onChange={(e) => setLlmEnabled(e.target.checked)}
                color="primary"
                disabled={loading}
              />
            }
            label="Enable LLM Enhancement"
          />
          
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, mb: 2 }}>
            When enabled, the system will use language models to analyze and enhance relationships between contexts.
          </Typography>
          
          {ollamaStatus === 'unavailable' && llmEnabled && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              Ollama is not available. LLM enhancement will be disabled.
            </Alert>
          )}
          
          {ollamaStatus === 'error' && llmEnabled && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Error checking Ollama status. LLM enhancement may not work.
            </Alert>
          )}
        </Box>
        
        <Divider sx={{ mb: 3 }} />
        
        <Typography variant="subtitle1" gutterBottom>
          Ollama Model Selection
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Select the Ollama model to use for LLM enhancement.
        </Typography>
        
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel id="ollama-model-label">Ollama Model</InputLabel>
          <Select
            labelId="ollama-model-label"
            value={ollamaModel}
            onChange={(e) => setOllamaModel(e.target.value)}
            label="Ollama Model"
            disabled={loading || !llmEnabled || ollamaStatus !== 'available'}
          >
            {availableModels.map((model) => (
              <MenuItem key={model} value={model}>
                {model}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        
        <Button
          variant="contained"
          color="primary"
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <RestartAltIcon />}
          onClick={handleReinitialize}
          disabled={loading}
          sx={{ mb: 2 }}
        >
          {loading ? 'Reinitializing...' : 'Reinitialize Knowledge System'}
        </Button>
        
        <Typography variant="body2" color="text.secondary">
          Reinitializing will apply the new settings and restart the knowledge system.
        </Typography>
      </Paper>
      
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          System Information
        </Typography>
        
        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Ollama Status
            </Typography>
            <Typography variant="body2" color={
              ollamaStatus === 'available' ? 'success.main' :
              ollamaStatus === 'checking' ? 'info.main' :
              'error.main'
            }>
              {ollamaStatus === 'available' ? 'Available' :
               ollamaStatus === 'checking' ? 'Checking...' :
               ollamaStatus === 'unavailable' ? 'Unavailable' :
               'Error checking status'}
            </Typography>
            
            <Button
              variant="outlined"
              size="small"
              onClick={checkOllamaStatus}
              disabled={ollamaStatus === 'checking'}
              sx={{ mt: 1 }}
            >
              Refresh Status
            </Button>
          </CardContent>
        </Card>
        
        <Typography variant="subtitle1" gutterBottom>
          About
        </Typography>
        
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Adaptive Compressed World Model Framework
        </Typography>
        
        <Typography variant="body2" color="text.secondary">
          A system for compressing, storing, and expanding knowledge representations with adaptive context linking and LLM enhancement.
        </Typography>
      </Paper>
    </Box>
  );
};

export default SettingsPage;
