import React, { useState, useEffect } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import { 
  AppBar, Toolbar, Typography, Container, Button, Box,
  CircularProgress, Snackbar, Alert, CssBaseline
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import axios from 'axios';

// Import components
import HomePage from './components/HomePage';
import KnowledgeGraph from './components/KnowledgeGraph';
import KnowledgeForm from './components/KnowledgeForm';
import QueryKnowledge from './components/QueryKnowledge';
import Relationships from './components/Relationships';
import Communities from './components/Communities';
import SettingsPage from './components/SettingsPage';

// Define the theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f8f9fa',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
        },
      },
    },
  },
});

function App() {
  const [initializing, setInitializing] = useState(true);
  const [initialized, setInitialized] = useState(false);
  const [llmEnabled, setLlmEnabled] = useState(true);
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  const navigate = useNavigate();

  // Check if the knowledge system is initialized
  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    try {
      const response = await axios.get('/api/status');
      if (response.data.status === 'initialized') {
        setInitialized(true);
        setLlmEnabled(response.data.llm_enabled);
        setLlmAvailable(response.data.llm_available);
      } else {
        setInitialized(false);
      }
    } catch (error) {
      console.error('Error checking status:', error);
      setInitialized(false);
    } finally {
      setInitializing(false);
    }
  };

  const initializeSystem = async (useLlm = true, ollamaModel = "mistral-nemo:latest") => {
    try {
      setInitializing(true);
      const response = await axios.post('/api/init', { use_llm: useLlm, ollama_model: ollamaModel });
      if (response.data.success) {
        setInitialized(true);
        setLlmEnabled(useLlm);
        showNotification(`Knowledge system initialized ${useLlm ? 'with' : 'without'} LLM enhancement`, 'success');
        await checkStatus(); // Refresh status to check if LLM is available
      } else {
        showNotification('Failed to initialize knowledge system', 'error');
      }
    } catch (error) {
      console.error('Error initializing system:', error);
      showNotification(`Error: ${error.response?.data?.message || error.message}`, 'error');
    } finally {
      setInitializing(false);
    }
  };

  const showNotification = (message, severity = 'info') => {
    setNotification({
      open: true,
      message,
      severity,
    });
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  if (initializing) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
          <CircularProgress />
          <Typography sx={{ mt: 2 }}>Initializing Knowledge System...</Typography>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, cursor: 'pointer' }} onClick={() => navigate('/')}>
              Adaptive Compressed World Model Framework
            </Typography>
            <Button color="inherit" onClick={() => navigate('/')}>Home</Button>
            <Button color="inherit" onClick={() => navigate('/graph')}>Graph</Button>
            <Button color="inherit" onClick={() => navigate('/add')}>Add Knowledge</Button>
            <Button color="inherit" onClick={() => navigate('/query')}>Query</Button>
            <Button color="inherit" onClick={() => navigate('/relationships')}>Relationships</Button>
            <Button color="inherit" onClick={() => navigate('/communities')}>Communities</Button>
            <Button color="inherit" onClick={() => navigate('/settings')}>Settings</Button>
          </Toolbar>
        </AppBar>

        <Container sx={{ flexGrow: 1, mt: 4, mb: 4 }}>
          {!initialized ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 8 }}>
              <Typography variant="h4" gutterBottom>
                Knowledge System Not Initialized
              </Typography>
              <Typography variant="body1" gutterBottom sx={{ mb: 4 }}>
                The knowledge system needs to be initialized before use.
              </Typography>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={() => initializeSystem(true)}
                  disabled={initializing}
                >
                  Initialize with LLM Enhancement
                </Button>
                <Button 
                  variant="outlined" 
                  onClick={() => initializeSystem(false)}
                  disabled={initializing}
                >
                  Initialize without LLM
                </Button>
              </Box>
            </Box>
          ) : (
            <Routes>
              <Route path="/" element={<HomePage llmEnabled={llmEnabled} llmAvailable={llmAvailable} />} />
              <Route path="/graph" element={<KnowledgeGraph showNotification={showNotification} />} />
              <Route 
                path="/add" 
                element={
                  <KnowledgeForm 
                    showNotification={showNotification} 
                  />
                } 
              />
              <Route path="/query" element={<QueryKnowledge showNotification={showNotification} />} />
              <Route 
                path="/relationships" 
                element={
                  <Relationships 
                    showNotification={showNotification} 
                    llmEnabled={llmEnabled}
                    llmAvailable={llmAvailable}
                  />
                } 
              />
              <Route 
                path="/communities" 
                element={
                  <Communities 
                    showNotification={showNotification} 
                    llmEnabled={llmEnabled} 
                  />
                } 
              />
              <Route 
                path="/settings" 
                element={
                  <SettingsPage 
                    showNotification={showNotification}
                    llmEnabled={llmEnabled}
                    setLlmEnabled={setLlmEnabled}
                    initializeSystem={initializeSystem}
                  />
                } 
              />
            </Routes>
          )}
        </Container>

        <Box component="footer" sx={{ p: 2, mt: 'auto', backgroundColor: 'background.paper' }}>
          <Typography variant="body2" color="text.secondary" align="center">
            Adaptive Compressed World Model Framework - {llmEnabled && llmAvailable ? 'LLM Enhanced' : 'Standard'} Mode
          </Typography>
        </Box>
      </Box>

      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: '100%' }}>
          {notification.message}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;
