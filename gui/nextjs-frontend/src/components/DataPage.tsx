'use client';

import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Paper, CircularProgress, Alert,
  Card, CardContent, Divider, Chip, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Button,
  IconButton, Dialog, DialogTitle, DialogContent,
  DialogContentText, DialogActions, Tooltip
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import InfoIcon from '@mui/icons-material/Info';
import { getStatus, getAllContexts, deleteKnowledge } from '@/lib/api';

/**
 * Interface for component props
 */
interface DataPageProps {
  showNotification?: (message: string, severity?: 'success' | 'info' | 'warning' | 'error') => void;
}

/**
 * Interface for context data
 */
interface ContextData {
  id: string;
  summary: string;
  critical_entities: string[];
  creation_time: number;
  access_count: number;
  last_accessed: number;
  file_info: {
    size: number;
    created: number;
    modified: number;
  };
  source: string;
}

/**
 * DataPage component
 * Displays information about the stored knowledge contexts and data
 * 
 * @param showNotification - Function to show notifications
 * @returns The data visualization component
 */
const DataPage: React.FC<DataPageProps> = ({ showNotification }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [stats, setStats] = useState<any>(null);
  const [contexts, setContexts] = useState<ContextData[]>([]);
  const [storageInfo, setStorageInfo] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<boolean>(false);
  const [contextToDelete, setContextToDelete] = useState<string | null>(null);
  
  /**
   * Fetch system stats and data information
   */
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch general system status
      const statusResponse = await getStatus();
      if (statusResponse.status === 'initialized') {
        setStats(statusResponse.stats || {});
      } else {
        setError('Knowledge system not initialized');
      }
      
      // Fetch all contexts
      const contextsResponse = await getAllContexts();
      if (contextsResponse.success) {
        setContexts(contextsResponse.contexts || []);
        setStorageInfo(contextsResponse.storage_info || {});
      }
    } catch (error) {
      console.error('Error loading data:', error);
      setError('Failed to load data information');
      if (showNotification) {
        showNotification('Error loading data information', 'error');
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch data on component mount
  useEffect(() => {
    fetchData();
  }, []);
  
  /**
   * Format a timestamp as a readable date string
   */
  const formatTimestamp = (timestamp: number) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp * 1000).toLocaleString();
  };
  
  /**
   * Format a number with commas for better readability
   */
  const formatNumber = (num: number) => {
    if (num === undefined || num === null) return 'N/A';
    return num.toLocaleString();
  };
  
  /**
   * Format a percentage value
   */
  const formatPercentage = (value: number) => {
    if (value === undefined || value === null) return 'N/A';
    return (value * 100).toFixed(1) + '%';
  };
  
  /**
   * Format file size in a human-readable format
   */
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  /**
   * Open delete confirmation dialog
   */
  const confirmDelete = (contextId: string) => {
    setContextToDelete(contextId);
    setDeleteDialogOpen(true);
  };
  
  /**
   * Cancel deletion
   */
  const cancelDelete = () => {
    setContextToDelete(null);
    setDeleteDialogOpen(false);
  };
  
  /**
   * Execute deletion of a context
   */
  const executeDelete = async () => {
    if (!contextToDelete) return;
    
    try {
      const response = await deleteKnowledge(contextToDelete);
      
      if (response.success) {
        if (showNotification) {
          showNotification(`Successfully deleted context ${contextToDelete}`, 'success');
        }
        // Refresh data after deletion
        fetchData();
      } else {
        if (showNotification) {
          showNotification(`Failed to delete context: ${response.message}`, 'error');
        }
      }
    } catch (error) {
      console.error('Error deleting context:', error);
      if (showNotification) {
        showNotification('Error occurred during deletion', 'error');
      }
    } finally {
      setDeleteDialogOpen(false);
      setContextToDelete(null);
    }
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Knowledge Data
      </Typography>
      
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          View and manage knowledge contexts and data stored in the system.
        </Typography>
        
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />} 
          onClick={fetchData}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              System Overview
            </Typography>
            
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Metric</strong></TableCell>
                    <TableCell><strong>Value</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>Total Contexts</TableCell>
                    <TableCell>{formatNumber(contexts.length || 0)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Total Entities</TableCell>
                    <TableCell>{formatNumber(stats?.total_entities || 0)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Average Compression Ratio</TableCell>
                    <TableCell>{formatPercentage(stats?.avg_compression_ratio || 0)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Relationships</TableCell>
                    <TableCell>{formatNumber(stats?.graph_stats?.edge_count || 0)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Storage Used</TableCell>
                    <TableCell>{storageInfo ? formatFileSize(storageInfo.total_size || 0) : 'N/A'}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
          
          <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Knowledge Contexts ({contexts.length})
              </Typography>
            </Box>
            
            {contexts.length > 0 ? (
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>ID</strong></TableCell>
                      <TableCell><strong>Summary</strong></TableCell>
                      <TableCell><strong>Source</strong></TableCell>
                      <TableCell><strong>Created</strong></TableCell>
                      <TableCell><strong>Size</strong></TableCell>
                      <TableCell><strong>Actions</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {contexts.map((context) => (
                      <TableRow key={context.id} hover>
                        <TableCell>{context.id.substring(0, 8)}...</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                              {context.summary}
                            </Typography>
                            <Tooltip title={context.summary}>
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                            {context.critical_entities.slice(0, 3).map(entity => (
                              <Chip 
                                key={entity} 
                                label={entity} 
                                size="small" 
                                variant="outlined" 
                                sx={{ mr: 0.5, mb: 0.5 }}
                              />
                            ))}
                            {context.critical_entities.length > 3 && (
                              <Typography variant="body2" color="text.secondary">
                                +{context.critical_entities.length - 3} more
                              </Typography>
                            )}
                          </Box>
                        </TableCell>
                        <TableCell>{context.source || 'Unknown'}</TableCell>
                        <TableCell>{formatTimestamp(context.creation_time)}</TableCell>
                        <TableCell>
                          {context.file_info ? formatFileSize(context.file_info.size) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          <Tooltip title="Delete">
                            <IconButton 
                              edge="end" 
                              aria-label="delete" 
                              onClick={() => confirmDelete(context.id)}
                              color="error"
                              size="small"
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Alert severity="info">
                No knowledge contexts found. Add some knowledge to get started.
              </Alert>
            )}
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Supported File Types
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                <Chip label="TXT" size="small" />
                <Chip label="MD" size="small" />
                <Chip label="CSV" size="small" />
                <Chip label="JSON" size="small" />
                <Chip label="XML" size="small" />
                <Chip label="HTML" size="small" />
                <Chip label="PDF" size="small" />
                <Chip label="DOCX" size="small" />
                <Chip label="XLSX" size="small" />
              </Box>
            </Box>
          </Paper>
          
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Storage Information
            </Typography>
            
            {storageInfo ? (
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    Data Directory
                  </Typography>
                  <Typography variant="body2" component="pre" sx={{ 
                    bgcolor: '#f5f5f5', 
                    p: 1, 
                    borderRadius: 1,
                    overflow: 'auto'
                  }}>
                    {storageInfo.data_directory || 'Data directory information not available'}
                  </Typography>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      <strong>Total Storage Used:</strong> {formatFileSize(storageInfo.total_size || 0)}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Total Contexts:</strong> {formatNumber(storageInfo.total_contexts || 0)}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            ) : (
              <Alert severity="info">
                Storage information not available.
              </Alert>
            )}
          </Paper>
        </>
      )}
      
      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={cancelDelete}
        aria-labelledby="delete-dialog-title"
        aria-describedby="delete-dialog-description"
      >
        <DialogTitle id="delete-dialog-title">
          Confirm Deletion
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="delete-dialog-description">
            Are you sure you want to delete this knowledge context? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={cancelDelete} color="primary">
            Cancel
          </Button>
          <Button onClick={executeDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DataPage;
