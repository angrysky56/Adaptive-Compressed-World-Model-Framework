'use client';

import React, { useState, useCallback } from 'react';
import { 
  Box, Typography, Paper, TextField, Button, 
  CircularProgress, Chip, InputAdornment, IconButton,
  Card, CardContent, Divider, Alert, Tab, Tabs
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import DescriptionIcon from '@mui/icons-material/Description';
import TextFieldsIcon from '@mui/icons-material/TextFields';
import DeleteIcon from '@mui/icons-material/Delete';
import { addKnowledge, uploadFile } from '@/lib/api';

/**
 * Common stopwords that should be excluded from entity extraction
 */
const STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
  'through', 'during', 'before', 'after', 'above', 'below', 'from',
  'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
  'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
  'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
  'now', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'this',
  'that', 'these', 'those', 'am', 'it', 'its', 'they', 'them', 'their',
  'what', 'which', 'who', 'whom', 'would', 'could', 'should', 'shall'
]);

/**
 * Interface for component props
 */
interface KnowledgeFormProps {
  showNotification?: (message: string, severity?: 'success' | 'info' | 'warning' | 'error') => void;
}

/**
 * KnowledgeForm component
 * Form for adding knowledge to the system via text or file upload
 * 
 * @param showNotification - Function to show notifications
 * @returns The knowledge form component
 */
const KnowledgeForm: React.FC<KnowledgeFormProps> = ({ showNotification }) => {
  // Input mode: text or file
  const [inputMode, setInputMode] = useState<'text' | 'file'>('text');
  
  // Text input state
  const [text, setText] = useState<string>('');
  
  // File input state
  const [file, setFile] = useState<File | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [filePreview, setFilePreview] = useState<string>('');
  
  // Entity management
  const [entity, setEntity] = useState<string>('');
  const [entities, setEntities] = useState<string[]>([]);
  
  // UI state
  const [loading, setLoading] = useState<boolean>(false);
  const [success, setSuccess] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);
  
  /**
   * Handle text input change
   */
  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setText(e.target.value);
  };
  
  /**
   * Handle file input change
   */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;
    
    const selectedFile = fileList[0];
    setFile(selectedFile);
    
    // Create a preview of the file
    const fileReader = new FileReader();
    
    fileReader.onload = (event) => {
      const content = event.target?.result as string || '';
      setFileContent(content);
      
      // Limit preview to first 1000 characters
      const preview = content.slice(0, 1000) + (content.length > 1000 ? '...' : '');
      setFilePreview(preview);
    };
    
    fileReader.readAsText(selectedFile);
  };
  
  /**
   * Clear selected file
   */
  const clearFile = () => {
    setFile(null);
    setFileContent('');
    setFilePreview('');
  };
  
  /**
   * Handle entity input change
   */
  const handleEntityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEntity(e.target.value);
  };
  
  /**
   * Add entity to the list
   */
  const addEntity = () => {
    if (entity.trim() !== '' && !entities.includes(entity.trim())) {
      setEntities([...entities, entity.trim()]);
      setEntity('');
    }
  };
  
  /**
   * Add entity when Enter key is pressed
   */
  const handleEntityKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addEntity();
    }
  };
  
  /**
   * Remove entity from the list
   */
  const removeEntity = (entityToRemove: string) => {
    setEntities(entities.filter((item) => item !== entityToRemove));
  };
  
  /**
   * Extract potential entities from text
   */
  const extractEntities = useCallback(() => {
    // Get the right text source based on input mode
    const sourceText = inputMode === 'text' ? text : fileContent;
    
    if (!sourceText) {
      if (showNotification) {
        showNotification('No text to extract entities from', 'warning');
      }
      return;
    }
    
    // More sophisticated extraction with stopwords filtering
    const words = sourceText.toLowerCase().match(/\b[a-z][a-z0-9]*\b/g) || [];
    const wordCounts: {[key: string]: number} = {};
    
    // Count word occurrences, excluding stopwords and short words
    words.forEach((word) => {
      if (!STOPWORDS.has(word) && word.length > 3) {
        wordCounts[word] = (wordCounts[word] || 0) + 1;
      }
    });
    
    // Find capitalized phrases which are likely to be named entities
    const namedEntities = new Set<string>();
    const capitalizedRegex = /\b([A-Z][a-z]+ )+[A-Z][a-z]+\b|[A-Z][a-z]+\b/g;
    const matches = sourceText.match(capitalizedRegex) || [];
    
    matches.forEach(match => {
      namedEntities.add(match.toLowerCase());
    });
    
    // Convert to array and sort by frequency, favor named entities
    let sortedWords = Object.entries(wordCounts)
      .sort((a, b) => {
        // Prioritize named entities
        const aIsNamed = namedEntities.has(a[0]);
        const bIsNamed = namedEntities.has(b[0]);
        
        if (aIsNamed && !bIsNamed) return -1;
        if (!aIsNamed && bIsNamed) return 1;
        
        // Then sort by frequency
        return b[1] - a[1];
      })
      .slice(0, 15)
      .map(([word]) => word);
    
    // Add only words that aren't already in the entities list
    const newEntities = [...new Set([...entities, ...sortedWords])];
    setEntities(newEntities);
    
    if (showNotification) {
      showNotification('Entities extracted successfully', 'success');
    }
  }, [text, fileContent, entities, showNotification, inputMode]);
  
  /**
   * Submit the knowledge to the API
   */
  const handleSubmit = async () => {
    try {
      setLoading(true);
      setSuccess(false);
      setResult(null);
      
      let response;
      
      // Different handling based on input mode
      if (inputMode === 'text') {
        // For text input, use standard JSON API
        if (!text.trim()) {
          if (showNotification) {
            showNotification('Please enter some text', 'error');
          }
          setLoading(false);
          return;
        }
        
        response = await addKnowledge(text, entities, 'manual_input');
      } else {
        // For file input, use FormData and file upload API
        if (!file) {
          if (showNotification) {
            showNotification('Please select a file', 'error');
          }
          setLoading(false);
          return;
        }
        
        response = await uploadFile(file, entities.join(','));
      }
      
      if (response.success) {
        setSuccess(true);
        setResult(response);
        
        // If entities were extracted automatically, update the UI
        if (response.entities_extracted && response.entities_extracted.length > 0) {
          setEntities(response.entities_extracted);
        }
        
        if (showNotification) {
          showNotification('Knowledge added successfully', 'success');
        }
      } else {
        if (showNotification) {
          showNotification(response.message || 'Failed to add knowledge', 'error');
        }
      }
    } catch (error) {
      console.error('Error adding knowledge:', error);
      if (showNotification) {
        showNotification('Failed to add knowledge', 'error');
      }
    } finally {
      setLoading(false);
    }
  };
  
  /**
   * Reset the form
   */
  const resetForm = () => {
    setText('');
    setFile(null);
    setFileContent('');
    setFilePreview('');
    setEntity('');
    setEntities([]);
    setSuccess(false);
    setResult(null);
  };
  
  /**
   * Content for text input mode
   */
  const renderTextInputMode = () => (
    <>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Enter the text you want to add to the knowledge system. The system will compress this text while preserving the most important information.
      </Typography>
      
      <TextField
        fullWidth
        multiline
        rows={6}
        variant="outlined"
        placeholder="Enter knowledge text here..."
        value={text}
        onChange={handleTextChange}
        disabled={loading}
        sx={{ mb: 3 }}
      />
    </>
  );
  
  /**
   * Content for file input mode
   */
  const renderFileInputMode = () => (
    <>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Upload a text file (.txt, .md, .csv, etc.) to add to the knowledge system. The system will extract and compress the content while preserving the most important information.
      </Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Button
          variant="contained"
          component="label"
          startIcon={<FileUploadIcon />}
          disabled={loading || file !== null}
        >
          Upload File
          <input
            type="file"
            hidden
            accept=".txt,.md,.csv,.json,.xml,.html,.pdf,.xlsx,.xls,.docx,.doc"
            onChange={handleFileChange}
          />
        </Button>
        
        {file && (
          <Button
            variant="outlined"
            color="error"
            startIcon={<DeleteIcon />}
            onClick={clearFile}
            sx={{ ml: 2 }}
            disabled={loading}
          >
            Clear
          </Button>
        )}
      </Box>
      
      {file && (
        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="subtitle2" gutterBottom>
              {file.name} ({Math.round(file.size / 1024)} KB)
            </Typography>
            
            <Divider sx={{ my: 1 }} />
            
            <Typography variant="body2" sx={{ 
              whiteSpace: 'pre-wrap',
              maxHeight: '200px',
              overflow: 'auto',
              backgroundColor: '#f5f5f5',
              p: 1,
              borderRadius: 1,
              fontSize: '0.8rem'
            }}>
              {filePreview}
            </Typography>
          </CardContent>
        </Card>
      )}
      
      {!file && (
        <Alert severity="info" sx={{ mb: 3 }}>
          No file selected. Please upload a file to continue.
        </Alert>
      )}
    </>
  );
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Add Knowledge
      </Typography>
      
      <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
        <Tabs 
          value={inputMode} 
          onChange={(_, value) => setInputMode(value as 'text' | 'file')}
          sx={{ mb: 3 }}
        >
          <Tab 
            value="text" 
            label="Text Input" 
            icon={<TextFieldsIcon />} 
            iconPosition="start"
          />
          <Tab 
            value="file" 
            label="File Upload" 
            icon={<DescriptionIcon />} 
            iconPosition="start"
          />
        </Tabs>
        
        <Typography variant="h6" gutterBottom>
          Knowledge Content
        </Typography>
        
        {inputMode === 'text' ? renderTextInputMode() : renderFileInputMode()}
        
        <Typography variant="h6" gutterBottom>
          Critical Entities
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Specify important entities, concepts, or terms that should be preserved during compression.
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <TextField
            fullWidth
            variant="outlined"
            size="small"
            placeholder="Add an entity..."
            value={entity}
            onChange={handleEntityChange}
            onKeyPress={handleEntityKeyPress}
            disabled={loading}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={addEntity}
                    disabled={entity.trim() === '' || loading}
                    edge="end"
                  >
                    <AddIcon />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          <Button
            variant="outlined"
            color="secondary"
            onClick={extractEntities}
            disabled={(inputMode === 'text' && text.trim() === '') || 
                     (inputMode === 'file' && !fileContent) || 
                     loading}
            sx={{ ml: 2, whiteSpace: 'nowrap' }}
          >
            Extract Entities
          </Button>
        </Box>
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', mb: 3 }}>
          {entities.map((item) => (
            <Chip
              key={item}
              label={item}
              onDelete={() => removeEntity(item)}
              disabled={loading}
              sx={{ m: 0.5 }}
            />
          ))}
          {entities.length === 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ pl: 1 }}>
              No entities added. The system will identify important entities automatically.
            </Typography>
          )}
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Button
            variant="outlined"
            color="secondary"
            onClick={resetForm}
            disabled={loading || 
                     (inputMode === 'text' && !text && !entities.length) ||
                     (inputMode === 'file' && !file && !entities.length)}
          >
            Reset
          </Button>
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            disabled={loading || 
                     (inputMode === 'text' && !text.trim()) ||
                     (inputMode === 'file' && !fileContent)}
            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : undefined}
          >
            {loading ? 'Adding...' : 'Add Knowledge'}
          </Button>
        </Box>
      </Paper>
      
      {success && result && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" color="primary" gutterBottom>
              Knowledge Added Successfully
            </Typography>
            
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Context ID:</strong> {result.context_id}
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="body2" color="text.secondary">
              You can now query this knowledge or view it in the knowledge graph.
            </Typography>
            
            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
              <Button 
                variant="outlined" 
                color="primary" 
                size="small"
                onClick={() => resetForm()}
              >
                Add More Knowledge
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default KnowledgeForm;
