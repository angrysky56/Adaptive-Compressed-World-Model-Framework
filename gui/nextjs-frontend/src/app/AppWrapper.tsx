'use client';

import React from 'react';
import { AppContextProvider } from '@/contexts/AppContext';
import { CircularProgress, Box, Typography, Button } from '@mui/material';
import MainLayout from '@/components/MainLayout';
import { useAppContext } from '@/contexts/AppContext';

/**
 * Props for the AppContent component
 */
interface AppContentProps {
  children: React.ReactNode;
}

/**
 * The main content component for the application
 * Shows loading state or initialization screen based on app status
 * 
 * @param children - The child components to render when initialized
 * @returns The appropriate UI based on initialization status
 */
function AppContent({ children }: AppContentProps) {
  const { initialized, initializing, llmEnabled, llmAvailable, initializeSystem } = useAppContext();

  if (initializing) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
        <CircularProgress />
        <Typography sx={{ mt: 2 }}>Initializing Knowledge System...</Typography>
      </Box>
    );
  }

  if (!initialized) {
    return (
      <MainLayout>
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
      </MainLayout>
    );
  }

  return (
    <MainLayout llmEnabled={llmEnabled} llmAvailable={llmAvailable}>
      {children}
    </MainLayout>
  );
}

/**
 * The application wrapper component
 * Provides context and layout for the entire application
 * 
 * @param children - The child components to render
 * @returns The wrapped application with context and layout
 */
export default function AppWrapper({ children }: { children: React.ReactNode }) {
  return (
    <AppContextProvider>
      <AppContent>{children}</AppContent>
    </AppContextProvider>
  );
}
