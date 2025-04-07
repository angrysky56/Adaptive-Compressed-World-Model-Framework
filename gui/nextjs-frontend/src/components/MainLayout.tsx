'use client';

import React, { ReactNode } from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Container, Snackbar, Alert } from '@mui/material';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';

/**
 * Interface for the MainLayout component props
 */
interface MainLayoutProps {
  children: ReactNode;
  llmEnabled?: boolean;
  llmAvailable?: boolean;
}

/**
 * Interface for notification state
 */
interface NotificationState {
  open: boolean;
  message: string;
  severity: 'success' | 'info' | 'warning' | 'error';
}

/**
 * MainLayout component
 * Provides the main application layout with app bar, navigation, and footer
 * 
 * @param children - The child components to render in the main content area
 * @param llmEnabled - Whether LLM enhancement is enabled
 * @param llmAvailable - Whether LLM is available
 * @returns The layout component with navigation and content
 */
export default function MainLayout({ 
  children, 
  llmEnabled = false, 
  llmAvailable = false 
}: MainLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  const [notification, setNotification] = React.useState<NotificationState>({ 
    open: false, 
    message: '', 
    severity: 'info' 
  });

  /**
   * Show a notification message
   * 
   * @param message - The message to display
   * @param severity - The severity level of the message
   */
  const showNotification = (message: string, severity: NotificationState['severity'] = 'info') => {
    setNotification({
      open: true,
      message,
      severity,
    });
  };

  /**
   * Handle closing the notification
   */
  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  /**
   * Navigation items with their paths
   */
  const navItems = [
    { label: 'Home', path: '/' },
    { label: 'Graph', path: '/graph' },
    { label: 'Add Knowledge', path: '/add' },
    { label: 'Query', path: '/query' },
    { label: 'Relationships', path: '/relationships' },
    { label: 'Communities', path: '/communities' },
    { label: 'Data', path: '/data' },
    { label: 'Settings', path: '/settings' },
  ];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static">
        <Toolbar>
          <Typography
            variant="h6"
            component="div"
            sx={{ flexGrow: 1, cursor: 'pointer' }}
            onClick={() => router.push('/')}
          >
            Adaptive Compressed World Model Framework
          </Typography>
          
          {navItems.map((item) => (
            <Link key={item.path} href={item.path} passHref>
              <Button 
                color="inherit" 
                sx={{ 
                  color: pathname === item.path ? '#fff' : 'rgba(255, 255, 255, 0.7)',
                  fontWeight: pathname === item.path ? 'bold' : 'normal',
                }}
              >
                {item.label}
              </Button>
            </Link>
          ))}
        </Toolbar>
      </AppBar>

      <Container sx={{ flexGrow: 1, mt: 4, mb: 4 }}>
        {React.cloneElement(children as React.ReactElement, { showNotification })}
      </Container>

      <Box component="footer" sx={{ p: 2, mt: 'auto', backgroundColor: 'background.paper' }}>
        <Typography variant="body2" color="text.secondary" align="center">
          Adaptive Compressed World Model Framework - {llmEnabled && llmAvailable ? 'LLM Enhanced' : 'Standard'} Mode
        </Typography>
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
    </Box>
  );
}
