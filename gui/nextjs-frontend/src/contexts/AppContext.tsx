'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { getStatus, initializeSystem } from '@/lib/api';

/**
 * Interface for the application context state
 */
interface AppContextState {
  initialized: boolean;
  initializing: boolean;
  llmEnabled: boolean;
  llmAvailable: boolean;
  stats: any;
  error: string | null;
  initializeSystem: (useLlm?: boolean, ollamaModel?: string) => Promise<void>;
  setLlmEnabled: (enabled: boolean) => void;
  showNotification?: (message: string, severity?: 'success' | 'info' | 'warning' | 'error') => void;
}

/**
 * Create the application context with default values
 */
const AppContext = createContext<AppContextState>({
  initialized: false,
  initializing: true,
  llmEnabled: false,
  llmAvailable: false,
  stats: null,
  error: null,
  initializeSystem: async () => {},
  setLlmEnabled: () => {},
});

/**
 * Custom hook to use the app context
 * 
 * @returns The app context state
 */
export const useAppContext = () => useContext(AppContext);

/**
 * Props for the AppContextProvider component
 */
interface AppContextProviderProps {
  children: ReactNode;
}

/**
 * AppContextProvider component
 * Provides the global application state to all child components
 * 
 * @param children - The child components to wrap with the provider
 * @returns The provider component with children
 */
export function AppContextProvider({ children }: AppContextProviderProps) {
  const [initialized, setInitialized] = useState<boolean>(false);
  const [initializing, setInitializing] = useState<boolean>(true);
  const [llmEnabled, setLlmEnabled] = useState<boolean>(false);
  const [llmAvailable, setLlmAvailable] = useState<boolean>(false);
  const [stats, setStats] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  /**
   * Check the system status
   */
  const checkStatus = async () => {
    try {
      const response = await getStatus();
      
      if (response.status === 'initialized') {
        setInitialized(true);
        setLlmEnabled(response.llm_enabled || false);
        setLlmAvailable(response.llm_available || false);
        setStats(response.stats || null);
      } else {
        setInitialized(false);
      }
    } catch (err) {
      console.error('Error checking status:', err);
      setInitialized(false);
      setError('Failed to connect to the knowledge system');
    } finally {
      setInitializing(false);
    }
  };

  /**
   * Initialize the knowledge system
   * 
   * @param useLlm - Whether to use LLM enhancement
   * @param ollamaModel - The Ollama model to use
   */
  const initialize = async (useLlm: boolean = true, ollamaModel: string = "mistral-nemo:latest") => {
    try {
      setInitializing(true);
      const response = await initializeSystem(useLlm, ollamaModel);
      
      if (response.success) {
        setInitialized(true);
        setLlmEnabled(useLlm);
        await checkStatus(); // Refresh status to get updated LLM availability
      } else {
        setError('Failed to initialize knowledge system');
      }
    } catch (err) {
      console.error('Error initializing system:', err);
      setError('Error connecting to the knowledge system');
    } finally {
      setInitializing(false);
    }
  };

  // Check status when the component mounts
  useEffect(() => {
    checkStatus();
  }, []);

  const contextValue: AppContextState = {
    initialized,
    initializing,
    llmEnabled,
    llmAvailable,
    stats,
    error,
    initializeSystem: initialize,
    setLlmEnabled,
  };

  return <AppContext.Provider value={contextValue}>{children}</AppContext.Provider>;
}
