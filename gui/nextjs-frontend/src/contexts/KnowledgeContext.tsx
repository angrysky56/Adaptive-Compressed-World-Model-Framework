'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { getAllContexts, deleteKnowledge } from '@/lib/api';
import { useAppContext } from './AppContext';

/**
 * Interface for a knowledge context
 */
export interface KnowledgeContext {
  id: string;
  summary: string;
  critical_entities: string[];
  creation_time: number;
  access_count: number;
  last_accessed: number;
  file_info?: {
    size: number;
    created: number;
    modified: number;
  };
  source: string;
}

/**
 * Interface for the storage information
 */
export interface StorageInfo {
  data_directory: string;
  total_contexts: number;
  total_size: number;
}

/**
 * Interface for the knowledge context state
 */
interface KnowledgeContextState {
  contexts: KnowledgeContext[];
  storageInfo: StorageInfo | null;
  loading: boolean;
  error: string | null;
  lastUpdated: number;
  refreshData: () => Promise<void>;
  deleteContext: (contextId: string) => Promise<boolean>;
}

/**
 * Create the knowledge context with default values
 */
const KnowledgeContext = createContext<KnowledgeContextState>({
  contexts: [],
  storageInfo: null,
  loading: false,
  error: null,
  lastUpdated: 0,
  refreshData: async () => {},
  deleteContext: async () => false,
});

/**
 * Custom hook to use the knowledge context
 * 
 * @returns The knowledge context state
 */
export const useKnowledgeContext = () => useContext(KnowledgeContext);

/**
 * Props for the KnowledgeContextProvider component
 */
interface KnowledgeContextProviderProps {
  children: ReactNode;
}

/**
 * KnowledgeContextProvider component
 * Provides the knowledge context state to all child components
 * 
 * @param children - The child components to wrap with the provider
 * @returns The provider component with children
 */
export function KnowledgeContextProvider({ children }: KnowledgeContextProviderProps) {
  const { initialized, showNotification } = useAppContext();
  const [contexts, setContexts] = useState<KnowledgeContext[]>([]);
  const [storageInfo, setStorageInfo] = useState<StorageInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number>(0);

  /**
   * Refresh the knowledge data
   */
  const refreshData = async () => {
    if (!initialized) {
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await getAllContexts();
      
      if (response.success) {
        setContexts(response.contexts || []);
        setStorageInfo(response.storage_info || null);
        setLastUpdated(Date.now());
      } else {
        setError(response.message || 'Failed to load contexts');
        if (showNotification) {
          showNotification('Failed to load knowledge contexts', 'error');
        }
      }
    } catch (error) {
      console.error('Error loading contexts:', error);
      setError('Failed to load contexts');
      if (showNotification) {
        showNotification('Error loading knowledge contexts', 'error');
      }
    } finally {
      setLoading(false);
    }
  };

  /**
   * Delete a knowledge context
   * 
   * @param contextId - ID of the context to delete
   * @returns Promise that resolves to true if deletion was successful
   */
  const deleteContext = async (contextId: string): Promise<boolean> => {
    try {
      setLoading(true);
      
      const response = await deleteKnowledge(contextId);
      
      if (response.success) {
        // Update local state to remove the deleted context
        setContexts(prevContexts => prevContexts.filter(
          context => context.id !== contextId
        ));
        
        // Update last updated timestamp
        setLastUpdated(Date.now());
        
        if (showNotification) {
          showNotification(`Successfully deleted context ${contextId}`, 'success');
        }
        
        // Refresh data to ensure consistent state
        await refreshData();
        return true;
      } else {
        if (showNotification) {
          showNotification(`Failed to delete context: ${response.message}`, 'error');
        }
        return false;
      }
    } catch (error) {
      console.error('Error deleting context:', error);
      if (showNotification) {
        showNotification('Error deleting context', 'error');
      }
      return false;
    } finally {
      setLoading(false);
    }
  };

  // Fetch data when the component mounts and when the system is initialized
  useEffect(() => {
    if (initialized) {
      refreshData();
    }
  }, [initialized]);

  const contextValue: KnowledgeContextState = {
    contexts,
    storageInfo,
    loading,
    error,
    lastUpdated,
    refreshData,
    deleteContext
  };

  return (
    <KnowledgeContext.Provider value={contextValue}>
      {children}
    </KnowledgeContext.Provider>
  );
}
