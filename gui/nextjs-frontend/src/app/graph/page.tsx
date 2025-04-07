'use client';

import React from 'react';
import AppWrapper from '../AppWrapper';
import KnowledgeGraph from '@/components/KnowledgeGraph';

/**
 * Graph page component
 * Displays the knowledge graph visualization
 * 
 * @returns The graph page component
 */
export default function GraphPage() {
  return (
    <AppWrapper>
      <KnowledgeGraph />
    </AppWrapper>
  );
}
