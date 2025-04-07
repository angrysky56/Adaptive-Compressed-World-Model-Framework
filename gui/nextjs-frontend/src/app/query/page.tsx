'use client';

import React from 'react';
import AppWrapper from '../AppWrapper';
import QueryKnowledge from '@/components/QueryKnowledge';

/**
 * Query page component
 * Displays the knowledge query interface
 * 
 * @returns The query page component
 */
export default function QueryPage() {
  return (
    <AppWrapper>
      <QueryKnowledge />
    </AppWrapper>
  );
}
