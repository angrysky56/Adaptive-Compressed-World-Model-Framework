'use client';

import React from 'react';
import AppWrapper from '../AppWrapper';
import Relationships from '@/components/Relationships';

/**
 * Relationships page component
 * Displays the relationships between knowledge contexts
 * 
 * @returns The relationships page component
 */
export default function RelationshipsPage() {
  return (
    <AppWrapper>
      <Relationships />
    </AppWrapper>
  );
}
