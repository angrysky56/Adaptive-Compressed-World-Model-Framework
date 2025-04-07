'use client';

import React from 'react';
import AppWrapper from '../AppWrapper';
import Communities from '@/components/Communities';

/**
 * Communities page component
 * Displays the knowledge communities and gap analysis
 * 
 * @returns The communities page component
 */
export default function CommunitiesPage() {
  return (
    <AppWrapper>
      <Communities />
    </AppWrapper>
  );
}
