'use client';

import React from 'react';
import AppWrapper from '../AppWrapper';
import KnowledgeForm from '@/components/KnowledgeForm';

/**
 * Add knowledge page component
 * Displays the form for adding knowledge to the system
 * 
 * @returns The add knowledge page component
 */
export default function AddKnowledgePage() {
  return (
    <AppWrapper>
      <KnowledgeForm />
    </AppWrapper>
  );
}
