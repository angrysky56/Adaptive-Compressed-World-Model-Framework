import React from 'react';
import AppWrapper from './AppWrapper';
import HomePage from '@/components/HomePage';

/**
 * Main page component
 * Wraps the HomePage component with the AppWrapper
 * 
 * @returns The main page component
 */
export default function Home() {
  return (
    <AppWrapper>
      <HomePage />
    </AppWrapper>
  );
}
