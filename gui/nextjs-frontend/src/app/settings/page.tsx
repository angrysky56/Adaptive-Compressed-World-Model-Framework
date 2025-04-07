'use client';

import React from 'react';
import AppWrapper from '../AppWrapper';
import SettingsPage from '@/components/SettingsPage';

/**
 * Settings page component
 * Displays the system settings interface
 * 
 * @returns The settings page component
 */
export default function Settings() {
  return (
    <AppWrapper>
      <SettingsPage />
    </AppWrapper>
  );
}
