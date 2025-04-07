import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import ThemeRegistry from '@/components/ThemeRegistry';

/**
 * Configure the default font for the application
 */
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

/**
 * Configure metadata for the application
 */
export const metadata: Metadata = {
  title: "Adaptive Compressed World Model Framework",
  description: "A framework for building and managing knowledge graphs with adaptive compression",
};

/**
 * Root layout component for the application
 * Wraps all pages with common structure and theme
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <ThemeRegistry>
          {children}
        </ThemeRegistry>
      </body>
    </html>
  );
}
