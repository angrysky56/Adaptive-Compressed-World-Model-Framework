import type { NextConfig } from "next";

/**
 * Next.js configuration
 * 
 * This configuration includes:
 * - Async rewrites to proxy API requests to the Flask backend
 * - Any other Next.js configuration options
 */
const nextConfig: NextConfig = {
  // Proxy API requests to the Flask backend
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/api/:path*', // Proxy to Flask backend
      },
    ];
  },
  
  // Enable React strict mode for development
  reactStrictMode: true,
  
  // Configure allowed image domains if needed
  images: {
    domains: ['localhost'],
  },
};

export default nextConfig;
