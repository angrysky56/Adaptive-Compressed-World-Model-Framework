# Adaptive Compressed World Model Framework - Next.js Frontend

This is the Next.js frontend for the Adaptive Compressed World Model Framework. It provides a modern, efficient, and type-safe interface for interacting with the knowledge system.

## Features

- **Server-Side Rendering**: Improved initial load performance and SEO
- **TypeScript**: Type safety and better developer experience
- **API Routes**: Built-in API proxy to the Flask backend
- **File-Based Routing**: Simplified navigation structure
- **Material UI**: Modern UI components
- **Responsive Design**: Works well on desktop and mobile devices

## Directory Structure

```
nextjs-frontend/
├── public/                # Static files
├── src/
│   ├── app/               # Next.js 13+ app directory (file-based routing)
│   │   ├── add/           # Add knowledge page
│   │   ├── communities/   # Communities page 
│   │   ├── graph/         # Graph visualization page
│   │   ├── query/         # Query knowledge page
│   │   ├── relationships/ # Relationships page
│   │   ├── settings/      # Settings page
│   │   ├── AppWrapper.tsx # Application wrapper with context providers
│   │   ├── layout.tsx     # Root layout
│   │   ├── page.tsx       # Home page
│   ├── components/        # Reusable components
│   ├── contexts/          # React context providers
│   ├── lib/               # Utility functions and API client
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Running backend server (Flask)

### Installation

1. Install dependencies:

```bash
cd gui/nextjs-frontend
npm install
```

### Development

Run the development server:

```bash
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000).

### Building for Production

Build the application:

```bash
npm run build
```

Start the production server:

```bash
npm start
```

## Environment Variables

Create a `.env.local` file to customize the configuration:

```
# API URL (default is http://localhost:5000)
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## Migrating from Create React App

This Next.js frontend replaces the previous Create React App (CRA) implementation. Key improvements include:

1. **Performance**: Server-side rendering and static generation for faster initial load
2. **API Integration**: Built-in API routes and proxy to the backend
3. **TypeScript**: Improved type safety and developer experience
4. **Modern React**: Uses React 19 with the new app directory structure

## Backend Integration

The Next.js frontend connects to the Flask backend via API proxy. All API requests are forwarded to the Flask server running on port 5000.
