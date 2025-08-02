# Acausal Experiment Dashboard

A web-based dashboard for visualizing and analyzing acausal cooperation experiment results.

## Setup

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+ (for backend API)
- Virtual environment with project dependencies installed

### Environment Configuration

1. Copy `.env.example` to `.env` in the project root:
   ```bash
   cp ../.env.example ../.env
   ```

2. Edit `.env` and set the required variables:
   ```
   JWT_SECRET_KEY=your-secret-key-here  # Generate a strong random key
   ```

3. (Optional) Set frontend API URL if different from default:
   ```
   VITE_API_URL=http://localhost:8000/api/v1
   ```

### Installation

1. Install frontend dependencies:
   ```bash
   npm install
   ```

2. Install backend dependencies (if not already done):
   ```bash
   cd ..
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running the Dashboard

1. Start the backend API server:
   ```bash
   # From project root
   source venv/bin/activate
   export JWT_SECRET_KEY="your-secret-key"
   python -m uvicorn src.api.server:app --reload --port 8000
   ```

2. Start the frontend development server:
   ```bash
   # From dashboard directory
   npm run dev
   ```

3. Open your browser to http://localhost:5173

### Default Login

For development, use these credentials:
- Username: `admin`
- Password: `admin`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Testing

```bash
# Run unit tests
npm test

# Run tests in watch mode
npm run test:watch

# Run E2E tests (requires running server)
npm run test:e2e
```

## Architecture

- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + Python
- **Authentication**: JWT tokens
- **Real-time Updates**: WebSocket support
- **Data Storage**: Reads from existing JSON experiment files

## Features

- View list of experiments with metadata
- Responsive design for desktop and tablet
- Real-time updates via WebSocket
- JWT-based authentication
- Error handling and loading states
- Dark mode support