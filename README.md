# Deepfake Detection Application

This application consists of a React frontend and FastAPI backend for detecting deepfake images.

## Project Structure

```
├── src/                  # Frontend React application
│   ├── components/       # React components
│   ├── services/        # API services
│   └── config.ts        # Configuration
├── backend/             # FastAPI backend
│   ├── app/
│   │   ├── models/      # ML models and Pydantic models
│   │   ├── routers/     # API routes
│   │   └── utils/       # Utility functions
│   ├── main.py         # FastAPI application entry
│   └── requirements.txt # Python dependencies
```

## Setup Instructions

1. Frontend Setup:
   ```bash
   # Install dependencies
   npm install

   # Start development server
   npm run dev
   ```

2. Backend Setup:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r backend/requirements.txt

   # Start backend server
   python backend/main.py
   ```

3. Add your ML models:
   - Place your CNN model in `backend/app/models/cnn_model/`
   - Place your Xception model in `backend/app/models/xception_model/`

## Integration Points

1. Frontend API Integration:
   - API configuration in `src/config.ts`
   - API services in `src/services/api.ts`

2. Backend Integration:
   - Add your model inference code in `backend/app/models/`
   - Configure routes in `backend/app/routers/`
   - Add preprocessing in `backend/app/utils/`

## Development

- Frontend runs on: http://localhost:5173
- Backend runs on: http://localhost:8000
- API documentation: http://localhost:8000/docs