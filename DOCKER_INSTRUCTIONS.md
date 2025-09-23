# Docker Setup Instructions for SDCpy App

## Quick Start with Docker Compose (Recommended)

1. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   Open your browser and navigate to: `http://localhost:8050`

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

## Manual Docker Commands

### Build the Docker image:
```bash
docker build -t sdcpy-app .
```

### Run the container:
```bash
docker run -p 8050:8050 --name sdcpy-app-container sdcpy-app
```

### Run in background (detached mode):
```bash
docker run -d -p 8050:8050 --name sdcpy-app-container sdcpy-app
```

### Stop the container:
```bash
docker stop sdcpy-app-container
```

### Remove the container:
```bash
docker rm sdcpy-app-container
```

## Customization Options

### Use a different port:
```bash
# Docker Compose
PORT=3000 docker-compose up

# Manual Docker
docker run -p 3000:8050 -e PORT=8050 sdcpy-app
```

### Development mode with volume mounting:
```bash
docker run -p 8050:8050 -v $(pwd):/app sdcpy-app
```

## Troubleshooting

1. **Port already in use:** Change the local port mapping:
   ```bash
   docker run -p 8051:8050 sdcpy-app
   ```
   Then access at `http://localhost:8051`

2. **Permission issues:** Make sure Docker has appropriate permissions on your system.

3. **Build issues:** Clear Docker cache and rebuild:
   ```bash
   docker system prune
   docker-compose up --build --no-cache
   ```

## What's included:

- **Dockerfile**: Multi-stage build optimized for production
- **.dockerignore**: Excludes unnecessary files from build context
- **docker-compose.yml**: Easy development and deployment setup
- **Updated app.py**: Configured to work properly in Docker containers

Your app will be accessible at `http://localhost:8050` once running!