#!/bin/bash
set -e

echo "Building frontend..."
cd frontend
npm ci
npm run build

echo "Building backend..."
cd ../backend
pip install -r requirements.txt

echo "Build complete!"