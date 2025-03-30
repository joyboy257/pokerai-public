#!/bin/bash

# Check for changes and add them
echo "Checking for changes..."
git add .

# Commit the changes
read -p "Enter commit message: " commit_message
git commit -m "$commit_message"

# Pull the latest changes from main to avoid conflicts
echo "Pulling latest changes from main..."
git pull origin main

# Push to GitHub
echo "Pushing to GitHub..."
git push origin main

echo "✅ Done! Git repository is up to date."
