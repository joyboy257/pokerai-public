#!/bin/bash

# Set your Paperspace instance details
PAPERSPACE_USER="your_username"
PAPERSPACE_IP="your_paperspace_ip"
PAPERSPACE_PROJECT_PATH="~/TexasHoldemAI"

# Sync local project to Paperspace
echo "Syncing project files to Paperspace..."
scp -r ./* $PAPERSPACE_USER@$PAPERSPACE_IP:$PAPERSPACE_PROJECT_PATH

# SSH into Paperspace and execute setup commands
echo "Deploying on Paperspace..."
ssh $PAPERSPACE_USER@$PAPERSPACE_IP << EOF
    cd $PAPERSPACE_PROJECT_PATH
    source venv/bin/activate
    pip install -r requirements.txt
    echo "Running Abel Local Test on Paperspace..."
    PYTHONPATH=$(pwd) python scripts/abel_local_test.py
EOF

echo "Deployment to Paperspace completed."
