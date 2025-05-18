#!/bin/bash

# Install dependencies
pip3 install -r requirements.txt

# Run Alembic migrations
alembic upgrade head

chmod +x build_files.sh
