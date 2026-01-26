#!/bin/bash

echo ""
echo "========================================"
echo "   YOLO Training Studio"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found!"
    exit 1
fi

echo "[INFO] Starting application..."
python3 app.py
