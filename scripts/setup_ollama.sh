#!/bin/bash
# Setup script: Install Ollama and pull all configured models
# Usage: bash scripts/setup_ollama.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "  Model Eval Framework — Ollama Setup"
echo "========================================"
echo ""

# Step 1: Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama is already installed ($(ollama --version))"
else
    echo -e "${YELLOW}⚠${NC} Ollama not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ollama
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo -e "${RED}✗${NC} Unsupported OS. Install Ollama manually: https://ollama.com/download"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Ollama installed"
fi

# Step 2: Start Ollama if not running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Ollama is already running"
else
    echo -e "${YELLOW}⚠${NC} Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 3

    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Ollama server started (PID: $OLLAMA_PID)"
    else
        echo -e "${RED}✗${NC} Failed to start Ollama. Try running 'ollama serve' manually."
        exit 1
    fi
fi

# Step 3: Pull models
MODELS=(
    "qwen2.5-coder:7b"
    "mistral:7b"
    "llama3:8b"
    "gemma2:9b"
)

echo ""
echo "Pulling models (this will take a while — ~18 GB total)..."
echo "----------------------------------------"

FAILED=0
for model in "${MODELS[@]}"; do
    echo ""
    echo -e "${YELLOW}→${NC} Pulling ${model}..."
    if ollama pull "$model"; then
        echo -e "${GREEN}✓${NC} ${model} ready"
    else
        echo -e "${RED}✗${NC} Failed to pull ${model}"
        FAILED=$((FAILED + 1))
    fi
done

# Step 4: Summary
echo ""
echo "========================================"
PULLED=$((${#MODELS[@]} - FAILED))
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All ${#MODELS[@]} models pulled successfully!${NC}"
    echo ""
    echo "Run the preflight check:"
    echo "  python scripts/preflight.py"
else
    echo -e "${YELLOW}⚠ ${PULLED}/${#MODELS[@]} models pulled. ${FAILED} failed.${NC}"
    echo ""
    echo "Retry failed models manually:"
    echo "  ollama pull <model-name>"
fi
echo "========================================"
