#!/usr/bin/env python3
"""
Setup Ollama

This script helps set up Ollama for use with the Adaptive Compressed World Model Framework.
It checks if Ollama is installed, installs it if needed, and pulls required models.
"""

import os
import sys
import subprocess
import platform
import requests
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama models required for the framework
EMBEDDING_MODELS = ["all-minilm", "nomic-embed-text"]
LLM_MODELS = ["phi3", "llama3", "gemma:2b", "mistral"]

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama based on the current platform."""
    system = platform.system().lower()
    
    if system == "linux":
        logger.info("Installing Ollama on Linux...")
        try:
            subprocess.run(
                ['curl', '-fsSL', 'https://ollama.com/install.sh', '|', 'sh'],
                shell=True,
                check=True
            )
            logger.info("Ollama installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Ollama: {e}")
            logger.info("Please install Ollama manually from https://ollama.com/download")
            return False
            
    elif system == "darwin":  # macOS
        logger.info("Installing Ollama on macOS...")
        logger.info("Please download and install Ollama from https://ollama.com/download")
        return False
        
    elif system == "windows":
        logger.info("Installing Ollama on Windows...")
        logger.info("Please download and install Ollama from https://ollama.com/download")
        return False
        
    else:
        logger.error(f"Unsupported platform: {system}")
        return False

def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_ollama_server():
    """Start the Ollama server."""
    system = platform.system().lower()
    
    if system == "linux":
        logger.info("Starting Ollama server...")
        try:
            # Start server in the background
            subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to start
            for _ in range(10):
                if check_ollama_running():
                    logger.info("Ollama server started successfully!")
                    return True
                time.sleep(1)
                
            logger.warning("Ollama server did not start in time.")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    else:
        logger.info(f"On {system}, please start Ollama server manually.")
        return False

def pull_model(model_name):
    """Pull a model using Ollama."""
    logger.info(f"Pulling model: {model_name}...")
    try:
        subprocess.run(['ollama', 'pull', model_name], check=True)
        logger.info(f"Successfully pulled model: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        return False

def setup_ollama():
    """Main function to set up Ollama."""
    logger.info("Setting up Ollama for the Adaptive Compressed World Model Framework...")
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        logger.info("Ollama is not installed. Installing now...")
        if not install_ollama():
            logger.error("Failed to install Ollama. Please install it manually.")
            return False
    else:
        logger.info("Ollama is already installed.")
    
    # Check if Ollama server is running
    if not check_ollama_running():
        logger.info("Ollama server is not running. Starting now...")
        if not start_ollama_server():
            logger.error("Failed to start Ollama server. Please start it manually with 'ollama serve'.")
            return False
    else:
        logger.info("Ollama server is already running.")
    
    # Get available models
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = response.json().get("models", [])
            available_model_names = [model.get("name") for model in available_models]
            logger.info(f"Available models: {', '.join(available_model_names)}")
        else:
            logger.warning(f"Ollama API returned status code {response.status_code}")
            available_model_names = []
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama API.")
        return False
    
    # Pull embedding models if needed
    embedding_pulled = False
    for model in EMBEDDING_MODELS:
        if not any(model in name for name in available_model_names):
            logger.info(f"Embedding model {model} not found. Pulling now...")
            if pull_model(model):
                embedding_pulled = True
                break
        else:
            # Model already exists
            embedding_pulled = True
            logger.info(f"Embedding model {model} is already available.")
            break
    
    if not embedding_pulled and not any(model in name for name in available_model_names for model in EMBEDDING_MODELS):
        logger.warning("No embedding models were pulled. Using fallback embedding method.")
    
    # Pull at least one LLM model if needed
    llm_pulled = False
    for model in LLM_MODELS:
        if any(model in name for name in available_model_names):
            # Model already exists
            llm_pulled = True
            logger.info(f"LLM model containing '{model}' is already available.")
            break
            
    if not llm_pulled:
        for model in LLM_MODELS:
            logger.info(f"LLM model {model} not found. Pulling now...")
            if pull_model(model):
                llm_pulled = True
                break
                
    if not llm_pulled:
        logger.warning("No LLM models were pulled. Using extractive summarization.")
    
    logger.info("Ollama setup completed successfully!")
    return True

if __name__ == "__main__":
    setup_ollama()
