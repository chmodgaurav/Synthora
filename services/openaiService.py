"""Compatibility shim.

Old project files imported `openai_service`. This keeps the app running while the
codebase is migrated to local Ollama. New code should import ollama_service from
services.ollamaService directly.
"""

from services.ollamaService import ollama_service

openai_service = ollama_service
