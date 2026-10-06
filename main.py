"""Authenticated API entry point for Vast. Existing CPU entry point is unchanged."""
import os

if not os.environ.get("API_KEY", "").strip():
    raise RuntimeError("Set API_KEY in the Vast instance environment before starting main:app")

from vehicle_pipeline.api import app  # noqa: E402,F401
