# src/config/settings.py
import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")


def load_settings():
    """Loads settings from settings.json."""
    if not os.path.exists(SETTINGS_PATH):
        return {}
    with open(SETTINGS_PATH, "r") as f:
        return json.load(f)

def save_settings(updated_settings):
    """Saves updated settings to settings.json."""
    with open(SETTINGS_PATH, "w") as f:
        json.dump(updated_settings, f, indent=4)
