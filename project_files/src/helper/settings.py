import sys
import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Pydantic Models for Validation (remain unchanged) ---
class ModelSettings(BaseModel):
    embedding_id: str
    reranker_id: str
    ollama_llm: str
    query_expander_id: str
    condenser_model_id: str
    ollama_options: Dict[str, Any]

class DatabaseSettings(BaseModel):
    persist_path: str
    collection_name: str
    force_rebuild: bool

class ProcessingSettings(BaseModel):
    initial_split_level: int = Field(..., gt=0)
    max_chars_per_chunk: int = Field(..., gt=0)
    min_chars_per_chunk: int = Field(..., gt=0)

class PipelineSettings(BaseModel):
    enable_conversation_memory: bool
    use_reranker: bool
    enable_query_expansion: bool
    query_expansion_char_threshold: int
    retrieval_top_k: int
    default_retrieval_top_k: int
    min_chunks_to_llm: int
    max_chunks_to_llm: int
    min_absolute_score_threshold: float
    min_chunks_for_gap_detection: int
    gap_detection_factor: float
    small_epsilon: float

class SystemSettings(BaseModel):
    low_vram_mode: bool

class AppSettings(BaseModel):
    models: ModelSettings
    database: DatabaseSettings
    processing: ProcessingSettings
    pipeline: PipelineSettings
    system: SystemSettings

# --- Simplified Loading Function ---
def load_settings_from_any_path(paths: list) -> AppSettings:
    """
    Tries to load the YAML configuration from a list of possible paths.
    """
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            if config_data:
                print(f"Konfiguration erfolgreich von '{path}' geladen und validiert.")
                return AppSettings(**config_data)
        except FileNotFoundError:
            continue # Try the next path
        except Exception as e:
            print(f"FEHLER: Fehler beim Validieren der Konfiguration von '{path}': {e}")
            raise

    # If the loop finishes without finding a file
    raise FileNotFoundError(f"FEHLER: Konfigurationsdatei konnte in keinem der Pfade gefunden werden: {paths}")

# --- Main execution block ---
try:
    # List of paths to try, in order of priority
    possible_paths = [
        "./helper/config.yaml",
        "./project_files/src/helper/config.yaml",
        "config.yaml" # A fallback for the root directory
    ]
    settings = load_settings_from_any_path(possible_paths)
except Exception as e:
    print(e)
    sys.exit(1)