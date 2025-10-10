import streamlit as st
import os
import gc
import torch
import weakref
from typing import Dict, Any, Literal
from llama_cpp import Llama

# Import model loading functions from their original modules
# This helps keep the loading logic with the component that uses it,
# while the manager just orchestrates the loading/unloading.
from embeddings.embedding_generator import load_embedding_model
from retrieval.reranker import load_reranker_model

# A type hint for the different models we'll be managing
ModelType = Literal["embedding", "reranker", "llm", "condenser", "expander"]
VRAMStrategy = Literal["Aggressive", "Balanced", "Performance"]

class ModelManager:
    """
    A centralized manager for loading, unloading, and accessing all models
    in the application based on a selected VRAM management strategy.
    """
    def __init__(self, settings_models: Any, device: str):
        self.models_config = settings_models
        self.device = device
        self.loaded_models: Dict[ModelType, Any] = {}
        self._vram_strategy: VRAMStrategy = "Performance" # Default strategy
        print(f"ModelManager initialized with device '{self.device}'.")

    def set_vram_strategy(self, strategy: VRAMStrategy):
        """Sets the VRAM management strategy and unloads all models to apply the new strategy."""
        if strategy != self._vram_strategy:
            print(f"Changing VRAM strategy to: {strategy}. Unloading all models.")
            self.unload_all_models()
            self._vram_strategy = strategy

    def get_vram_strategy(self) -> VRAMStrategy:
        return self._vram_strategy

    def _load_model(self, model_type: ModelType):
        """Internal method to load a specific model if not already loaded."""
        if model_type in self.loaded_models:
            return

        print(f"Loading '{model_type}' model...")
        model = None
        try:
            if model_type == "embedding":
                model = load_embedding_model(self.models_config.embedding_id, self.device)
            elif model_type == "reranker":
                model = load_reranker_model(self.models_config.reranker_id, device=self.device)
            elif model_type == "llm":
                model = self._load_llama_cpp_model(self.models_config.llm_model_path, self.models_config.llm_generation_config)
            elif model_type == "condenser":
                model = self._load_llama_cpp_model(self.models_config.condenser_model_path, self.models_config.condenser_generation_config)
            elif model_type == "expander":
                model = self._load_llama_cpp_model(self.models_config.query_expander_model_path, self.models_config.query_expander_generation_config)

            if model:
                self.loaded_models[model_type] = model
                print(f"Successfully loaded '{model_type}' model.")
            else:
                raise ValueError("Model loading function returned None.")
        except Exception as e:
            st.error(f"Failed to load '{model_type}' model: {e}")
            self.loaded_models[model_type] = None # Mark as failed to avoid retrying

    def _load_llama_cpp_model(self, model_path: str, gen_config: Dict[str, Any]) -> Llama:
        """Helper to load any Llama.cpp model with detailed diagnostics."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        n_gpu_layers = gen_config.get("n_gpu_layers", -1)

        # If GPU layers are requested but no GPU is detected, warn the user.
        if n_gpu_layers != 0 and self.device == "cpu":
            st.warning(
                f"Attempting to load model '{os.path.basename(model_path)}' with n_gpu_layers={n_gpu_layers} but no GPU was detected. "
                "The model will run on CPU. Please check your PyTorch and CUDA/ROCm installation."
            )
            # Force n_gpu_layers to 0 to prevent errors.
            n_gpu_layers = 0

        # Force verbose=True to print detailed logs from llama.cpp backend for diagnostics.
        print(f"Initializing Llama.cpp model at '{model_path}' with n_gpu_layers={n_gpu_layers} and verbose=True for diagnostics.")

        return Llama(
            model_path=model_path,
            n_ctx=gen_config.get("n_ctx", 2048),
            n_gpu_layers=n_gpu_layers,
            verbose=True  # Always on for better debugging
        )

    def get_model(self, model_type: ModelType) -> Any:
        """Gets a model, loading it if necessary based on the VRAM strategy."""
        if model_type not in self.loaded_models:
            self._load_model(model_type)
        return self.loaded_models.get(model_type)

    def unload_model(self, model_type: ModelType):
        """Unloads a specific model, frees up VRAM, and confirms garbage collection."""
        if model_type in self.loaded_models:
            print(f"Attempting to unload '{model_type}' model...")
            model_instance = self.loaded_models.pop(model_type, None)

            if model_instance is None:
                print(f"Model '{model_type}' was already unloaded or not in manager.")
                return

            # Create a weak reference to check for garbage collection
            model_ref = weakref.ref(model_instance)
            del model_instance

            # Manually trigger garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if model_ref() is None:
                print(f"Successfully garbage collected and unloaded '{model_type}' model.")
            else:
                st.warning(
                    f"Failed to unload '{model_type}' model completely. "
                    "There might be other references to it preventing memory release."
                )
                print(f"WARNING: Failed to garbage collect '{model_type}' model. Other references may exist.")

    def unload_all_models(self):
        """Unloads all currently loaded models."""
        # Create a list of keys to avoid issues with changing dict size during iteration
        model_keys = list(self.loaded_models.keys())
        for model_type in model_keys:
            self.unload_model(model_type)
        print("All models have been unloaded.")

    def get_loaded_models_status(self) -> Dict[ModelType, bool]:
        """Returns a dictionary indicating the status of each model."""
        all_models: List[ModelType] = ["embedding", "reranker", "llm", "condenser", "expander"]
        return {model: model in self.loaded_models for model in all_models}