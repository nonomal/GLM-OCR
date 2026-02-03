"""GlmOcr - Document Parsing with GLM OCR.

Document parsing service that supports layout detection and text recognition.

Supports two modes:
1. MaaS Mode: Passthrough to Zhipu cloud API (no GPU required)
2. Self-hosted Mode: Local vLLM/SGLang deployment (GPU required)
"""

__version__ = "0.1.1"
__author__ = "ZHIPUAI"

# Import main components
from . import dataloader
from . import layout
from . import postprocess
from . import utils
from .pipeline import Pipeline
from .config import GlmOcrConfig, load_config
from .parser_result import PipelineResult
from .maas_client import MaaSClient

# Import API
from .api import GlmOcr, parse

__all__ = [
    "dataloader",
    "layout",
    "postprocess",
    "utils",
    "Pipeline",
    "PipelineResult",
    "GlmOcrConfig",
    "load_config",
    "MaaSClient",
    "GlmOcr",
    "parse",
]
