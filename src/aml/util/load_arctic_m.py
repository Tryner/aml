from importlib.resources import files

from transformers import AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer

from .gte import GteConfig, GteModel
from aml import models

def register_gte():
    AutoConfig.register("gte", GteConfig)
    AutoModel.register(GteConfig, GteModel)

def load_arctic_m(model_path: str | None = None):
    register_gte()
    if model_path is None:
        model_path = files(models).joinpath("snowflake-arctic-embed-m-v2.0")
    return SentenceTransformer(model_path, config_kwargs={"use_memory_efficient_attention": False})
