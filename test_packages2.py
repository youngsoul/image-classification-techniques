import torch
from torch.nn import Linear
from torch.quantization import quantize_dynamic
from sentence_transformers import SentenceTransformer as SBERT