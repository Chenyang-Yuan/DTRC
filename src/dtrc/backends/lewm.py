"""Frozen LeWM checkpoint adapter. See licenses/LeWM-LICENSE for upstream attribution."""

from __future__ import annotations
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor


class PatchEmbeddings(nn.Module):

    def __init__(self, image_size: int, patch_size: int, dim: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection = nn.Conv2d(3, dim, patch_size, stride=patch_size)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels).flatten(2).transpose(1, 2)


class Embeddings(nn.Module):

    def __init__(self, image_size: int, patch_size: int, dim: int):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, dim)

    def _position_embedding(self, height, width, dtype):
        target_h = height // self.patch_embeddings.patch_size
        target_w = width // self.patch_embeddings.patch_size
        source_tokens = self.position_embeddings[:, 1:]
        source_size = int(math.sqrt(source_tokens.shape[1]))
        if source_size == target_h == target_w:
            return self.position_embeddings.to(dtype=dtype)
        source_tokens = source_tokens.reshape(1, source_size, source_size, -1).permute(
            0, 3, 1, 2
        )
        resized = F.interpolate(
            source_tokens.float(),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=dtype)
        resized = resized.permute(0, 2, 3, 1).reshape(1, target_h * target_w, -1)
        return torch.cat(
            [self.position_embeddings[:, :1].to(dtype=dtype), resized], dim=1
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        patch = self.patch_embeddings(pixels)
        cls = self.cls_token.expand(pixels.shape[0], -1, -1)
        tokens = torch.cat([cls, patch], dim=1)
        return tokens + self._position_embedding(
            pixels.shape[-2], pixels.shape[-1], tokens.dtype
        )


class ViTAttention(nn.Module):

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (b, n, _) = x.shape

        def split_heads(tensor):
            return tensor.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)

        (q, k, v) = (
            split_heads(self.q_proj(x)),
            split_heads(self.k_proj(x)),
            split_heads(self.v_proj(x)),
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class ViTMLP(nn.Module):

    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate)
        self.fc2 = nn.Linear(intermediate, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class ViTLayer(nn.Module):

    def __init__(self, dim: int, heads: int, intermediate: int):
        super().__init__()
        self.attention = ViTAttention(dim, heads)
        self.mlp = ViTMLP(dim, intermediate)
        self.layernorm_before = nn.LayerNorm(dim, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x):
        x = x + self.attention(self.layernorm_before(x))
        return x + self.mlp(self.layernorm_after(x))


class PixelEncoder(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=14,
        dim=192,
        depth=12,
        heads=3,
        intermediate=768,
    ):
        super().__init__()
        self.embeddings = Embeddings(image_size, patch_size, dim)
        self.layers = nn.ModuleList(
            [ViTLayer(dim, heads, intermediate) for _ in range(depth)]
        )
        self.layernorm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, pixels, interpolate_pos_encoding: bool = True):
        del interpolate_pos_encoding
        x = self.embeddings(pixels)
        for layer in self.layers:
            x = layer(x)
        return EncoderOutput(self.layernorm(x))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CausalAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        (b, t, _) = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        (q, k, v) = [
            item.reshape(b, t, self.heads, self.dim_head).transpose(1, 2)
            for item in qkv
        ]
        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        return self.to_out(out.transpose(1, 2).reshape(b, t, -1))


class ConditionalBlock(nn.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = CausalAttention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, condition):
        (shift_a, scale_a, gate_a, shift_m, scale_m, gate_m) = self.adaLN_modulation(
            condition
        ).chunk(6, dim=-1)
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        return x + gate_m * self.mlp(modulate(self.norm2(x), shift_m, scale_m))


class ConditionalTransformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x, condition):
        for layer in self.layers:
            x = layer(x, condition)
        return self.norm(x)


class Predictor(nn.Module):

    def __init__(
        self,
        num_frames,
        input_dim,
        depth,
        heads,
        mlp_dim,
        dim_head,
        dropout,
        emb_dropout,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ConditionalTransformer(
            input_dim, depth, heads, dim_head, mlp_dim, dropout
        )

    def forward(self, x, condition):
        x = self.dropout(x + self.pos_embedding[:, : x.shape[1]])
        return self.transformer(x, condition)


class ActionEncoder(nn.Module):

    def __init__(self, input_dim, emb_dim, smoothed_dim=10):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, 1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, 4 * emb_dim),
            nn.SiLU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, action):
        x = self.patch_embed(action.float().transpose(1, 2)).transpose(1, 2)
        return self.embed(x)


class ProjectionMLP(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class StandaloneLeWM(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        (enc, pred, act, proj) = (
            config["encoder"],
            config["predictor"],
            config["action_encoder"],
            config["projector"],
        )
        self.config = config
        self.encoder = PixelEncoder(
            image_size=enc["image_size"],
            patch_size=enc["patch_size"],
            dim=pred["input_dim"],
            depth=int(enc.get("depth", 12)),
            heads=int(enc.get("heads", 3)),
            intermediate=int(enc.get("intermediate", 4 * pred["input_dim"])),
        )
        self.predictor = Predictor(
            num_frames=pred["num_frames"],
            input_dim=pred["input_dim"],
            depth=pred["depth"],
            heads=pred["heads"],
            mlp_dim=pred["mlp_dim"],
            dim_head=pred["dim_head"],
            dropout=pred["dropout"],
            emb_dropout=pred["emb_dropout"],
        )
        self.action_encoder = ActionEncoder(
            input_dim=act["input_dim"],
            emb_dim=act["emb_dim"],
            smoothed_dim=act.get("smoothed_dim", 10),
        )
        self.projector = ProjectionMLP(
            dim=proj["input_dim"], hidden_dim=proj["hidden_dim"]
        )
        self.pred_proj = ProjectionMLP(
            dim=config["pred_proj"]["input_dim"],
            hidden_dim=config["pred_proj"]["hidden_dim"],
        )

    def encode_pixels(self, pixels):
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        return self.projector(output.last_hidden_state[:, 0])

    def predict_sequence(self, latents, actions):
        action_embedding = self.action_encoder(actions)
        prediction = self.predictor(latents, action_embedding)
        (b, t, d) = prediction.shape
        return self.pred_proj(prediction.reshape(b * t, d)).reshape(b, t, d)


def _canonicalize_checkpoint_keys(state):
    legacy_prefix = "encoder.encoder.layer."
    if not any((key.startswith(legacy_prefix) for key in state)):
        return state
    suffix_map = {
        "attention.attention.query": "attention.q_proj",
        "attention.attention.key": "attention.k_proj",
        "attention.attention.value": "attention.v_proj",
        "attention.output.dense": "attention.o_proj",
        "intermediate.dense": "mlp.fc1",
        "output.dense": "mlp.fc2",
        "layernorm_before": "layernorm_before",
        "layernorm_after": "layernorm_after",
    }
    converted = {}
    for key, value in state.items():
        mapped = key
        if key.startswith(legacy_prefix):
            remainder = key[len(legacy_prefix) :]
            (layer_index, separator, layer_key) = remainder.partition(".")
            if not separator or not layer_index.isdigit():
                raise RuntimeError(f"Malformed legacy ViT checkpoint key: {key}")
            for old_suffix, new_suffix in suffix_map.items():
                if layer_key == old_suffix or layer_key.startswith(old_suffix + "."):
                    mapped = f"encoder.layers.{layer_index}.{new_suffix}{layer_key[len(old_suffix):]}"
                    break
            else:
                raise RuntimeError(f"Unsupported legacy ViT checkpoint key: {key}")
        if mapped in converted:
            raise RuntimeError(f"Checkpoint key collision: {key} -> {mapped}")
        converted[mapped] = value
    return converted


def load_standalone_lewm(model_dir, device="cpu"):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    state = torch.load(model_dir / "weights.pt", map_location="cpu", weights_only=True)
    state = _canonicalize_checkpoint_keys(state)
    patch_weight = state.get("action_encoder.patch_embed.weight")
    if patch_weight is None or patch_weight.ndim != 3:
        raise RuntimeError("Checkpoint lacks action_encoder.patch_embed.weight")
    config = deepcopy(config)
    config["action_encoder"]["smoothed_dim"] = int(patch_weight.shape[0])
    if int(patch_weight.shape[1]) != int(config["action_encoder"]["input_dim"]):
        raise RuntimeError(
            f"Checkpoint action input dimension differs from config: {int(patch_weight.shape[1])} vs {config['action_encoder']['input_dim']}"
        )
    model = StandaloneLeWM(config)
    (missing, unexpected) = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Official checkpoint is incompatible with the standalone model. Missing={missing}, unexpected={unexpected}"
        )
    return model.eval().requires_grad_(False).to(device)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_uint8(pixels: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    if pixels.ndim != 4:
        raise ValueError(f"expected 4-D pixel batch, got {tuple(pixels.shape)}")
    if pixels.shape[-1] in (1, 3, 4):
        pixels = pixels.permute(0, 3, 1, 2)
    pixels = pixels[:, :3]
    if pixels.shape[1] == 1:
        pixels = pixels.expand(-1, 3, -1, -1)
    x = pixels.float()
    if x.detach().amax() > 1.5:
        x = x / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(
            x,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return x.contiguous()


def sha256_file(path, chunk=1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class LeWMBackend:

    def __init__(self, model, source: str, kind: str, device):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.source = str(source)
        self.kind = kind
        self.device = torch.device(device)
        self.model.to(self.device)
        with torch.no_grad():
            probe = torch.zeros(1, 3, 224, 224, device=self.device)
            self.latent_dim = int(self._encode(probe).shape[-1])
        self.history = int(getattr(self._predictor(), "num_frames", 3))
        conv = self._action_encoder().patch_embed
        self.action_block_dim = int(conv.in_channels)

    def _predictor(self):
        return self.model.predictor

    def _action_encoder(self):
        return self.model.action_encoder

    def _encode(self, pixels):
        if hasattr(self.model, "encode_pixels"):
            return self.model.encode_pixels(pixels)
        output = self.model.encoder(pixels, interpolate_pos_encoding=True)
        return self.model.projector(output.last_hidden_state[:, 0])

    @torch.no_grad()
    def encode_pixels(
        self, pixels: torch.Tensor, batch_size: int = 256
    ) -> torch.Tensor:
        pixels = pixels.to(self.device)
        chunks = [
            self._encode(pixels[start : start + batch_size]).float()
            for start in range(0, len(pixels), batch_size)
        ]
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_uint8(self, frames: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        chunks = []
        for start in range(0, len(frames), batch_size):
            batch = preprocess_uint8(frames[start : start + batch_size].to(self.device))
            chunks.append(self._encode(batch).float())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def predict_sequence(
        self, latents: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self.model, "predict_sequence"):
            return self.model.predict_sequence(latents, actions)
        act_emb = self.model.action_encoder(actions)
        return self.model.predict(latents, act_emb)

    @torch.no_grad()
    def rollout(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        (batch, h0, _) = latents.shape
        total = actions.shape[1]
        if total < h0:
            raise ValueError("need at least one action block per history frame")
        history = self.history
        frames = list(latents.unbind(dim=1))
        for t in range(total - h0 + 1):
            lo = max(0, h0 + t - history)
            emb = torch.stack(frames[lo:], dim=1)
            act = actions[:, lo : h0 + t]
            frames.append(self.predict_sequence(emb, act)[:, -1])
        return torch.stack(frames, dim=1)

    def describe(self) -> dict:
        return {
            "source": self.source,
            "kind": self.kind,
            "latent_dim": self.latent_dim,
            "predictor_history": self.history,
            "action_block_dim": self.action_block_dim,
        }


def load_lewm_backend(source, device="cuda") -> LeWMBackend:
    path = Path(str(source))
    if (
        path.is_dir()
        and (path / "weights.pt").is_file()
        and (path / "config.json").is_file()
    ):
        model = load_standalone_lewm(path, device=device)
        backend = LeWMBackend(model, path, "hf_standalone", device)
        backend.weights_sha256 = sha256_file(path / "weights.pt")
        return backend
    import stable_worldmodel as swm

    object_path = Path(swm.data.utils.get_cache_dir(), f"{source}_object.ckpt")
    if object_path.is_file():
        model = torch.load(object_path, map_location="cpu", weights_only=False)
        kind = "swm_object"
    else:
        model = swm.wm.utils.load_pretrained(str(source))
        kind = "swm_pretrained"
    model = (
        model.model
        if hasattr(model, "model") and (not hasattr(model, "encoder"))
        else model
    )
    backend = LeWMBackend(model, source, kind, device)
    backend.weights_sha256 = None
    return backend
