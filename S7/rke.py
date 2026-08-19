"""Reversible Kronecker Embeddings (RKE-Head), implemented with NumPy only."""

from __future__ import annotations

import hashlib
import json
import math
import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PAD = "<PAD>"
EOS = "<EOS>"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


@dataclass(frozen=True)
class ReversibleCodec:
    """An ordered character-position code with explicit termination.

    The experimental alphabet is intentionally small for fast training.  The
    same construction has 258 states for arbitrary bytes: PAD, EOS, and 0..255.
    """

    alphabet: str
    max_chars: int

    def __post_init__(self) -> None:
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet characters must be unique")

    @property
    def symbols(self) -> tuple[str, ...]:
        return (PAD, EOS, *tuple(self.alphabet))

    @property
    def symbol_to_id(self) -> dict[str, int]:
        return {symbol: index for index, symbol in enumerate(self.symbols)}

    @property
    def slots(self) -> int:
        return self.max_chars + 1

    @property
    def width(self) -> int:
        return len(self.symbols)

    @property
    def feature_dim(self) -> int:
        return self.slots * self.width

    def ids(self, text: str) -> np.ndarray:
        if len(text) > self.max_chars:
            raise ValueError(f"{text!r} exceeds max_chars={self.max_chars}")
        unknown = set(text) - set(self.alphabet)
        if unknown:
            raise ValueError(f"characters outside experimental alphabet: {unknown}")
        ids = np.zeros(self.slots, dtype=np.int64)
        lookup = self.symbol_to_id
        ids[:len(text)] = [lookup[char] for char in text]
        ids[len(text)] = lookup[EOS]
        return ids

    def encode(self, text: str) -> np.ndarray:
        matrix = np.zeros((self.slots, self.width), dtype=np.float64)
        matrix[np.arange(self.slots), self.ids(text)] = 1.0
        return matrix

    def decode_ids(self, ids: np.ndarray) -> str:
        chars = []
        for value in ids.tolist():
            symbol = self.symbols[int(value)]
            if symbol == EOS:
                return "".join(chars)
            if symbol == PAD:
                raise ValueError("PAD occurred before EOS")
            chars.append(symbol)
        raise ValueError("code has no EOS")

    def decode_logits(self, logits: np.ndarray) -> str:
        return self.decode_ids(np.asarray(logits).argmax(axis=-1))

    def proof_record(self, text: str) -> dict[str, Any]:
        code = self.encode(text)
        recovered = self.decode_logits(code)
        return {
            "text": text, "ids": self.ids(text).tolist(), "code_hash": sha256(code.tolist()),
            "decoded": recovered, "exact": recovered == text,
        }


@dataclass(frozen=True)
class FullByteCodec:
    """The deployment form: PAD, EOS, and all 256 byte values."""

    max_bytes: int

    @property
    def slots(self) -> int:
        return self.max_bytes + 1

    @property
    def width(self) -> int:
        return 258

    @property
    def feature_dim(self) -> int:
        return self.slots * self.width

    @property
    def symbol_to_id(self) -> dict[str, int]:
        return {PAD: 0, EOS: 1}

    def ids(self, payload: bytes) -> np.ndarray:
        if len(payload) > self.max_bytes:
            raise ValueError("payload exceeds max_bytes")
        result = np.zeros(self.slots, dtype=np.int64)
        if payload:
            result[:len(payload)] = np.frombuffer(payload, dtype=np.uint8).astype(np.int64) + 2
        result[len(payload)] = 1
        return result

    def encode(self, payload: bytes) -> np.ndarray:
        matrix = np.zeros((self.slots, self.width), dtype=np.uint8)
        matrix[np.arange(self.slots), self.ids(payload)] = 1
        return matrix

    def decode_ids(self, ids: np.ndarray) -> bytes:
        output = bytearray()
        for value in ids.tolist():
            if value == 1:
                return bytes(output)
            if value == 0:
                raise ValueError("PAD occurred before EOS")
            output.append(int(value) - 2)
        raise ValueError("code has no EOS")

    def decode_logits(self, logits: np.ndarray) -> bytes:
        return self.decode_ids(np.asarray(logits).argmax(axis=-1))


@dataclass(frozen=True)
class ContinuationByteCodec:
    """Lossless dynamic byte codec made from bounded continuation blocks.

    Each block terminates with EOS when the payload is complete or CONT when
    another block follows. This avoids both silent truncation and a global
    maximum word length while preserving fixed-size tensors inside each block.
    States are PAD=0, EOS=1, CONT=2, and byte_n=n+3.
    """

    block_bytes: int = 24

    def __post_init__(self) -> None:
        if self.block_bytes < 1:
            raise ValueError("block_bytes must be positive")

    @property
    def slots(self) -> int:
        return self.block_bytes + 1

    @property
    def width(self) -> int:
        return 259

    def ids(self, payload: bytes) -> list[np.ndarray]:
        chunks = ([payload[index:index + self.block_bytes]
                   for index in range(0, len(payload), self.block_bytes)] or [b""])
        blocks = []
        for index, chunk in enumerate(chunks):
            result = np.zeros(self.slots, dtype=np.int64)
            if chunk:
                result[:len(chunk)] = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64) + 3
            result[len(chunk)] = 1 if index == len(chunks) - 1 else 2
            blocks.append(result)
        return blocks

    def encode(self, payload: bytes) -> list[np.ndarray]:
        matrices = []
        for states in self.ids(payload):
            matrix = np.zeros((self.slots, self.width), dtype=np.uint8)
            matrix[np.arange(self.slots), states] = 1
            matrices.append(matrix)
        return matrices

    def decode_ids(self, blocks: list[np.ndarray]) -> bytes:
        if not blocks:
            raise ValueError("at least one block is required")
        output = bytearray()
        for block_index, raw in enumerate(blocks):
            states = np.asarray(raw)
            if states.shape != (self.slots,):
                raise ValueError("invalid block shape")
            terminator_seen = False
            for slot, state in enumerate(states.tolist()):
                state = int(state)
                if state >= 3:
                    if terminator_seen:
                        raise ValueError("byte after block terminator")
                    output.append(state - 3)
                elif state in (1, 2):
                    if terminator_seen:
                        raise ValueError("multiple block terminators")
                    terminator_seen = True
                    expected = 1 if block_index == len(blocks) - 1 else 2
                    if state != expected:
                        raise ValueError("EOS/CONT does not match block position")
                elif state == 0:
                    if not terminator_seen:
                        raise ValueError("PAD before block terminator")
                else:
                    raise ValueError("invalid state")
            if not terminator_seen:
                raise ValueError("block has no terminator")
        return bytes(output)


def utf8_prefix_is_valid(payload: bytes) -> bool:
    """True when payload is a valid complete or incomplete UTF-8 prefix."""
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        decoder.decode(payload, final=False)
        return True
    except UnicodeDecodeError:
        return False


def utf8_is_complete(payload: bytes) -> bool:
    try:
        payload.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError:
        return False


def constrained_utf8_decode(logits: np.ndarray, max_bytes: int) -> bytes:
    """Decode FullByteCodec logits while masking invalid UTF-8 transitions.

    State 0 is PAD, state 1 is EOS, and state byte+2 is a byte.  EOS is only
    legal at a complete code-point boundary.  PAD is never generated directly.
    """
    rows = np.asarray(logits)
    if rows.shape != (max_bytes + 1, 258):
        raise ValueError(f"expected {(max_bytes + 1, 258)}, got {rows.shape}")
    output = bytearray()
    for row in rows:
        for state in np.argsort(row)[::-1]:
            state = int(state)
            if state == 0:
                continue
            if state == 1:
                if utf8_is_complete(bytes(output)):
                    return bytes(output)
                continue
            candidate = bytes(output) + bytes([state - 2])
            if len(candidate) <= max_bytes and utf8_prefix_is_valid(candidate):
                output.append(state - 2)
                break
        else:
            raise ValueError("no valid UTF-8 transition")
    raise ValueError("no legal EOS before maximum length")


def original_truncated_code(text: str, max_positions: int) -> tuple[int, ...]:
    """The information-bearing support of the paper's codec before projection."""
    return tuple(text.encode("utf-8")[:max_positions])


def make_split(seed: int, alphabet: str, max_chars: int, train_size: int, test_size: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    values: set[str] = set()
    while len(values) < train_size + test_size:
        length = int(rng.integers(1, max_chars + 1))
        values.add("".join(rng.choice(list(alphabet), size=length).tolist()))
    ordered = sorted(values, key=lambda text: sha256({"seed": seed, "text": text}))
    return ordered[:train_size], ordered[train_size:]


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 0.006):
        self.lr, self.step = lr, 0
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.step += 1
        for key in params:
            grad = np.clip(grads[key], -1.0, 1.0)
            self.m[key] = .9 * self.m[key] + .1 * grad
            self.v[key] = .999 * self.v[key] + .001 * grad * grad
            mhat = self.m[key] / (1 - .9 ** self.step)
            vhat = self.v[key] / (1 - .999 ** self.step)
            params[key] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


class TinyTransformer:
    """A one-layer, single-head causal transformer trained from scratch.

    It receives [COPY, distractor, payload] and must emit the payload through a
    structured position×symbol head.  There is no token lookup table or vocab
    classifier anywhere in the model.
    """

    def __init__(self, codec: ReversibleCodec, d_slot: int = 12, seed: int = 7):
        self.codec = codec
        self.d_slot = d_slot
        self.d_model = codec.slots * d_slot
        rng = np.random.default_rng(seed)
        d = self.d_model
        self.params = {
            # The same map is used for input and transposed for output.  This is
            # the Kronecker constraint and removes a separate output head.
            "Ein": rng.normal(0, 0.08, (codec.width, d_slot)),
            "Wq": rng.normal(0, 0.06, (d, d)),
            "Wk": rng.normal(0, 0.06, (d, d)),
            "Wv": rng.normal(0, 0.06, (d, d)),
            "Wo": rng.normal(0, 0.02, (d, d)),
            "pos": rng.normal(0, 0.02, (3, d)),
        }

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def features(self, payloads: list[str], distractors: list[str]) -> np.ndarray:
        command = self.codec.encode("c")
        return np.stack([np.stack([command, self.codec.encode(noise), self.codec.encode(payload)])
                         for payload, noise in zip(payloads, distractors)])

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        p = self.params
        batch = len(features)
        x = (features @ p["Ein"]).reshape(batch, 3, self.d_model) + p["pos"][None, :, :]
        q, k, v = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(self.d_model)
        mask = np.triu(np.ones((3, 3), dtype=bool), 1)
        scores[:, mask] = -1e9
        attention = self._softmax(scores)
        attended = attention @ v
        hidden = x + attended @ p["Wo"]
        last_slots = hidden[:, -1, :].reshape(batch, self.codec.slots, self.d_slot)
        logits = last_slots @ p["Ein"].T
        return logits, {"features": features, "x": x, "q": q, "k": k, "v": v,
                        "attention": attention, "attended": attended, "hidden": hidden}

    def loss_and_grads(self, features: np.ndarray, targets: np.ndarray,
                       loss_mask: np.ndarray | None = None) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self.forward(features)
        probabilities = self._softmax(logits)
        batch = len(features)
        chosen = probabilities[np.arange(batch)[:, None], np.arange(self.codec.slots)[None, :], targets]
        mask = np.ones_like(chosen) if loss_mask is None else np.asarray(loss_mask, dtype=np.float64)
        denominator = max(float(mask.sum()), 1.0)
        loss = float((-np.log(chosen + 1e-12) * mask).sum() / denominator)
        dlogits = probabilities
        dlogits[np.arange(batch)[:, None], np.arange(self.codec.slots)[None, :], targets] -= 1
        dlogits *= mask[:, :, None] / denominator
        p, c = self.params, cache
        grads: dict[str, np.ndarray] = {}
        last_slots = c["hidden"][:, -1, :].reshape(batch, self.codec.slots, self.d_slot)
        d_ein_output = dlogits.reshape(-1, self.codec.width).T @ last_slots.reshape(-1, self.d_slot)
        dhidden = np.zeros_like(c["hidden"])
        dhidden[:, -1, :] = (dlogits @ p["Ein"]).reshape(batch, self.d_model)
        dx = dhidden.copy()
        dattn_out = dhidden @ p["Wo"].T
        grads["Wo"] = c["attended"].reshape(-1, self.d_model).T @ dhidden.reshape(-1, self.d_model)
        dattention = dattn_out @ c["v"].transpose(0, 2, 1)
        dv = c["attention"].transpose(0, 2, 1) @ dattn_out
        dscores = c["attention"] * (dattention - (dattention * c["attention"]).sum(axis=-1, keepdims=True))
        scale = math.sqrt(self.d_model)
        dq = dscores @ c["k"] / scale
        dk = dscores.transpose(0, 2, 1) @ c["q"] / scale
        flat_x = c["x"].reshape(-1, self.d_model)
        grads["Wq"] = flat_x.T @ dq.reshape(-1, self.d_model)
        grads["Wk"] = flat_x.T @ dk.reshape(-1, self.d_model)
        grads["Wv"] = flat_x.T @ dv.reshape(-1, self.d_model)
        dx += dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
        dslots = dx.reshape(batch, 3, self.codec.slots, self.d_slot)
        grads["Ein"] = (c["features"].reshape(-1, self.codec.width).T @ dslots.reshape(-1, self.d_slot)
                        + d_ein_output)
        grads["pos"] = dx.sum(axis=0)
        return loss, grads

    def predict(self, payloads: list[str], distractors: list[str]) -> list[str]:
        logits, _ = self.forward(self.features(payloads, distractors))
        results = []
        for item in logits:
            try:
                results.append(self.codec.decode_logits(item))
            except ValueError:
                results.append("<INVALID>")
        return results

    def state_hash(self) -> str:
        payload = b"".join(key.encode() + self.params[key].tobytes() for key in sorted(self.params))
        return sha256(payload)


class MaskedSlotRKE(TinyTransformer):
    """Parallel RKE head with a causal convolution over latent output slots.

    The convolution is masked by construction: slot p can receive latent
    information only from slots <p. All slots are nevertheless computed in a
    fixed number of vectorized operations, and output remains tied to Ein.T.
    """

    def __init__(self, codec: ReversibleCodec, d_slot: int = 12, kernel_size: int = 4,
                 seed: int = 7):
        super().__init__(codec, d_slot=d_slot, seed=seed)
        if kernel_size < 1 or kernel_size >= codec.slots:
            raise ValueError("kernel_size must be between 1 and slots-1")
        self.kernel_size = kernel_size
        rng = np.random.default_rng(seed + 991)
        self.params["Wslot"] = rng.normal(0, 0.06, (kernel_size, d_slot, d_slot))
        self.params["slot_pos"] = rng.normal(0, 0.02, (codec.slots, d_slot))

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        p = self.params
        batch = len(features)
        x = (features @ p["Ein"]).reshape(batch, 3, self.d_model) + p["pos"][None, :, :]
        q, k, v = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(self.d_model)
        mask = np.triu(np.ones((3, 3), dtype=bool), 1)
        scores[:, mask] = -1e9
        attention = self._softmax(scores)
        attended = attention @ v
        hidden = x + attended @ p["Wo"]
        base_slots = hidden[:, -1, :].reshape(batch, self.codec.slots, self.d_slot)
        slot_pre = base_slots + p["slot_pos"][None, :, :]
        for lag in range(1, self.kernel_size + 1):
            slot_pre[:, lag:, :] += base_slots[:, :-lag, :] @ p["Wslot"][lag - 1]
        # Keep the tied-codebook geometry linear. A tanh bottleneck here
        # destroys prototype margins and performed substantially worse.
        slot_hidden = slot_pre
        logits = slot_hidden @ p["Ein"].T
        return logits, {"features": features, "x": x, "q": q, "k": k, "v": v,
                        "attention": attention, "attended": attended, "hidden": hidden,
                        "base_slots": base_slots, "slot_pre": slot_pre, "slot_hidden": slot_hidden}

    def loss_and_grads(self, features: np.ndarray, targets: np.ndarray,
                       loss_mask: np.ndarray | None = None) -> tuple[float, dict[str, np.ndarray]]:
        logits, c = self.forward(features)
        probabilities = self._softmax(logits)
        batch = len(features)
        chosen = probabilities[np.arange(batch)[:, None], np.arange(self.codec.slots)[None, :], targets]
        mask = np.ones_like(chosen) if loss_mask is None else np.asarray(loss_mask, dtype=np.float64)
        denominator = max(float(mask.sum()), 1.0)
        loss = float((-np.log(chosen + 1e-12) * mask).sum() / denominator)
        dlogits = probabilities
        dlogits[np.arange(batch)[:, None], np.arange(self.codec.slots)[None, :], targets] -= 1
        dlogits *= mask[:, :, None] / denominator

        p = self.params
        grads: dict[str, np.ndarray] = {}
        d_ein_output = (dlogits.reshape(-1, self.codec.width).T
                        @ c["slot_hidden"].reshape(-1, self.d_slot))
        dslot_hidden = dlogits @ p["Ein"]
        dslot_pre = dslot_hidden
        dbase = dslot_pre.copy()
        grads["slot_pos"] = dslot_pre.sum(axis=0)
        grads["Wslot"] = np.zeros_like(p["Wslot"])
        for lag in range(1, self.kernel_size + 1):
            source, destination_grad = c["base_slots"][:, :-lag, :], dslot_pre[:, lag:, :]
            grads["Wslot"][lag - 1] = np.einsum("bsi,bsj->ij", source, destination_grad)
            dbase[:, :-lag, :] += destination_grad @ p["Wslot"][lag - 1].T

        dhidden = np.zeros_like(c["hidden"])
        dhidden[:, -1, :] = dbase.reshape(batch, self.d_model)
        dx = dhidden.copy()
        dattn_out = dhidden @ p["Wo"].T
        grads["Wo"] = c["attended"].reshape(-1, self.d_model).T @ dhidden.reshape(-1, self.d_model)
        dattention = dattn_out @ c["v"].transpose(0, 2, 1)
        dv = c["attention"].transpose(0, 2, 1) @ dattn_out
        dscores = c["attention"] * (dattention - (dattention * c["attention"]).sum(axis=-1, keepdims=True))
        scale = math.sqrt(self.d_model)
        dq = dscores @ c["k"] / scale
        dk = dscores.transpose(0, 2, 1) @ c["q"] / scale
        flat_x = c["x"].reshape(-1, self.d_model)
        grads["Wq"] = flat_x.T @ dq.reshape(-1, self.d_model)
        grads["Wk"] = flat_x.T @ dk.reshape(-1, self.d_model)
        grads["Wv"] = flat_x.T @ dv.reshape(-1, self.d_model)
        dx += dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
        dslots = dx.reshape(batch, 3, self.codec.slots, self.d_slot)
        grads["Ein"] = (c["features"].reshape(-1, self.codec.width).T
                        @ dslots.reshape(-1, self.d_slot) + d_ein_output)
        grads["pos"] = dx.sum(axis=0)
        return loss, grads


class RefinedSlotRKE(TinyTransformer):
    """Two-pass parallel RKE: propose all slots, then refine them together.

    Pass one produces a distribution for every slot. Those distributions are
    mapped back through the tied codebook into expected symbol embeddings.
    Pass two uses only earlier proposals through a masked convolution and emits
    every corrected slot in one vectorized operation. There is still no
    separately learned symbol or vocabulary output matrix.
    """

    def __init__(self, codec: ReversibleCodec, d_slot: int = 12, kernel_size: int = 8,
                 auxiliary_weight: float = 0.25, proposal_temperature: float = 0.25,
                 seed: int = 7):
        super().__init__(codec, d_slot=d_slot, seed=seed)
        if kernel_size < 1 or kernel_size >= codec.slots:
            raise ValueError("kernel_size must be between 1 and slots-1")
        if proposal_temperature <= 0:
            raise ValueError("proposal_temperature must be positive")
        self.kernel_size, self.auxiliary_weight = kernel_size, auxiliary_weight
        self.proposal_temperature = proposal_temperature
        # A zero residual makes the initial second pass exactly equal to the
        # proposal path; refinement must earn every deviation during training.
        self.params["Wref"] = np.zeros((kernel_size, d_slot, d_slot))
        self.params["refine_pos"] = np.zeros((codec.slots, d_slot))

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        p = self.params
        batch = len(features)
        x = (features @ p["Ein"]).reshape(batch, 3, self.d_model) + p["pos"][None, :, :]
        q, k, v = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(self.d_model)
        mask = np.triu(np.ones((3, 3), dtype=bool), 1)
        scores[:, mask] = -1e9
        attention = self._softmax(scores)
        attended = attention @ v
        hidden = x + attended @ p["Wo"]
        base_slots = hidden[:, -1, :].reshape(batch, self.codec.slots, self.d_slot)

        proposal_logits = base_slots @ p["Ein"].T
        proposal_probs = self._softmax(proposal_logits)
        proposal_ids = proposal_probs.argmax(axis=-1)
        refine_probs = self._softmax(proposal_logits / self.proposal_temperature)
        # Sharpened probabilities approximate discrete proposed symbols while
        # preserving an exact differentiable path into the proposal pass.
        expected = refine_probs @ p["Ein"]
        refined_slots = base_slots + p["refine_pos"][None, :, :]
        for lag in range(1, self.kernel_size + 1):
            refined_slots[:, lag:, :] += expected[:, :-lag, :] @ p["Wref"][lag - 1]
        logits = refined_slots @ p["Ein"].T
        return logits, {"features": features, "x": x, "q": q, "k": k, "v": v,
                        "attention": attention, "attended": attended, "hidden": hidden,
                        "base_slots": base_slots, "proposal_logits": proposal_logits,
                        "proposal_probs": proposal_probs, "proposal_ids": proposal_ids,
                        "refine_probs": refine_probs, "expected": expected,
                        "refined_slots": refined_slots}

    def loss_and_grads(self, features: np.ndarray, targets: np.ndarray,
                       loss_mask: np.ndarray | None = None) -> tuple[float, dict[str, np.ndarray]]:
        logits, c = self.forward(features)
        probabilities = self._softmax(logits)
        proposal_probabilities = c["proposal_probs"]
        batch = len(features)
        rows, slots = np.arange(batch)[:, None], np.arange(self.codec.slots)[None, :]
        chosen = probabilities[rows, slots, targets]
        proposal_chosen = proposal_probabilities[rows, slots, targets]
        mask = np.ones_like(chosen) if loss_mask is None else np.asarray(loss_mask, dtype=np.float64)
        denominator = max(float(mask.sum()), 1.0)
        final_loss = (-np.log(chosen + 1e-12) * mask).sum() / denominator
        proposal_loss = (-np.log(proposal_chosen + 1e-12) * mask).sum() / denominator
        loss = float(final_loss + self.auxiliary_weight * proposal_loss)

        dlogits = probabilities
        dlogits[rows, slots, targets] -= 1
        dlogits *= mask[:, :, None] / denominator
        dproposal_aux = proposal_probabilities.copy()
        dproposal_aux[rows, slots, targets] -= 1
        dproposal_aux *= self.auxiliary_weight * mask[:, :, None] / denominator

        p = self.params
        grads: dict[str, np.ndarray] = {}
        d_ein_output = (dlogits.reshape(-1, self.codec.width).T
                        @ c["refined_slots"].reshape(-1, self.d_slot))
        drefined = dlogits @ p["Ein"]
        dbase = drefined.copy()
        dexpected = np.zeros_like(c["expected"])
        grads["refine_pos"] = drefined.sum(axis=0)
        grads["Wref"] = np.zeros_like(p["Wref"])
        for lag in range(1, self.kernel_size + 1):
            source, destination_grad = c["expected"][:, :-lag, :], drefined[:, lag:, :]
            grads["Wref"][lag - 1] = np.einsum("bsi,bsj->ij", source, destination_grad)
            dexpected[:, :-lag, :] += destination_grad @ p["Wref"][lag - 1].T

        d_ein_expected = (c["refine_probs"].reshape(-1, self.codec.width).T
                          @ dexpected.reshape(-1, self.d_slot))
        drefine_probs = dexpected @ p["Ein"].T
        dproposal_refine = c["refine_probs"] * (
            drefine_probs - (drefine_probs * c["refine_probs"]).sum(axis=-1, keepdims=True))
        dproposal_refine /= self.proposal_temperature
        dproposal = dproposal_aux + dproposal_refine
        dbase += dproposal @ p["Ein"]
        d_ein_proposal = (dproposal.reshape(-1, self.codec.width).T
                          @ c["base_slots"].reshape(-1, self.d_slot))

        dhidden = np.zeros_like(c["hidden"])
        dhidden[:, -1, :] = dbase.reshape(batch, self.d_model)
        dx = dhidden.copy()
        dattn_out = dhidden @ p["Wo"].T
        grads["Wo"] = c["attended"].reshape(-1, self.d_model).T @ dhidden.reshape(-1, self.d_model)
        dattention = dattn_out @ c["v"].transpose(0, 2, 1)
        dv = c["attention"].transpose(0, 2, 1) @ dattn_out
        dscores = c["attention"] * (dattention - (dattention * c["attention"]).sum(axis=-1, keepdims=True))
        scale = math.sqrt(self.d_model)
        dq = dscores @ c["k"] / scale
        dk = dscores.transpose(0, 2, 1) @ c["q"] / scale
        flat_x = c["x"].reshape(-1, self.d_model)
        grads["Wq"] = flat_x.T @ dq.reshape(-1, self.d_model)
        grads["Wk"] = flat_x.T @ dk.reshape(-1, self.d_model)
        grads["Wv"] = flat_x.T @ dv.reshape(-1, self.d_model)
        dx += dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
        dslots = dx.reshape(batch, 3, self.codec.slots, self.d_slot)
        grads["Ein"] = (c["features"].reshape(-1, self.codec.width).T
                        @ dslots.reshape(-1, self.d_slot)
                        + d_ein_output + d_ein_expected + d_ein_proposal)
        grads["pos"] = dx.sum(axis=0)
        return loss, grads


def accuracy(expected: list[str], predicted: list[str]) -> dict[str, float]:
    exact = sum(a == b for a, b in zip(expected, predicted)) / max(1, len(expected))
    total_chars = correct_chars = 0
    for truth, guess in zip(expected, predicted):
        width = max(len(truth), len(guess), 1)
        total_chars += width
        correct_chars += sum(i < len(truth) and i < len(guess) and truth[i] == guess[i] for i in range(width))
    return {"exact_match": exact, "character_accuracy": correct_chars / total_chars}


def train_model(codec: ReversibleCodec, train: list[str], test: list[str], seed: int = 7,
                steps: int = 900, batch_size: int = 64) -> tuple[TinyTransformer, list[dict[str, float]], dict[str, Any]]:
    rng = np.random.default_rng(seed + 1)
    model = TinyTransformer(codec, seed=seed)
    optimizer = Adam(model.params)
    curve = []
    noise_pool = train
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(train), size=batch_size)
        noise_indices = rng.integers(0, len(noise_pool), size=batch_size)
        payloads = [train[int(i)] for i in indices]
        distractors = [noise_pool[int(i)] for i in noise_indices]
        features = model.features(payloads, distractors)
        targets = np.stack([codec.ids(text) for text in payloads])
        loss, grads = model.loss_and_grads(features, targets)
        optimizer.update(model.params, grads)
        if step == 1 or step % 50 == 0:
            sample_train = train[:min(200, len(train))]
            sample_test = test[:min(200, len(test))]
            train_pred = model.predict(sample_train, list(reversed(sample_train)))
            test_pred = model.predict(sample_test, list(reversed(sample_test)))
            curve.append({"step": step, "loss": loss,
                          "train_exact": accuracy(sample_train, train_pred)["exact_match"],
                          "test_exact": accuracy(sample_test, test_pred)["exact_match"]})
    train_pred = model.predict(train, list(reversed(train)))
    test_pred = model.predict(test, list(reversed(test)))
    metrics = {
        "train": accuracy(train, train_pred), "held_out_oov": accuracy(test, test_pred),
        "examples": [{"target": truth, "prediction": guess, "exact": truth == guess}
                     for truth, guess in list(zip(test, test_pred))[:20]],
    }
    return model, curve, metrics


def save_npz(path: Path, model: TinyTransformer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model.params)
