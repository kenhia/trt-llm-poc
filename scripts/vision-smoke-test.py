#!/usr/bin/env python3
"""
scripts/vision-smoke-test.py — Vision pipeline smoke test.

Uses the Triton HTTP binary extension to call ensemble_vision.

Default mode: sends synthetic CLIP-normalised noise (no dependencies beyond numpy).
Image mode:   pass --image <path> to send a real image (requires Pillow / python3-pil).
              The image is resized to 336×336 and CLIP-normalised before sending.

Usage:
    python3 vision-smoke-test.py                     # synthetic noise
    python3 vision-smoke-test.py --image foo.png     # real image
    python3 vision-smoke-test.py --image foo.png --prompt "What colour is the water?"

For LLaVA 1.5 (model_type == 'llava'), the client must supply pre-processed FP16 pixel
values; raw image bytes are not supported server-side.

Exit code: 0 = PASS, 1 = FAIL.
"""
import argparse
import json
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np

# CLIP ViT-L/14@336px normalisation constants
_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_CLIP_SIZE = 336


def make_clip_image(seed: int = 42) -> np.ndarray:
    """Return a plausible CLIP-normalised 336×336 image as FP16 (synthetic noise).
    Shape: (1, 1, 3, 336, 336) — [batch, num_images, C, H, W]
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(1, 1, 3, _CLIP_SIZE, _CLIP_SIZE)).astype(np.float16)


def load_clip_image(path: str) -> np.ndarray:
    """Load an image file, resize to 336×336, CLIP-normalise, return FP16.
    Shape: (1, 1, 3, 336, 336)
    Requires: python3-pil (apt) or Pillow (pip).
    """
    try:
        from PIL import Image
    except ImportError:
        sys.exit("ERROR: Pillow is required for --image mode.\n"
                 "Install with:  sudo apt-get install python3-pil")
    img = Image.open(path).convert("RGB").resize(
        (_CLIP_SIZE, _CLIP_SIZE), Image.BICUBIC
    )
    arr = np.array(img, dtype=np.float32) / 255.0          # (336,336,3) in [0,1]
    arr = (arr - _CLIP_MEAN) / _CLIP_STD                   # CLIP normalise
    arr = arr.transpose(2, 0, 1)                            # (3,336,336) CHW
    return arr[np.newaxis, np.newaxis].astype(np.float16)   # (1,1,3,336,336)


def run(image_path: str | None, prompt: str) -> None:
    base_url = "http://localhost:8000"

    # 1. Verify ensemble_vision is ready.
    try:
        with urllib.request.urlopen(f"{base_url}/v2/models/ensemble_vision/ready", timeout=5):
            pass
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ensemble_vision not ready: {e}") from e
    print("[vision-smoke] ensemble_vision is ready.")

    # 2. Build pixel tensor.
    if image_path:
        print(f"[vision-smoke] Loading image: {image_path}")
        pixel_values = load_clip_image(image_path)
    else:
        print("[vision-smoke] Using synthetic CLIP-normalised noise (no --image supplied).")
        pixel_values = make_clip_image()

    # Serialise inputs as raw bytes:
    #   BYTES tensor: each element = 4-byte LE length prefix + UTF-8 data
    #   FP16 tensor:  raw little-endian float16 bytes
    #   INT32 tensor: raw 4-byte LE integer
    prompt_bytes = prompt.encode("utf-8")
    text_raw     = struct.pack("<I", len(prompt_bytes)) + prompt_bytes
    image_raw    = pixel_values.tobytes()      # FP16 → 2 bytes/element
    tokens_raw   = np.int32(128).tobytes()     # 4 bytes

    header = {
        "inputs": [
            {"name": "text_input",  "shape": [1, 1],               "datatype": "BYTES",
             "parameters": {"binary_data_size": len(text_raw)}},
            {"name": "image_input", "shape": list(pixel_values.shape), "datatype": "FP16",
             "parameters": {"binary_data_size": len(image_raw)}},
            {"name": "max_tokens",  "shape": [1, 1],               "datatype": "INT32",
             "parameters": {"binary_data_size": len(tokens_raw)}},
        ],
        "outputs": [{"name": "text_output", "parameters": {"binary_data": True}}],
    }
    header_bytes = json.dumps(header).encode("utf-8")
    body = header_bytes + text_raw + image_raw + tokens_raw

    req = urllib.request.Request(
        f"{base_url}/v2/models/ensemble_vision/infer",
        data=body, method="POST",
        headers={
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(len(header_bytes)),
        },
    )
    print("[vision-smoke] Sending inference request ...")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            ohl = int(resp.headers.get("Inference-Header-Content-Length", len(raw)))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Inference HTTP {e.code}: {e.read().decode()}") from e

    print(f"[vision-smoke] Response received in {time.monotonic() - t0:.1f}s")

    resp_hdr = json.loads(raw[:ohl])
    bindata  = raw[ohl:]

    for out in resp_hdr.get("outputs", []):
        if out["name"] == "text_output":
            idx, texts = 0, []
            while idx + 4 <= len(bindata):
                slen = struct.unpack_from("<I", bindata, idx)[0]
                idx += 4
                texts.append(bindata[idx: idx + slen].decode("utf-8", errors="replace"))
                idx += slen
            print(f"[Caption] {''.join(texts).strip() or '(empty response)'}")
            return

    print("[vision-smoke] WARNING: text_output not found")
    print("[vision-smoke] response:", json.dumps(resp_hdr, indent=2)[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA 1.5 vision smoke test")
    parser.add_argument("--image",  metavar="PATH", help="Image file to caption (png/jpg/…)")
    parser.add_argument("--prompt", default="Describe this image briefly.",
                        help="Text prompt (default: 'Describe this image briefly.')")
    args = parser.parse_args()

    try:
        run(args.image, args.prompt)
        print("[vision-smoke] PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"[vision-smoke] FAILED: {e}")
        sys.exit(1)
