"""Images attached to chat messages: strict decoding and validation of what the browser sends.

The browser re-encodes every photo through a canvas before sending it (which also strips EXIF, GPS
included), but the server trusts nothing it receives: the type comes from the file's first bytes,
never from a client-provided media type, and size and count are capped.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass
from typing import List, Optional

from engine import InvalidUserMessage

MAX_IMAGES_PER_MESSAGE = 4
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # decoded; the Claude API refuses images above 5 MB
# Extension -> media type. The extension is part of the stored name, so it alone decides the
# Content-Type the file is served with.
MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
IMAGE_NAME_PATTERN = r"[0-9a-f]{64}\.(?:png|jpg|gif|webp)"  # sha256 of the bytes + extension


class InvalidImage(InvalidUserMessage):
    """An attached image is malformed, of an unsupported type, too big, or one too many."""


def sniff_extension(data: bytes) -> Optional[str]:
    """File extension from the magic bytes, or None for anything that is not png/jpeg/gif/webp."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


@dataclass(frozen=True)
class Image:
    data: bytes
    extension: str

    @property
    def name(self) -> str:
        """Content-addressed file name: the same photo sent twice is stored once."""
        return f"{hashlib.sha256(self.data).hexdigest()}.{self.extension}"

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES[self.extension]


def media_type_of(name: str) -> str:
    """Media type of a stored image name (its extension)."""
    return MEDIA_TYPES[name.rsplit(".", 1)[1]]


def parse_images(raw: object) -> List[Image]:
    """Decode the ``images`` field of a chat message: a list of base64 strings (None: no images).
    Raises InvalidImage with a message the UI can show."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise InvalidImage("images must be a list of base64 strings")
    if len(raw) > MAX_IMAGES_PER_MESSAGE:
        raise InvalidImage(f"at most {MAX_IMAGES_PER_MESSAGE} images per message")
    images = []
    for number, item in enumerate(raw, 1):
        if len(item) > (MAX_IMAGE_BYTES + 2) // 3 * 4:  # cheap check before decoding
            raise InvalidImage(f"image {number} exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
        try:
            data = base64.b64decode(item, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidImage(f"image {number} is not valid base64") from exc
        if not data:
            raise InvalidImage(f"image {number} is empty")
        extension = sniff_extension(data)
        if extension is None:
            raise InvalidImage(f"image {number} is not a PNG, JPEG, GIF or WebP file")
        images.append(Image(data=data, extension=extension))
    return images
