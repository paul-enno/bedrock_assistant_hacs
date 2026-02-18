"""Image processing utilities for Bedrock integration."""

from __future__ import annotations

from io import BytesIO
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError
from urllib.request import urlopen

import PIL.Image
from PIL.Image import Image

from homeassistant.exceptions import HomeAssistantError

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


async def build_converse_prompt_content(image: Image) -> dict[str, Any]:
    """Convert PIL Image to Bedrock converse format."""
    buffered = BytesIO()
    image.save(buffered, format=image.format)
    file_image_byte = buffered.getvalue()
    file_image_format = (
        image.format if image.format in ["jpeg", "png", "gif", "webp"] else "jpeg"
    )
    return {
        "image": {
            "format": file_image_format,
            "source": {"bytes": file_image_byte},
        },
    }


class ImageProcessor:
    """Handle image loading and validation."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the image processor."""
        self.hass = hass

    async def load_image_from_file(self, filename: str) -> Image:
        """Load and validate an image from a file path."""
        if not self.hass.config.is_allowed_path(filename):
            raise HomeAssistantError(
                f"Cannot read `{filename}`, no access to path; "
                "`allowlist_external_dirs` may need to be adjusted in "
                "`configuration.yaml`"
            )

        if not Path(filename).exists():
            raise HomeAssistantError(f"`{filename}` does not exist")

        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None or not mime_type.startswith("image"):
            raise HomeAssistantError(f"`{filename}` is not an image")

        return await self.hass.async_add_executor_job(PIL.Image.open, filename)

    async def load_image_from_url(self, url: str) -> Image:
        """Load and validate an image from a URL."""
        mime_type, _ = mimetypes.guess_type(url)
        if mime_type is None or not mime_type.startswith("image"):
            raise HomeAssistantError(f"`{url}` is not an image")

        try:
            opened_url = await self.hass.async_add_executor_job(urlopen, url)
            return PIL.Image.open(opened_url)
        except HTTPError as error:
            raise HomeAssistantError(
                f"Cannot access file from `{url}`. "
                f"Status: `{error.status}`, Reason: `{error.reason}`"
            ) from error
