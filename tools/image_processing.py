from dataclasses import dataclass
from io import BytesIO

from PIL import Image


EGGI_TARGET_WIDTH = 240


@dataclass(frozen=True)
class ResizedImage:
    blob: bytes
    width: int
    height: int
    mime_type: str


def resize_image_to_eggi_width(image_blob: bytes) -> ResizedImage:
    source = Image.open(BytesIO(image_blob))
    source.load()

    target_width = EGGI_TARGET_WIDTH
    target_height = round(source.height * target_width / source.width)
    resized = source.convert("RGB").resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )

    output = BytesIO()
    resized.save(output, format="JPEG", quality=90, optimize=True)
    return ResizedImage(
        blob=output.getvalue(),
        width=target_width,
        height=target_height,
        mime_type="image/jpeg",
    )
