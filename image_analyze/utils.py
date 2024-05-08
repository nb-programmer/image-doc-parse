from PIL.Image import Image
from datauri import DataURI
import io


def io2uri(f: io.BytesIO, format_mime: str = "image/png"):
    return DataURI.make(
        format_mime, charset=None, base64=True, data=f.getvalue()
    )


def img2uri(img: Image) -> str:
    with io.BytesIO() as f:
        img.save(f, format="png")
        return io2uri(f, "image/png")
