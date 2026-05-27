import unittest
from io import BytesIO

from PIL import Image

from tools.image_processing import resize_image_to_eggi_width


def make_jpeg(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


class EggiImageResizeTest(unittest.TestCase):
    def test_square_image_resizes_to_240_by_240(self):
        result = resize_image_to_eggi_width(make_jpeg(512, 512))

        self.assertEqual(result.width, 240)
        self.assertEqual(result.height, 240)
        self.assertEqual(result.mime_type, "image/jpeg")

        resized = Image.open(BytesIO(result.blob))
        self.assertEqual(resized.size, (240, 240))

    def test_portrait_image_keeps_width_240_and_scales_height(self):
        result = resize_image_to_eggi_width(make_jpeg(576, 1024))

        self.assertEqual(result.width, 240)
        self.assertEqual(result.height, 427)

        resized = Image.open(BytesIO(result.blob))
        self.assertEqual(resized.size, (240, 427))

    def test_landscape_image_keeps_width_240_and_scales_height(self):
        result = resize_image_to_eggi_width(make_jpeg(1024, 576))

        self.assertEqual(result.width, 240)
        self.assertEqual(result.height, 135)

        resized = Image.open(BytesIO(result.blob))
        self.assertEqual(resized.size, (240, 135))


if __name__ == "__main__":
    unittest.main()
