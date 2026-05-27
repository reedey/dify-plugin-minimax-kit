import unittest
from unittest.mock import patch

from tools import base


class MiniMaxBaseToolTest(unittest.TestCase):
    def test_text_to_image_uses_global_endpoint_and_512_square_payload(self):
        captured_request = {}

        def fake_request(method, url, **kwargs):
            captured_request["method"] = method
            captured_request["url"] = url
            captured_request["kwargs"] = kwargs
            return object()

        tool = base.MiniMaxBaseTool(api_key="test-api-key", group_id="test-group-id")
        with patch.object(base.requests, "request", fake_request):
            tool.text_to_image(
                model="image-01",
                prompt="square image",
                aspect_ratio="1:1",
                response_format="url",
                prompt_optimizer=True,
                n=1,
            )

        self.assertEqual(captured_request["method"], "POST")
        self.assertEqual(
            captured_request["url"], "https://api.minimax.io/v1/image_generation"
        )
        payload = captured_request["kwargs"]["json"]
        self.assertEqual(payload["width"], 512)
        self.assertEqual(payload["height"], 512)
        self.assertNotIn("aspect_ratio", payload)

    def test_text_to_image_keeps_non_square_aspect_ratio_payload(self):
        captured_request = {}

        def fake_request(method, url, **kwargs):
            captured_request["method"] = method
            captured_request["url"] = url
            captured_request["kwargs"] = kwargs
            return object()

        tool = base.MiniMaxBaseTool(api_key="test-api-key", group_id="test-group-id")
        with patch.object(base.requests, "request", fake_request):
            tool.text_to_image(
                model="image-01",
                prompt="wide image",
                aspect_ratio="16:9",
                response_format="url",
                prompt_optimizer=False,
                n=2,
            )

        payload = captured_request["kwargs"]["json"]
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertNotIn("width", payload)
        self.assertNotIn("height", payload)

    def test_text_to_image_adds_subject_reference_when_url_is_provided(self):
        captured_request = {}

        def fake_request(method, url, **kwargs):
            captured_request["method"] = method
            captured_request["url"] = url
            captured_request["kwargs"] = kwargs
            return object()

        tool = base.MiniMaxBaseTool(api_key="test-api-key", group_id="test-group-id")
        with patch.object(base.requests, "request", fake_request):
            tool.text_to_image(
                model="image-01",
                prompt="use this character reference",
                aspect_ratio="1:1",
                response_format="url",
                prompt_optimizer=False,
                n=1,
                reference_image_url="https://example.com/reference.jpg",
            )

        payload = captured_request["kwargs"]["json"]
        self.assertEqual(
            payload["subject_reference"],
            [
                {
                    "type": "character",
                    "image_file": "https://example.com/reference.jpg",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
