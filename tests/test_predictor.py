import unittest
from types import SimpleNamespace

from predictor import MaskPredictor, validate_input


class Tokenizer:
    mask_token_id = 999
    model_max_length = 20
    def __call__(self, text):
        return {"input_ids": [999 if word == "[MASK]" else 1 for word in text.split()]}


class Pipeline:
    tokenizer = Tokenizer()
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision"))
    def __call__(self, text, top_k):
        return [{"token": 42, "token_str": "water", "score": 0.5,
                 "sequence": text.replace("[MASK]", "water")}]


class InputTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(validate_input(" A [MASK] . ", 5), "A [MASK] .")
    def test_invalid_input(self):
        for text, k in [(" ", 5), ("No mask", 5), ("[MASK] [MASK]", 5),
                        ("A [MASK] .", 0), ("A [MASK] .", 11), ("A [MASK] .", True)]:
            with self.subTest(text=text, k=k), self.assertRaises(ValueError):
                validate_input(text, k)


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.predictor = MaskPredictor()
        self.predictor._pipeline = Pipeline()
    def test_output_contract(self):
        result = self.predictor.predict("A glass of [MASK] .", 5)
        self.assertEqual(result["revision"], "test-revision")
        self.assertEqual(result["predictions"][0]["token_str"], "water")
        self.assertIsInstance(result["predictions"][0]["score"], float)
    def test_long_input_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict("word " * 25 + "[MASK]")
    def test_encoded_mask_required(self):
        class WrongTokenizer(Tokenizer):
            mask_token_id = 555
        self.predictor._pipeline.tokenizer = WrongTokenizer()
        with self.assertRaises(ValueError):
            self.predictor.predict("A [MASK] .")


if __name__ == "__main__":
    unittest.main()
