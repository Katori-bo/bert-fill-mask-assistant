"""Interface checks use a controlled backend; they do not evaluate BERT."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


class DemoPredictor:
    def predict(self, text, top_k):
        return {"input": text, "top_k": top_k, "model": "interface-test-double",
                "predictions": [{"token_str": "paris", "score": 0.9,
                                 "sequence": "the capital of france is paris."}]}


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Install Streamlit to run interface tests")
class InterfaceTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()
    def test_form_result_and_download(self):
        from streamlit.testing.v1 import AppTest
        with patch("predictor.MaskPredictor", DemoPredictor):
            app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.subheader[0].value, "Suggested replacements")
            self.assertEqual(len(app.dataframe), 1)
            self.assertEqual(len(app.download_button), 1)
    def test_missing_mask_shows_error(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        app.text_area[0].set_value("No missing word.")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("exactly one", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
