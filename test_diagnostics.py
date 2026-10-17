"""Offline API integration tests: no real key or network required."""
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from diagnostics import Diagnostics, _parse_result


class TestDiagnostics(unittest.TestCase):
    @patch.dict(os.environ, {"NRP_API_KEY": "test-key"})
    @patch("diagnostics.urlopen")
    def test_request_and_result(self, urlopen):
        response = {"choices": [{"message": {"content":
            '{"disease":"Cancer","probability":0.4367}'}}]}
        urlopen.return_value.__enter__.return_value = io.StringIO(json.dumps(response))
        answer = Diagnostics().diagnose("No", "Yes", "Abnormal", "Absent")
        self.assertEqual(answer, ["Cancer", 0.4367])
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        evidence = payload["messages"][1]["content"]
        self.assertIn('"Asia": false', evidence)
        self.assertIn('"Dyspnea": false', evidence)
        self.assertIn("P(TB=True|Asia=True)=0.05", payload["messages"][0]["content"])

    @patch.dict(os.environ, {"NRP_API_KEY": "test-key"})
    @patch("diagnostics.urlopen")
    def test_unknowns_omitted(self, urlopen):
        response = {"choices": [{"message": {"content":
            '{"disease":"Bronchitis","probability":0.45}'}}]}
        urlopen.return_value.__enter__.return_value = io.StringIO(json.dumps(response))
        Diagnostics().diagnose("NA", "NA", "NA", "NA")
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["messages"][1]["content"], "Observed evidence: {}")

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            Diagnostics().diagnose("Sometimes", "NA", "NA", "NA")

    @patch.dict(os.environ, {"NRP_API_KEY": ""})
    def test_missing_key(self):
        with self.assertRaisesRegex(RuntimeError, "NRP_API_KEY"):
            Diagnostics().diagnose("NA", "NA", "NA", "NA")

    def test_invalid_model_outputs(self):
        for disease, probability in [("Flu", .5), ("TB", True),
                                     ("TB", "0.5"), ("TB", 1.1), ("TB", float('nan'))]:
            with self.subTest(disease=disease, probability=probability):
                with self.assertRaises(ValueError):
                    _parse_result(json.dumps(dict(disease=disease, probability=probability)))

    @patch.dict(os.environ, {"NRP_API_KEY": "test-key"})
    @patch("diagnostics.urlopen", side_effect=HTTPError("url", 401, "Unauthorized", {}, None))
    def test_auth_error(self, urlopen):
        with self.assertRaisesRegex(RuntimeError, "Check your NRP API key"):
            Diagnostics().diagnose("NA", "NA", "NA", "NA")


if __name__ == "__main__":
    unittest.main()
