"""Offline API integration tests: no real key or network required."""
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from diagnostics import Diagnostics, _parse_result


@patch.dict(os.environ, {"LLM_API_KEY_ENV": "LLM_API_KEY"})
class TestDiagnostics(unittest.TestCase):
    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
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

    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
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

    @patch.dict(os.environ, {"LLM_API_KEY": ""})
    def test_missing_key(self):
        with self.assertRaisesRegex(RuntimeError, "LLM_API_KEY"):
            Diagnostics().diagnose("NA", "NA", "NA", "NA")

    def test_invalid_model_outputs(self):
        for disease, probability in [("Flu", .5), ("TB", True),
                                     ("TB", "0.5"), ("TB", 1.1), ("TB", float('nan'))]:
            with self.subTest(disease=disease, probability=probability):
                with self.assertRaises(ValueError):
                    _parse_result(json.dumps(dict(disease=disease, probability=probability)))

    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
    @patch("diagnostics.urlopen", side_effect=HTTPError("url", 401, "Unauthorized", {}, None))
    def test_auth_error(self, urlopen):
        with self.assertRaisesRegex(RuntimeError, "Check your API key"):
            Diagnostics().diagnose("NA", "NA", "NA", "NA")


class TestFailureHandling(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"LLM_API_KEY_ENV": "LLM_API_KEY"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
    @patch("diagnostics.time.sleep")
    @patch("diagnostics.urlopen")
    def test_temporary_failure_retried_once(self, urlopen, sleep):
        from unittest.mock import MagicMock
        success = MagicMock()
        success.__enter__.return_value = io.StringIO(json.dumps({"choices": [
            {"message": {"content": '{"disease":"TB","probability":0.6}'}}]}))
        urlopen.side_effect = [HTTPError("url", 503, "Unavailable", {}, None), success]
        self.assertEqual(Diagnostics().diagnose("NA", "NA", "NA", "NA"), ["TB", .6])
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
    @patch("diagnostics.time.sleep")
    @patch("diagnostics.urlopen", side_effect=HTTPError("url", 429, "Limit", {}, None))
    def test_retries_are_bounded(self, urlopen, sleep):
        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            Diagnostics().diagnose("NA", "NA", "NA", "NA")
        self.assertEqual(urlopen.call_count, 2)

    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
    @patch("diagnostics.urlopen")
    def test_non_json_response(self, urlopen):
        urlopen.return_value.__enter__.return_value = io.StringIO("<html>Service down</html>")
        with self.assertRaisesRegex(RuntimeError, "non-JSON"):
            Diagnostics().diagnose("NA", "NA", "NA", "NA")
        self.assertEqual(urlopen.call_count, 1)

    @patch.dict(os.environ, {"LLM_API_KEY_ENV": "GRADER_KEY", "GRADER_KEY": "test-key"})
    @patch("diagnostics.urlopen")
    def test_configurable_key_variable(self, urlopen):
        urlopen.return_value.__enter__.return_value = io.StringIO(json.dumps({"choices": [
            {"message": {"content": '{"disease":"Bronchitis","probability":0.45}'}}]}))
        self.assertEqual(Diagnostics().diagnose("NA", "NA", "NA", "NA"), ["Bronchitis", .45])
        self.assertEqual(urlopen.call_args.args[0].get_header("Authorization"), "Bearer test-key")

    @patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_MODEL": "gpt-5-nano",
                             "LLM_BASE_URL": "https://api.openai.com/v1"})
    @patch("diagnostics.urlopen")
    def test_openai_nano_request(self, urlopen):
        urlopen.return_value.__enter__.return_value = io.StringIO(json.dumps({"choices": [
            {"message": {"content": '{"disease":"Cancer","probability":0.436681}'}}]}))
        Diagnostics().diagnose("No", "Yes", "Abnormal", "Absent")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "gpt-5-nano")
        self.assertNotIn("temperature", payload)

    def test_extra_output_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            _parse_result('{"disease":"TB","probability":0.5,"explanation":"extra"}')


if __name__ == "__main__":
    unittest.main()
