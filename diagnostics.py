"""Project 2: diagnose using an NRP-hosted LLM, without local BN inference."""

import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


NETWORK_PROMPT = """
Solve this educational probability problem using ONLY this Bayesian network.
All eight variables are Boolean. The following are probabilities of True;
probabilities of False are one minus the stated probability.

Asia and Smoking are independent roots:
P(Asia=True)=0.01; P(Smoking=True)=0.5.
TB has parent Asia: P(TB=True|Asia=True)=0.05;
P(TB=True|Asia=False)=0.01.
Cancer has parent Smoking: P(Cancer=True|Smoking=True)=0.1;
P(Cancer=True|Smoking=False)=0.01.
Bronchitis has parent Smoking: P(Bronchitis=True|Smoking=True)=0.6;
P(Bronchitis=True|Smoking=False)=0.3.
TBorC is deterministically (TB OR Cancer): its probability of True is
1 for (TB,Cancer)=(True,True),(True,False),(False,True), and 0 otherwise.
Xray has parent TBorC: P(Xray=True|TBorC=True)=0.99;
P(Xray=True|TBorC=False)=0.05.
Dyspnea has parents (TBorC,Bronchitis): P(Dyspnea=True|parents) is
0.9 for (True,True), 0.7 for (True,False),
0.8 for (False,True), 0.1 for (False,False).

The joint distribution factors as:
P(A)P(S)P(T|A)P(C|S)P(B|S)P(E|T,C)P(X|E)P(D|E,B),
where A=Asia,S=Smoking,T=TB,C=Cancer,B=Bronchitis,E=TBorC,
X=Xray,D=Dyspnea.

For the supplied evidence, calculate P(TB=True|evidence),
P(Cancer=True|evidence), and P(Bronchitis=True|evidence).
Marginalize ALL unobserved variables, including unobserved roots.
For each disease, sum the joint probabilities consistent with the evidence
and that disease being True, then divide by P(evidence).
Diseases may coexist: do NOT normalize the three disease probabilities
against each other. Do not substitute general medical knowledge.
Select the disease with the largest posterior probability. In an exact tie,
prefer TB, then Cancer, then Bronchitis.
Return ONLY a JSON object with exactly these keys:
{"disease": "TB or Cancer or Bronchitis", "probability": 0.123456}
Use one exact disease name and a numeric probability between 0 and 1,
with at least six decimal places when needed. No markdown or explanation.
"""


def _convert(value, options, name):
    if value == "NA":
        return None
    if value not in options:
        raise ValueError(f"{name} must be one of {tuple(options)} or 'NA'")
    return options[value]


def _parse_result(content):
    """Accept JSON, tolerating a single Markdown code fence, then validate it."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned no answer text.")
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("The model must return a JSON object.")
    disease = result.get("disease")
    probability = result.get("probability")
    if disease not in ("TB", "Cancer", "Bronchitis"):
        raise ValueError("The model returned an invalid disease name.")
    if (type(probability) not in (int, float)
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("The model probability must be a number between 0 and 1.")
    return [disease, float(probability)]


class Diagnostics:
    """Keep the Project 1 interface; delegate probability inference to the LLM."""

    def __init__(self):
        self.model = os.environ.get("NRP_MODEL", "gpt-oss")
        self.base_url = os.environ.get(
            "NRP_BASE_URL", "https://ellm.nrp-nautilus.io/v1"
        ).rstrip("/")

    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):
        observed = {
            "Asia": _convert(visit_to_asia, {"Yes": True, "No": False}, "visit_to_asia"),
            "Smoking": _convert(smoking, {"Yes": True, "No": False}, "smoking"),
            "Xray": _convert(xray_result, {"Abnormal": True, "Normal": False}, "xray_result"),
            "Dyspnea": _convert(dyspnea, {"Present": True, "Absent": False}, "dyspnea"),
        }
        # NA means unobserved, not False. Omit it from the evidence.
        evidence = {key: value for key, value in observed.items() if value is not None}
        api_key = os.environ.get("NRP_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set NRP_API_KEY in your terminal before running the program.")
        print("Evidence:", json.dumps(evidence), flush=True)
        print(f"Requesting Bayesian inference from {self.model}...", flush=True)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": NETWORK_PROMPT},
                {"role": "user", "content": "Observed evidence: " + json.dumps(evidence)},
            ],
            "temperature": 0,
        }
        # Use the documented OpenAI-compatible HTTP API. No external packages.
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + api_key,
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                completion = json.load(response)
        except HTTPError as error:
            hints = {401: "Check your NRP API key.", 403: "Check NRP access permissions.",
                     404: "Check NRP_BASE_URL and NRP_MODEL.",
                     429: "NRP is rate limiting requests; try again later."}
            raise RuntimeError(
                f"NRP HTTP {error.code}. " + hints.get(error.code, "Try again later.")
            ) from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Could not reach NRP or the request timed out. Check your connection and retry.") from None

        try:
            choice = completion["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("The model response was truncated.")
            answer = _parse_result(choice["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid NRP response: {error}") from None
        print("Diagnosis:", answer, flush=True)
        return answer


if __name__ == "__main__":
    # A terminal smoke test using the known Project 1 example.
    from getpass import getpass

    if not os.environ.get("NRP_API_KEY"):
        os.environ["NRP_API_KEY"] = getpass("NRP API key (hidden): ").strip()
    try:
        print(Diagnostics().diagnose("No", "Yes", "Abnormal", "Absent"))
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error))
