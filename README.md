# CPSC 481 Project 2: LLM Diagnostics

`Diagnostics.diagnose()` sends the complete Asia Bayesian network and observed
evidence to a configured OpenAI-compatible LLM API (NRP by default). It returns `[disease, probability]` and does
not call AIMA or compute posteriors locally. The implementation uses only the
Python standard library. `NA` means unobserved.

## Run

```bash
cd /home/yyuqi1010/Desktop/Courses/Codes/CPSC481-project-2
python3 diagnostics.py
```

Paste the instructor's NRP key at the hidden prompt. This runs the known case
`No, Yes, Abnormal, Absent`; the Project 1 reference is Cancer, about 0.4367.
LLM arithmetic is not guaranteed: compare additional cases against Project 1.

For the original GUI, export the key in the same terminal first:

```bash
read -rsp 'LLM API key: ' LLM_API_KEY
echo
export LLM_API_KEY
python3 diagnostics_gui.py
```

The GUI requires Tkinter (Ubuntu package `python3-tk`). Network requests can take
a while. Errors are shown by the GUI; retry after correcting configuration or
waiting for the service to recover.

Optional environment variables: `LLM_MODEL` (default `gpt-oss`) and
`LLM_BASE_URL` (default `https://ellm.nrp-nautilus.io/v1`).
Never put the key into source control or the submitted file.

## Verify locally

```bash
python3 -m unittest -v
```

These offline tests mock the HTTP response to check integration and validation;
they do not verify the model's arithmetic or whether a key is active.

## Submit

Upload `diagnostics.py`, a real execution screenshot (PNG/JPG), and a short PDF
report together to Gradescope. The report explains model choice, main challenge,
and a surprising thing learned. Include every group member.

## Live accuracy verification

The separate development script imports the sibling `CPSC481-project-1` folder
as a reference. The submitted `diagnostics.py` never imports that reference.

```bash
# 12 representative inputs, including the eight Project 1 grading examples:
python3 verify_diagnostics.py

# All 81 valid inputs (makes 81 model requests, plus at most one retry per request):
python3 verify_diagnostics.py --all

# Reference generation only; no key or network needed:
python3 verify_diagnostics.py --all --reference-only
```

The script prompts for a hidden key if needed and writes `verification_results.json`.
It records disease agreement and absolute probability error. The default tolerance
is 0.001, a development choice, NOT a published Gradescope tolerance; change it
with `--tolerance`. A failed API request is recorded and stops the run. Exit status
is nonzero if any requested live comparison failed or could not be completed.
Reference-only output does not demonstrate LLM accuracy.

## Reliability and grading configuration

GUI requests run in a background thread; the window remains responsive and the
Diagnose button is disabled until the request finishes. Temporary HTTP statuses
429, 502, 503, and 504 are retried once after two seconds. Authentication failures,
timeouts, and malformed model output are not retried automatically.

The default credential variable is `LLM_API_KEY`. If your instructor specifies
another variable, set `LLM_API_KEY_ENV` to its NAME (not its secret value). Example:
`export LLM_API_KEY_ENV=INSTRUCTOR_SPECIFIED_VARIABLE`.
The constructor and imported `diagnose()` never prompt for a key, so automated
tests cannot hang waiting for input.

Confirm with the instructor: “Will Gradescope provide the API key through an
environment variable? If so, what is the variable name?” No grader configuration
has been confirmed. Do not submit a private key.

Output currently uses prompt-constrained JSON and strict local validation.
API-level JSON Schema has not been enabled because its support on the selected
NRP deployment has not been verified. Temperature zero does not guarantee exact
arithmetic or identical responses. Only live comparisons can assess accuracy.


## Select a service

Configuration uses the generic names `LLM_MODEL`, `LLM_BASE_URL`, and
`LLM_API_KEY`. Replace old `NRP_*` exports with these names. Use a key issued
by the selected service. Close the GUI and restart after changing configuration.

Choose ONE configuration:

```bash
# NRP (default)
export LLM_BASE_URL="https://ellm.nrp-nautilus.io/v1"
export LLM_MODEL="gpt-oss"
```

```bash
# Gemini (gemini-3.8-flash was successfully tested for one diagnosis)
export LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
export LLM_MODEL="gemini-3.8-flash"
```

```bash
# OpenAI (requires separate API credits)
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-5-nano"
```

Then start the GUI:

```bash
read -rsp 'API key for the selected service: ' LLM_API_KEY
echo
export LLM_API_KEY
python3 diagnostics_gui.py
```

Or run live comparison tests with `python3 verify_diagnostics.py`.
`gpt-5-nano` requests omit the incompatible temperature parameter.
