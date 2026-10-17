# CPSC 481 Project 2: LLM Diagnostics

`Diagnostics.diagnose()` sends the complete Asia Bayesian network and observed
evidence to NRP's `gpt-oss` model. It returns `[disease, probability]` and does
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
read -rsp 'NRP API key: ' NRP_API_KEY
echo
export NRP_API_KEY
python3 diagnostics_gui.py
```

The GUI requires Tkinter (Ubuntu package `python3-tk`). Network requests can take
a while. Errors are shown by the GUI; retry after correcting configuration or
waiting for the service to recover.

Optional environment variables: `NRP_MODEL` (default `gpt-oss`) and
`NRP_BASE_URL` (default `https://ellm.nrp-nautilus.io/v1`).
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
