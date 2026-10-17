"""Development-only comparison: Project 1 exact inference vs real LLM calls."""
import argparse
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
from getpass import getpass

from diagnostics import Diagnostics

# The eight Project 1 grading examples, plus normal-Xray and partial evidence.
CASES = [
    ('NA', 'NA', 'NA', 'NA'), ('Yes', 'NA', 'NA', 'NA'),
    ('No', 'NA', 'NA', 'NA'), ('Yes', 'No', 'NA', 'NA'),
    ('Yes', 'No', 'Abnormal', 'NA'), ('Yes', 'No', 'Abnormal', 'Present'),
    ('No', 'Yes', 'Abnormal', 'Present'), ('No', 'Yes', 'Abnormal', 'Absent'),
    ('No', 'Yes', 'Normal', 'Present'), ('NA', 'No', 'Normal', 'Absent'),
    ('NA', 'NA', 'Abnormal', 'NA'), ('NA', 'Yes', 'NA', 'Present'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Make real calls for all 81 inputs')
    parser.add_argument('--reference-only', action='store_true', help='Run only Project 1; no API calls')
    parser.add_argument('--tolerance', type=float, default=0.001,
                        help='Development absolute error tolerance; not the Gradescope tolerance')
    parser.add_argument('--output', type=Path, default=Path('verification_results.json'))
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error('--tolerance must be finite and nonnegative')

    project1 = Path(__file__).resolve().parent.parent / 'CPSC481-project-1'
    sys.path.insert(0, str(project1))
    spec = importlib.util.spec_from_file_location('reference_diagnostics', project1 / 'diagnostics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference, model = module.Diagnostics(), Diagnostics()
    cases = list(itertools.product(('NA', 'Yes', 'No'), ('NA', 'Yes', 'No'),
                                  ('NA', 'Abnormal', 'Normal'), ('NA', 'Present', 'Absent'))) if args.all else CASES
    if not args.reference_only:
        key_variable = os.environ.get('LLM_API_KEY_ENV', 'LLM_API_KEY')
        if not os.environ.get(key_variable):
            os.environ[key_variable] = getpass('LLM API key (hidden): ').strip()
        if not os.environ[key_variable]:
            parser.error('An API key is required for live verification')

    rows = []
    for index, case in enumerate(cases, 1):
        expected = reference.diagnose(*case)
        row = {'input': list(case), 'reference': expected}
        if not args.reference_only:
            try:
                actual = model.diagnose(*case)
                error = abs(actual[1] - expected[1])
                row.update(actual=actual, absolute_error=error,
                           disease_matches=actual[0] == expected[0],
                           passed=actual[0] == expected[0] and error <= args.tolerance)
            except (RuntimeError, ValueError) as error:
                row.update(error=str(error), passed=False)
        rows.append(row)
        print(f'{index}/{len(cases)} {json.dumps(row)}', flush=True)
        # Stop after an API error instead of repeating a configuration failure.
        if 'error' in row:
            break
    report = {'model': model.model, 'base_url': model.base_url,
              'reference_only': args.reference_only, 'tolerance': args.tolerance,
              'requested_cases': len(cases), 'completed_cases': len(rows), 'results': rows}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.reference_only:
        print(f'Saved {len(rows)} reference results to {args.output}; no LLM verification performed.')
        return 0
    passed = sum(row['passed'] for row in rows)
    print(f'Passed {passed}/{len(cases)}; report: {args.output}')
    return 0 if passed == len(cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
