"""Experimental PyTorch entry point in a clean evaluation container."""
import sys
from solution_torch import main

if __name__ == "__main__":
    # Task-aware defaults are chosen before dependency checks in solution_torch.
    # AutoGluon is available only to classification; regression stays isolated.
    sys.argv.append("--platform")
    # A cumulative AutoML search can exceed the legacy 1200-second step limit.
    if not any(arg == '--step-timeout' or arg.startswith('--step-timeout=') for arg in sys.argv):
        sys.argv.extend(['--step-timeout', '3600'])
    main()
