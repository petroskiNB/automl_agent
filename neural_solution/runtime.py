"""Assemble a versioned runtime from neural overrides and shared root sources.

Keep the legacy modules unchanged: their hashes identify existing experiments.
Workers run from one complete directory, including in subprocesses and ZIPs.
"""
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SHARED = (
    'solution.py', 'xgb_backend.py', 'diagnostics.py', 'temporal_research.py',
    'classification_checks.py', 'final_output.py', 'research_team.py',
    'torch_resources.py', 'torch_model.py', 'requirements.txt',
)
OVERRIDES = (
    'research.py', 'classification_research.py', 'parallel_research.py',
    'solution_torch.py', 'torch_backend.py', 'torch_suites.py',
    'benchmark_torch.py', 'test_torch_backend.py', 'build_torch_submission.py',
    'compare_combined.py',
)


def prepare(root=ROOT):
    root = Path(root).resolve()
    sources = {name: (root / name).read_bytes() for name in SHARED}
    sources.update({name: (root / 'neural_solution' / name).read_bytes() for name in OVERRIDES})
    digest = hashlib.sha256()
    for name, data in sorted(sources.items()):
        digest.update(name.encode() + b'\0' + data)
    destination = root / 'work' / 'neural_runtime' / digest.hexdigest()[:20]
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in sources.items():
        path = destination / name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f'Runtime source was modified: {path}')
        else:
            # Atomic publication also supports simultaneous launcher invocations.
            import tempfile
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
                stream.write(data)
                temporary = Path(stream.name)
            temporary.replace(path)
    return destination


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {'agent', 'test', 'benchmark', 'build', 'compare'}:
        raise SystemExit('Usage: run_neural.py {agent|test|benchmark|build|compare} [arguments]')
    mode, arguments = sys.argv[1], sys.argv[2:]
    def supplied(option):
        return any(value == option or value.startswith(option + '=') for value in arguments)
    runtime = prepare()
    scripts = dict(agent='solution_torch.py', benchmark='benchmark_torch.py', build='build_torch_submission.py', compare='compare_combined.py')
    if mode == 'test':
        command = [sys.executable, '-m', 'unittest', 'test_torch_backend', *arguments]
        cwd = runtime
    else:
        command = [sys.executable, str(runtime / scripts[mode]), *arguments]
        cwd = Path.cwd()
        if mode == 'agent' and not supplied('--output-dir') and not supplied('--platform') and arguments and not arguments[0].startswith('-'):
            task = Path(arguments[0]).resolve()
            key = hashlib.sha256(str(task).encode()).hexdigest()[:10]
            command.extend(['--output-dir', str(ROOT / 'outputs' / 'neural' / (task.parent.name + '_' + key))])
        if mode == 'build' and not supplied('--output'):
            command.extend(['--output', str(ROOT / 'outputs' / 'automl_combined.zip')])
    print(f'Combined runtime: {runtime}', flush=True)
    raise SystemExit(subprocess.call(command, cwd=cwd))


if __name__ == '__main__':
    main()
