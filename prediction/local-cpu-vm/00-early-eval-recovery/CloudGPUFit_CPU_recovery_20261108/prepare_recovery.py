#!/usr/bin/env python3
"""Apply the reviewed patch in a separate local worktree; never fetch or push."""
import argparse
from pathlib import Path
import subprocess

BASE_COMMIT = '7f822d76af13bea39191beb1852ec67513b18f15'


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, text=True,
                          capture_output=True, encoding='utf-8', errors='replace').stdout.strip()


def prepare(repo, target, base_commit=BASE_COMMIT):
    repo, target = Path(repo).resolve(), Path(target).expanduser().resolve()
    root = Path(git(repo, 'rev-parse', '--show-toplevel'))
    if target.exists():
        raise ValueError('Target already exists. Choose a new directory; nothing was overwritten.')
    git(root, 'cat-file', '-e', base_commit + '^{commit}')
    patch = Path(__file__).resolve().with_name('fix.patch')
    git(root, 'worktree', 'add', '--detach', str(target), base_commit)
    try:
        git(target, 'apply', '--check', str(patch))
        git(target, 'apply', str(patch))
    except Exception:
        print('The new worktree was left for inspection:', target)
        raise
    print('Prepared separate working directory:', target)
    print('Original checkout, container and prediction results were left unchanged.')
    print('No remote Git operation was performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.', help='Existing CloudGPUFit checkout')
    parser.add_argument('--target', default='~/CloudGPUFit-cpu-recovery-20261108')
    args = parser.parse_args()
    try:
        prepare(args.repo, args.target)
    except subprocess.CalledProcessError as exc:
        parser.exit(1, 'Git failed: ' + (exc.stderr or str(exc)) + '\n')
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + '\n')
