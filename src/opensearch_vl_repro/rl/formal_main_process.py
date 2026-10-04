"""Exclusive Linux subreaper lifetime for a concurrent collection wave."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import formal_smoke as supervision


class ParallelProcessRunner:
    def __init__(self):
        self.active = False

    def __call__(self, jobs, *, cwd):
        """Jobs contain explicit argv/env/log. No GPU/API knowledge in supervisor."""
        if self.active or os.name != "posix" or not 1 <= len(jobs) <= 4:
            raise RuntimeError("exclusive POSIX collection wave (1..4 workers) required")
        if supervision.descendant_identities(supervision.linux_process_table(), os.getpid()):
            raise RuntimeError("coordinator already owns children; GPU phase forbidden")
        self.active = True
        processes, readers, errors, owned = [], [], [], {}
        prior = None
        codes = [None] * len(jobs)
        try:
            prior = supervision.set_subreaper(True)
            for job in jobs:
                process = subprocess.Popen(job["command"], env=job["env"], cwd=cwd,
                    start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                processes.append(process)
                def read_log(p=process, job=job):
                    try:
                        with Path(job["log"]).open("w", encoding="utf-8") as stream:
                            for line in p.stdout:
                                # Credentials only inherited, never recorded.
                                line = supervision.redact_runtime_secrets(line)
                                stream.write(line)
                                stream.flush()
                                print(line, end="", flush=True)
                    except BaseException as exc:
                        errors.append(exc)
                        if p.poll() is None:
                            p.kill()
                reader = threading.Thread(target=read_log, daemon=True)
                reader.start()
                readers.append(reader)
                owned.update(supervision.descendant_identities(supervision.linux_process_table(), os.getpid()))
            while True:
                owned.update(supervision.descendant_identities(supervision.linux_process_table(), os.getpid()))
                codes = [p.poll() for p in processes]
                if errors or any(c is not None and c != 0 for c in codes):
                    break  # FIRST observed failure, do not wait for healthy peers
                if all(c is not None for c in codes):
                    break
                time.sleep(.05)
        finally:
            cleaned = False
            try:
                for p in processes:
                    if p.poll() is None:
                        p.kill()
                    p.wait()
                deadline = time.monotonic() + 30
                while True:
                    rows = supervision.linux_process_table()
                    owned.update(supervision.descendant_identities(rows, os.getpid()))
                    for pid in supervision.live_owned_processes(rows, owned):
                        if pid in supervision.live_owned_processes(supervision.linux_process_table(), {pid: owned[pid]}):
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    while True:
                        try:
                            pid, _ = os.waitpid(-1, os.WNOHANG)
                            if pid == 0:
                                break
                        except ChildProcessError:
                            break
                    remaining = supervision.descendant_identities(supervision.linux_process_table(), os.getpid())
                    if not remaining and not supervision.live_owned_processes(supervision.linux_process_table(), owned):
                        if not any(supervision.process_group_alive(p.pid) for p in processes):
                            break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("collection descendants remain; next GPU phase forbidden")
                    time.sleep(.05)
                for reader in readers:
                    reader.join(timeout=30)
                    if reader.is_alive():
                        raise RuntimeError("collection descendant holds log pipe; next GPU phase forbidden")
                for p in processes:
                    p.stdout.close()
                cleaned = True
            finally:
                if prior is not None:
                    supervision.set_subreaper(prior)
                self.active = not cleaned
        if errors:
            raise errors[0]
        return codes  # killed healthy peers remain None, explicitly not success
