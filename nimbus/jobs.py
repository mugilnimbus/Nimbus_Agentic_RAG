import threading
import time
from collections.abc import Callable


class JobQueue:
    def __init__(self) -> None:
        self._threads: set[threading.Thread] = set()
        self._lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._state: dict[int, dict] = {}
        self._next_id = 0

    def queue(self, job_type: str, detail: str, func: Callable, arg) -> int:
        job_id = self.create(job_type, detail)
        thread = threading.Thread(
            target=self._run,
            args=(job_id, func, arg),
            name=f"nimbus-job-{job_id}",
            daemon=True,
        )
        with self._lock:
            self._threads.add(thread)
        thread.start()
        return job_id

    def create(self, job_type: str, detail: str = "") -> int:
        now = time.time()
        with self._lock:
            self._next_id += 1
            self._state[self._next_id] = {
                "id": self._next_id,
                "type": job_type,
                "status": "queued",
                "detail": detail,
                "error": "",
                "result": {},
                "progress_current": 0,
                "progress_total": 0,
                "created_at": now,
                "updated_at": now,
            }
            return self._next_id

    def update(
        self,
        job_id: int,
        status: str,
        detail: str | None = None,
        error: str | None = None,
        result: dict | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> None:
        with self._lock:
            job = self._state.get(job_id)
            if not job:
                return
            job["status"] = status
            job["updated_at"] = time.time()
            if detail is not None:
                job["detail"] = detail
            if error is not None:
                job["error"] = error
            if result is not None:
                job["result"] = result
            if progress_current is not None:
                job["progress_current"] = max(0, int(progress_current))
            if progress_total is not None:
                job["progress_total"] = max(0, int(progress_total))

    def jobs(self, limit: int = 20) -> list[dict]:
        with self._lock:
            jobs = sorted(self._state.values(), key=lambda item: item["id"], reverse=True)
            return [dict(job) for job in jobs[: max(1, min(limit, 100))]]

    def _run(self, job_id: int, func: Callable, arg) -> None:
        try:
            with self._execution_lock:
                self.update(job_id, "running")
                result = func(arg, job_id)
                self.update(
                    job_id,
                    "complete",
                    result=result,
                    progress_current=1,
                    progress_total=1,
                )
        except Exception as exc:
            self.update(job_id, "failed", error=str(exc))
        finally:
            current = threading.current_thread()
            with self._lock:
                self._threads.discard(current)

