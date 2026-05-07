#!/usr/bin/env python3
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT / "data" / "runtime"
PID_FILE = RUNTIME_DIR / "nimbus.pid"
OUT_LOG = RUNTIME_DIR / "nimbus.out.log"
ERR_LOG = RUNTIME_DIR / "nimbus.err.log"


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def host() -> str:
    return os.environ.get("HOST") or "127.0.0.1"


def port() -> int:
    return env_int("PORT", 8000)


def qdrant_url() -> str:
    default_port = env_int("QDRANT_PORT", 6333)
    return os.environ.get("QDRANT_URL") or f"http://127.0.0.1:{default_port}"


def qdrant_port() -> int:
    parsed = urlparse(qdrant_url())
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout):
            return True
    except (OSError, URLError):
        return False


def wait_http_ok(url: str, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if http_ok(url):
            return True
        time.sleep(0.5)
    return http_ok(url)


def port_open(bind_host: str, bind_port: int) -> bool:
    probe_host = "127.0.0.1" if bind_host in {"0.0.0.0", "::"} else bind_host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe_host, bind_port)) == 0


def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_pid(pid: int, label: str) -> bool:
    if not process_exists(pid):
        return False
    print(f"Stopping {label} PID {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.time() + 10
    while time.time() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
    except OSError:
        pass
    return not process_exists(pid)


def start_qdrant() -> None:
    runtime = (os.environ.get("QDRANT_RUNTIME") or "").lower()
    if runtime != "docker":
        raise SystemExit(f"QDRANT_RUNTIME must be docker. Current value: {runtime!r}.")
    container = os.environ.get("QDRANT_CONTAINER_NAME")
    image = os.environ.get("QDRANT_DOCKER_IMAGE")
    volume = os.environ.get("QDRANT_DOCKER_VOLUME")
    if not container or not image or not volume:
        raise SystemExit("Docker Qdrant config is incomplete in .env.")
    if not docker_available():
        raise SystemExit("Docker is not available. Start Docker and try again.")

    existing = docker(["ps", "-a", "--filter", f"name=^/{container}$", "--format", "{{.Names}}"]).strip()
    if not existing:
        print(f"Creating Qdrant container {container!r}...")
        docker(["volume", "create", volume], check=True)
        docker([
            "run", "-d", "--name", container,
            "-p", f"{qdrant_port()}:6333",
            "-p", "6334:6334",
            "-v", f"{volume}:/qdrant/storage",
            image,
        ], check=True)
    else:
        running = docker([
            "ps", "--filter", f"name=^/{container}$",
            "--filter", "status=running",
            "--format", "{{.Names}}",
        ]).strip()
        if not running:
            print(f"Starting Qdrant container {container!r}...")
            docker(["start", container], check=True)

    if not wait_http_ok(f"{qdrant_url().rstrip('/')}/collections", 30):
        raise SystemExit(f"Qdrant did not become ready at {qdrant_url()}.")
    print(f"Qdrant is ready at {qdrant_url()}")


def stop_qdrant() -> None:
    container = os.environ.get("QDRANT_CONTAINER_NAME")
    if not container or not docker_available():
        return
    running = docker([
        "ps", "--filter", f"name=^/{container}$",
        "--filter", "status=running",
        "--format", "{{.Names}}",
    ]).strip()
    if running:
        print(f"Stopping Qdrant container {container!r}...")
        docker(["stop", container], check=True)


def docker_available() -> bool:
    try:
        subprocess.run(["docker", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def docker(args: list[str], check: bool = False) -> str:
    completed = subprocess.run(["docker", *args], capture_output=True, text=True, check=check)
    return completed.stdout


def start_nimbus(foreground: bool = False) -> None:
    if port_open(host(), port()):
        raise SystemExit(f"Nimbus port {port()} is already in use. Run stop first or choose another PORT.")
    if foreground:
        print(f"Starting Nimbus in foreground at http://{host()}:{port()}...")
        os.chdir(ROOT)
        subprocess.run([sys.executable, str(ROOT / "app.py")], check=True)
        return

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting Nimbus in background at http://{host()}:{port()}...")
    with OUT_LOG.open("ab") as stdout, ERR_LOG.open("ab") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "app.py")],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=(os.name != "nt"),
        )
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    health = f"http://127.0.0.1:{port()}/api/health"
    if wait_http_ok(health, 30):
        print(f"Nimbus is ready at http://127.0.0.1:{port()}")
    else:
        print(f"Nimbus PID {process.pid} started, but health did not respond. Check {ERR_LOG}.")


def stop_nimbus() -> None:
    stopped = False
    pid = read_pid()
    if pid:
        stopped = stop_pid(pid, "Nimbus") or stopped
    try:
        PID_FILE.unlink()
    except OSError:
        pass
    if not stopped:
        print("Nimbus is not running from the recorded PID.")


def status() -> None:
    pid = read_pid()
    print("Nimbus")
    print(f"  URL:    http://127.0.0.1:{port()}")
    print(f"  Health: {'ready' if http_ok(f'http://127.0.0.1:{port()}/api/health') else 'not responding'}")
    print(f"  PID:    {pid if pid else 'unknown'}")
    print(f"  Port:   {'listening' if port_open(host(), port()) else 'free'}")
    print(f"  Logs:   {OUT_LOG}")
    print("")
    print("Qdrant")
    qdrant_collections_url = f"{qdrant_url().rstrip('/')}/collections"
    print(f"  URL:    {qdrant_url()}")
    print(f"  Health: {'ready' if http_ok(qdrant_collections_url) else 'not responding'}")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Cross-platform Nimbus controller")
    parser.add_argument("action", choices=["start", "stop", "restart", "status", "start-qdrant", "stop-qdrant"])
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--keep-qdrant", action="store_true")
    args = parser.parse_args()

    if args.action == "start":
        start_qdrant()
        start_nimbus(foreground=args.foreground)
    elif args.action == "stop":
        stop_nimbus()
        if not args.keep_qdrant:
            stop_qdrant()
    elif args.action == "restart":
        stop_nimbus()
        if not args.keep_qdrant:
            stop_qdrant()
        time.sleep(1)
        start_qdrant()
        start_nimbus(foreground=args.foreground)
    elif args.action == "status":
        status()
    elif args.action == "start-qdrant":
        start_qdrant()
    elif args.action == "stop-qdrant":
        stop_qdrant()


if __name__ == "__main__":
    main()
