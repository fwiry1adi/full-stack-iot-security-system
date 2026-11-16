#!/usr/bin/env python3
"""
worker_db.py -- worker records and the PPE each role requires.

Requirements are stored per ROLE, not per worker. A turbine technician needs
a helmet because of the job, not because of who they are. Per-worker storage
would mean editing every record when a safety rule changes, and would let two
people in the same role drift apart in what they are checked against.

Worker IDs are integers because OpenCV's LBPH recognizer only accepts integer
labels -- it cannot store a name.
"""

import json
import os
from typing import Dict, List, Optional

DB_PATH = os.path.join("data", "workers.json")

DEFAULT_DB: Dict = {
    "roles": {
        "turbine_technician": ["helmet", "vest", "goggles", "gloves"],
        "maintenance":        ["helmet", "vest", "gloves", "boots"],
        "control_room":       ["vest"],
        "visitor":            ["helmet", "vest"],
    },
    "workers": {},
}


def _load() -> Dict:
    if not os.path.exists(DB_PATH):
        return json.loads(json.dumps(DEFAULT_DB))
    with open(DB_PATH) as f:
        return json.load(f)


def _save(db: Dict) -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def add_worker(name: str, role: str) -> int:
    """Register a worker and return their integer ID.

    IDs are never reused -- reusing one would silently attach an old face
    model to a new person.
    """
    db = _load()
    if role not in db["roles"]:
        raise ValueError(
            f"Unknown role {role!r}. Known roles: {sorted(db['roles'])}")
    existing = [int(k) for k in db["workers"]]
    worker_id = max(existing) + 1 if existing else 1
    db["workers"][str(worker_id)] = {"name": name, "role": role}
    _save(db)
    return worker_id


def get_worker(worker_id: int) -> Optional[Dict]:
    return _load()["workers"].get(str(worker_id))


def list_workers() -> Dict[str, Dict]:
    return _load()["workers"]


def required_ppe(worker_id: int) -> List[str]:
    """PPE this worker's role requires. Empty list if the worker is unknown.

    Returns [] rather than raising, so the caller decides what to do -- an
    unrecognised face is a reason to withhold a verdict, not to crash.
    """
    worker = get_worker(worker_id)
    if worker is None:
        return []
    return _load()["roles"].get(worker["role"], [])


def list_roles() -> Dict[str, List[str]]:
    return _load()["roles"]


def set_role(role: str, items: List[str]) -> None:
    db = _load()
    db["roles"][role] = items
    _save(db)


def main() -> None:
    db = _load()
    print("Roles:")
    for role, items in db["roles"].items():
        print(f"  {role:20s} {', '.join(items)}")
    print("\nWorkers:")
    if not db["workers"]:
        print("  (none enrolled yet -- run enroll_worker.py)")
    for wid, w in sorted(db["workers"].items(), key=lambda kv: int(kv[0])):
        print(f"  {wid:>3}  {w['name']:20s} {w['role']:20s} "
              f"needs: {', '.join(required_ppe(int(wid)))}")


if __name__ == "__main__":
    main()

