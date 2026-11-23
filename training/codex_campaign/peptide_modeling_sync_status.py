"""Read node23 training status and save a local, timestamped receipt."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="node23")
    parser.add_argument("--remote-root", default="/tmp/peptide_modeling_work_expanded5_v1")
    parser.add_argument("--campaign", action="store_true", help="Read the multi-seed journal")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/peptide_modeling_20261113_expanded5_v1",
    )
    args = parser.parse_args()
    source = "r=Path(" + repr(args.remote_root) + ")\n"
    source = "from pathlib import Path\nimport json,time,socket,os\n" + source
    source += """
receipt={'observed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'host':socket.gethostname()}
for name in ['full_launch.json','full_supervisor.json','dispatch/latest_full.json',
             'verification.json','canary_verification.json','campaign_status.json']:
    path=r/name
    if path.is_file():receipt[name]=json.loads(path.read_text())
"""
    if args.campaign:
        source += """
campaign=receipt.get('campaign_status.json',{})
identity=campaign.get('process_identity',{})
pid=campaign.get('pid')
proc=Path('/proc')/str(pid)
alive=False
if pid and (proc/'stat').exists():
    raw=(proc/'stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    alive=fields[0]!='Z' and int(fields[19])==identity.get('start_ticks')
receipt['controller_identity_verified_live']=alive
rounds=campaign.get('rounds',[])
if rounds:
    active=Path(rounds[-1]['run_dir'])/'dispatch/latest_full.json'
    if active.exists():receipt['dispatch/latest_full.json']=json.loads(active.read_text())
"""
    else:
        source += (
            "receipt['results']=[json.loads(p.read_text()) "
            "for p in sorted((r/'results/full').glob('*.json'))]\n"
        )
    source += "print(json.dumps(receipt))\n"
    result = subprocess.run(
        ["ssh.exe", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", args.host, "python3 -"],
        input=source,
        encoding="utf-8",
        capture_output=True,
        timeout=60,
        check=True,
    )
    receipt = json.loads(result.stdout)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    destination = args.run_dir / "remote_status.json"
    temporary = destination.with_suffix(".json.writing")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    dispatch = receipt.get("dispatch/latest_full.json", {})
    states = [task["status"] for task in dispatch.get("tasks", {}).values()]
    campaign = receipt.get("campaign_status.json", {})
    print(
        json.dumps(
            {
                "observed_at": receipt["observed_at"],
                "supervisor": receipt.get("full_supervisor.json", {}).get("status"),
                "tasks_started": len(states),
                "task_states": {state: states.count(state) for state in sorted(set(states))},
                "peak_cpu_budget": dispatch.get("peak_reserved_cpu_threads"),
                "local_receipt": str(destination),
                "campaign_status": campaign.get("status"),
                "controller_identity_verified_live": receipt.get(
                    "controller_identity_verified_live"
                ),
                "productive_hours": campaign.get("productive_wall_seconds", 0) / 3600,
                "completed_seeds": [
                    entry["seed"]
                    for entry in campaign.get("rounds", [])
                    if entry.get("status") == "completed"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
