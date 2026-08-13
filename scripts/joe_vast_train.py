#!/usr/bin/env python3
"""Local CLI for interruptible Joe training on vast.ai.

Subcommands: launch, status, resume, destroy, sync-env.

Packs the checkout, uploads it to R2, and creates an interruptible
instance whose onstart script converges on ``joe/<run_name>/`` (vast plan
§3). The training loop is unchanged; this script does not import JAX.

    python scripts/joe_vast_train.py sync-env
    python scripts/joe_vast_train.py launch --smoke
    python scripts/joe_vast_train.py status --run-name joe-S-vast-smoke-...
    python scripts/joe_vast_train.py resume --run-name ...
    python scripts/joe_vast_train.py destroy --run-name ... --purge-r2

``launch`` and ``resume`` also accept ``--instance-id <id>`` to adopt an
existing, already-running vast.ai instance instead of searching offers
and renting one. The launcher labels the instance, records it in R2, and
runs ``scripts/joe_vast_onstart.sh`` on it over SSH (``vastai ssh-url``)
with ``RUN_NAME`` and ``JOE_ROOT`` exported. ``--bid`` does not combine
with ``--instance-id``: there is no offer.

Never pipe the output through tail/head — redirect to a file (AGENTS.md).
R2 tokens never go on the vastai command line; they live in account env
vars (``sync-env``) and the gitignored ``.env``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ONSTART = REPO / "scripts" / "joe_vast_onstart.sh"
sys.path.insert(0, str(REPO))

from training.joe.launch import (  # noqa: E402
    DEFAULT_IMAGE,
    FULL_DISK_GB,
    SECRETS_HINT,
    SMOKE_DISK_GB,
    apply_smoke,
    code_object_name,
    env_var_names,
    find_vastai_bin,
    instance_label,
    offer_query,
    offer_unavailable,
    resolve_gpu_names,
    validate_run_name,
    vastai_cli_error,
)
from training.joe.pack import engine_sha, pack_checkout  # noqa: E402
from training.joe.store import (  # noqa: E402
    LAUNCH_SCHEMA,
    R2Store,
    load_dotenv,
)

REQUIRED_R2 = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
OPTIONAL_R2 = ("R2_BUCKET",)
_VASTAI_BIN = None


def vastai_bin():
    """The pip console script: PATH, then ``<venv>/bin/vastai``."""
    global _VASTAI_BIN
    if _VASTAI_BIN is None:
        found = find_vastai_bin(
            path_which=shutil.which("vastai"),
            executable=sys.executable,
            repo=REPO,
        )
        if found is None:
            raise SystemExit(
                "vastai CLI not found. It is a pip package in this repo: "
                "`pip install vastai` (see requirements-dev.txt). The "
                "console script lives at .venv/bin/vastai. Then "
                "`vastai set api-key <key>`.")
        _VASTAI_BIN = found
    return _VASTAI_BIN


def vastai(*args, raw=True, redact=None):
    """Run the vastai CLI. Print the command with secrets stripped."""
    redact = set(redact or ())
    cmd = [vastai_bin(), *[str(a) for a in args]]
    shown = ["<redacted>" if a in redact else a for a in cmd]
    if raw and "--raw" not in cmd:
        cmd.append("--raw")
        shown.append("--raw")
    print("+", " ".join(shown), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    fail = vastai_cli_error(proc.stdout, proc.stderr, proc.returncode)
    if fail:
        raise SystemExit(fail)
    out = proc.stdout.strip()
    if not out:
        return None
    if not raw:
        print(out, flush=True)
        return out
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") or line.startswith("["):
                return json.loads(line)
        # create/update env-var print a success sentence and return None,
        # even with --raw. Do not abort the rest of the loop.
        return out


def _as_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("offers", "instances", "results", "data"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]
    return [obj]


def require_vastai():
    vastai_bin()


def ensure_vast_r2_env():
    names = env_var_names(vastai("show", "env-vars"))
    missing = [k for k in REQUIRED_R2 if k not in names]
    if missing:
        raise SystemExit(
            "vast.ai account env vars missing: " + ", ".join(missing) +
            "\nRun: python scripts/joe_vast_train.py sync-env\n" +
            SECRETS_HINT)


def cmd_sync_env(_args):
    require_vastai()
    load_dotenv()
    existing = env_var_names(vastai("show", "env-vars"))
    for key in REQUIRED_R2 + OPTIONAL_R2:
        val = os.environ.get(key)
        if not val:
            if key in OPTIONAL_R2:
                continue
            raise SystemExit(
                f"{key} is not set locally (gitignored .env or environment)")
        if key in existing:
            print(f"updating vast env var {key}", flush=True)
            vastai("update", "env-var", key, val, raw=False, redact={val})
        else:
            print(f"creating vast env var {key}", flush=True)
            vastai("create", "env-var", key, val, raw=False, redact={val})
            existing.add(key)
    names = env_var_names(vastai("show", "env-vars"))
    missing = [k for k in REQUIRED_R2 if k not in names]
    if missing:
        raise SystemExit(
            "sync-env finished but vast.ai still has no: " +
            ", ".join(missing) + "\n" + SECRETS_HINT)
    print("account env vars synced (values not printed)", flush=True)


def _load_cfg_dict(tier, smoke, run_name, overrides):
    import yaml

    path = REPO / "training" / "joe" / "configs" / f"{tier}.yaml"
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    if smoke:
        cfg_dict = apply_smoke(cfg_dict)
    if run_name:
        cfg_dict["run_name"] = run_name
    if overrides:
        cfg_dict.update(json.loads(overrides))
    validate_run_name(cfg_dict["run_name"])
    return cfg_dict


def _upload_code_and_launch(store, cfg_dict, *, git_meta, engine, gpu_names,
                            num_gpus, image, disk_gb, smoke, bid, tier):
    import yaml

    run_name = cfg_dict["run_name"]
    with tempfile.TemporaryDirectory(prefix="joe-vast-") as td:
        tar_path = Path(td) / "code.tar.gz"
        meta = pack_checkout(REPO, tar_path, git_sha=git_meta["git_sha"])
        meta["git_dirty"] = git_meta["git_dirty"]
        code_name = code_object_name(
            meta["git_sha"], meta["git_dirty"], meta["sha256"])
        code_key = store.key(run_name, "code", code_name)
        print(f"uploading {code_key} ({meta['size']} bytes, "
              f"{meta['n_files']} files)", flush=True)
        ref = store.upload_file(code_key, str(tar_path))

        cfg_path = Path(td) / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg_dict, f, default_flow_style=False)
        store.upload_run_file(run_name, "config.yaml", str(cfg_path))

    launch = {
        "schema": LAUNCH_SCHEMA,
        "run_name": run_name,
        "git_sha": meta["git_sha"],
        "git_dirty": meta["git_dirty"],
        "engine_sha": engine,
        "code_key": code_key,
        "code_sha256": ref["sha256"],
        "code_size": ref["size"],
        "tier": tier,
        "smoke": bool(smoke),
        "image": image,
        "disk_gb": int(disk_gb),
        "num_gpus": int(num_gpus),
        "gpu_names": list(gpu_names),
        "bid_price": bid,
        "created": time.time(),
    }
    store.put_launch(run_name, launch)
    print(f"wrote launch.json for {run_name}", flush=True)
    return launch


CREATE_OFFER_ATTEMPTS = 20
CREATE_SEARCH_ROUNDS = 3


def _ranked_offers(offers, smoke):
    offers = _as_list(offers)
    if not offers:
        return []
    if smoke:
        return sorted(
            offers,
            key=lambda o: float(o.get("min_bid") or o.get("dph_total") or 1e9))
    return list(offers)


def _create_instance(launch, bid):
    run_name = launch["run_name"]
    query = offer_query(launch["gpu_names"], launch["num_gpus"])
    order = "min_bid" if launch.get("smoke") else "dlperf_usd-"
    print(f"searching interruptible offers: {query}", flush=True)
    offers = _ranked_offers(
        vastai("search", "offers", "--type", "bid",
               "-o", order, "--storage", str(launch["disk_gb"]), query),
        launch.get("smoke"))
    if not offers:
        raise SystemExit(f"no interruptible offers matched: {query}")
    last_err = None
    tried = 0
    for round_i in range(CREATE_SEARCH_ROUNDS):
        if round_i:
            print(f"search round {round_i + 1}: asking again", flush=True)
            time.sleep(2)
            offers = _ranked_offers(
                vastai("search", "offers", "--type", "bid",
                       "-o", order, "--storage", str(launch["disk_gb"]),
                       query),
                launch.get("smoke"))
            if not offers:
                last_err = f"no interruptible offers matched: {query}"
                continue
        for offer in offers[:CREATE_OFFER_ATTEMPTS]:
            offer_id = offer.get("id") or offer.get("ask_id")
            gpu = offer.get("gpu_name") or offer.get("gpu_name_id")
            min_bid = offer.get("min_bid")
            print(f"offer {offer_id} gpu={gpu} min_bid={min_bid} "
                  f"dph={offer.get('dph_total')}", flush=True)
            if bid is not None:
                this_bid = float(bid)
            elif min_bid is not None:
                this_bid = float(min_bid)
            else:
                last_err = "offer has no min_bid; pass --bid"
                continue
            env = f"-e RUN_NAME={run_name} -e JOE_ROOT=/workspace/joe"
            tried += 1
            try:
                result = vastai(
                    "create", "instance", str(offer_id),
                    "--image", launch["image"],
                    "--disk", str(launch["disk_gb"]),
                    "--ssh", "--direct",
                    "--onstart", str(ONSTART),
                    "--label", instance_label(run_name),
                    "--env", env,
                    "--bid_price", str(this_bid),
                    "--cancel-unavail",
                )
            except SystemExit as e:
                if offer_unavailable(e):
                    print(f"offer {offer_id} is gone; trying the next ask",
                          flush=True)
                    last_err = e
                    continue
                raise
            if not isinstance(result, dict) or not result.get("success"):
                last_err = f"create instance failed: {result}"
                if offer_unavailable(last_err):
                    print(f"offer {offer_id} is gone; trying the next ask",
                          flush=True)
                    continue
                raise SystemExit(last_err)
            instance_id = result.get("new_contract")
            if instance_id is None:
                raise SystemExit(
                    f"create instance returned no new_contract: {result}")
            return str(instance_id), str(offer_id), float(this_bid)
    raise SystemExit(
        f"no interruptible offer accepted after {tried} tries: {last_err}")


def _record_instance(store, run_name, instance_id, offer_id, bid):
    store.put_instance(run_name, instance_id,
                       extra={"offer_id": offer_id, "bid_price": bid})
    # Write the new id before the replacement boots so the lease check
    # sees the same instance id (vast plan §4).
    store.put_heartbeat(run_name, instance_id)
    print(f"instance {instance_id} labeled {instance_label(run_name)}",
          flush=True)
    return instance_id


ADOPT_ONSTART_REMOTE = "/workspace/joe-adopt-onstart.sh"
ADOPT_ONSTART_LOG = "/workspace/joe-adopt-onstart.log"

# Runs on the instance. Account env vars are injected into PID 1 at create
# time but a plain SSH session may not see them (the onstart script copies
# them to /etc/environment only once it runs), so pull R2_* from
# /proc/1/environ without echoing any value, then start onstart detached.
_ADOPT_BOOT_PY = """\
import os, subprocess
try:
    raw = open("/proc/1/environ", "rb").read().split(b"\\0")
except OSError:
    raw = []
for kv in raw:
    if b"=" not in kv:
        continue
    key, val = kv.split(b"=", 1)
    key = key.decode()
    if key.startswith("R2_") and key not in os.environ:
        os.environ[key] = val.decode()
log = open("{log}", "ab")
proc = subprocess.Popen(["/bin/sh", "{script}"], stdout=log, stderr=log,
                        start_new_session=True)
print("adopt onstart pid", proc.pid, flush=True)
"""


def _verify_instance(instance_id):
    """The instance must exist and be running before we touch it."""
    instance_id = str(instance_id)
    data = vastai("show", "instance", instance_id)
    inst = None
    for cand in _as_list(data):
        if not isinstance(cand, dict):
            continue
        iid = str(cand.get("id") or cand.get("instance_id") or "")
        if iid == instance_id or inst is None:
            inst = cand
        if iid == instance_id:
            break
    if not inst:
        raise SystemExit(
            f"vast.ai instance {instance_id} not found "
            f"(vastai show instance returned nothing); check the id")
    status = inst.get("actual_status") or inst.get("status")
    if status != "running":
        raise SystemExit(
            f"vast.ai instance {instance_id} is not running "
            f"(actual_status={status!r}); start it or pick another instance")
    return inst


def _ssh_target(instance_id):
    import urllib.parse

    out = vastai("ssh-url", str(instance_id), raw=False)
    text = str(out or "").strip().splitlines()[-1] if out else ""
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme != "ssh" or not parsed.hostname or not parsed.port:
        raise SystemExit(f"could not parse vastai ssh-url output: {text!r}")
    return (parsed.username or "root", parsed.hostname, parsed.port)


def _ssh_run(target, command, stdin_text=None, timeout=180):
    user, host, port = target
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
           "-p", str(port), f"{user}@{host}", command]
    print("+", " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(cmd, input=stdin_text, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"ssh to the instance timed out after {timeout}s")
    if proc.returncode:
        raise SystemExit(
            f"ssh to the instance failed ({proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}\n"
            "Adoption needs SSH access; check that your vast.ai SSH key "
            "is attached to the instance.")
    out = proc.stdout.strip()
    if out:
        print(out, flush=True)
    return out


def _adopt_instance(store, run_name, instance_id):
    """Adopt a pre-existing instance: verify, label, record, run onstart.

    launch.json must already be in R2 — onstart reads it to fetch the
    code. The instance record and heartbeat go in before boot (vast plan
    §4), with no offer_id or bid_price because there is no offer.
    """
    instance_id = str(instance_id)
    _verify_instance(instance_id)
    vastai("label", "instance", instance_id, instance_label(run_name))
    _record_instance(store, run_name, instance_id, None, None)
    target = _ssh_target(instance_id)
    _ssh_run(target,
             f"mkdir -p /workspace && cat > {ADOPT_ONSTART_REMOTE}",
             stdin_text=ONSTART.read_text())
    boot_py = _ADOPT_BOOT_PY.format(
        log=ADOPT_ONSTART_LOG, script=ADOPT_ONSTART_REMOTE)
    _ssh_run(target,
             f"RUN_NAME={run_name} JOE_ROOT=/workspace/joe python3 -",
             stdin_text=boot_py)
    print(f"onstart started on instance {instance_id} "
          f"(remote log: {ADOPT_ONSTART_LOG})", flush=True)


def _instance_ids_for_run(run_name, store):
    ids = set()
    rec = store.read_instance(run_name)
    if rec and rec.get("instance_id"):
        ids.add(str(rec["instance_id"]))
    try:
        instances = _as_list(vastai("show", "instances"))
    except SystemExit as e:
        print(f"warning: could not list instances: {e}", flush=True)
        instances = []
    label = instance_label(run_name)
    matched = []
    for inst in instances:
        iid = str(inst.get("id") or inst.get("instance_id") or "")
        if not iid:
            continue
        if inst.get("label") == label or iid in ids:
            ids.add(iid)
            matched.append(inst)
    return ids, matched


def destroy_run_instances(run_name, store, keep=None):
    require_vastai()
    ids, _matched = _instance_ids_for_run(run_name, store)
    if keep is not None:
        ids.discard(str(keep))
    if not ids:
        print(f"no vast.ai instance found for {run_name}", flush=True)
        return []
    destroyed = []
    for iid in sorted(ids):
        print(f"destroying instance {iid}", flush=True)
        try:
            vastai("destroy", "instance", iid, "-y")
        except SystemExit:
            vastai("destroy", "instance", iid)
        destroyed.append(iid)
    return destroyed


def cmd_launch(args):
    load_dotenv()
    instance_id = str(args.instance_id or "").strip()
    if instance_id and args.bid is not None:
        raise SystemExit(
            "--bid has no meaning with --instance-id: there is no offer "
            "to bid on. Drop --bid.")
    if instance_id and args.dry_run:
        raise SystemExit(
            "--dry-run searches offers; it does not combine with "
            "--instance-id.")
    smoke = bool(args.smoke)
    tier = args.tier or ("S" if smoke else "M")
    gpu = args.gpu or ("4090" if smoke else "h100")
    gpu_names = resolve_gpu_names(gpu)
    disk = args.disk or (SMOKE_DISK_GB if smoke else FULL_DISK_GB)
    image = args.image or DEFAULT_IMAGE
    stamp = time.strftime("%Y%m%d-%H%M")
    run_name = args.run_name or (
        f"joe-{tier}-vast-smoke-{stamp}" if smoke
        else f"joe-{tier}-vast-{stamp}")
    validate_run_name(run_name)
    cfg_dict = _load_cfg_dict(tier, smoke, run_name, args.overrides)

    store = R2Store.from_env()
    existing = store.read_launch(run_name)
    if existing is not None and not args.force:
        raise SystemExit(
            f"launch.json already exists for {run_name}; "
            f"use resume, or destroy --purge-r2, or pass --force")

    if instance_id and not args.pack_only:
        # Fail before the upload work if the instance is gone or stopped.
        require_vastai()
        _verify_instance(instance_id)

    from training.joe.pack import repo_git_state
    git_sha, dirty = repo_git_state(REPO)
    engine = engine_sha(REPO)
    print(f"packing {run_name} (tier {tier}, engine {engine[:12]}, "
          f"git {git_sha[:12]}{' dirty' if dirty else ''})", flush=True)
    launch = _upload_code_and_launch(
        store, cfg_dict,
        git_meta={"git_sha": git_sha, "git_dirty": dirty},
        engine=engine, gpu_names=gpu_names, num_gpus=args.num_gpus,
        image=image, disk_gb=disk, smoke=smoke, bid=args.bid, tier=tier)

    if args.pack_only:
        print(f"pack-only: code and launch.json are in R2 for {run_name}",
              flush=True)
        return

    require_vastai()
    ensure_vast_r2_env()
    if instance_id:
        _adopt_instance(store, run_name, instance_id)
        print(f"launched {run_name} on adopted instance {instance_id}",
              flush=True)
        return
    if args.dry_run:
        query = offer_query(gpu_names, args.num_gpus)
        order = "min_bid" if smoke else "dlperf_usd-"
        offers = vastai(
            "search", "offers", "--type", "bid",
            "-o", order, "--storage", str(disk), query)
        ranked = _ranked_offers(offers, smoke)
        print(f"dry-run: {len(ranked)} offers; first={ranked[0] if ranked else None}",
              flush=True)
        return

    instance_id, offer_id, bid = _create_instance(launch, args.bid)
    _record_instance(store, run_name, instance_id, offer_id, bid)
    launch["bid_price"] = bid
    store.put_launch(run_name, launch)
    print(f"launched {run_name} on instance {instance_id}", flush=True)


def cmd_resume(args):
    load_dotenv()
    run_name = validate_run_name(args.run_name)
    adopt_id = str(args.instance_id or "").strip()
    if adopt_id and args.bid is not None:
        raise SystemExit(
            "--bid has no meaning with --instance-id: there is no offer "
            "to bid on. Drop --bid.")
    store = R2Store.from_env()
    launch = store.read_launch(run_name)
    if launch is None:
        raise SystemExit(f"no launch.json for {run_name}; run launch first")
    require_vastai()
    ensure_vast_r2_env()
    if adopt_id:
        _verify_instance(adopt_id)
        print(f"resume {run_name}: destroy-before-adopt", flush=True)
        destroy_run_instances(run_name, store, keep=adopt_id)
        _adopt_instance(store, run_name, adopt_id)
        print(f"resumed {run_name} on adopted instance {adopt_id}",
              flush=True)
        return
    print(f"resume {run_name}: destroy-before-create", flush=True)
    destroy_run_instances(run_name, store)
    instance_id, offer_id, bid = _create_instance(launch, args.bid)
    _record_instance(store, run_name, instance_id, offer_id, bid)
    print(f"resumed {run_name} on instance {instance_id}", flush=True)


def cmd_destroy(args):
    load_dotenv()
    run_name = validate_run_name(args.run_name)
    store = R2Store.from_env()
    if not args.purge_r2_only:
        require_vastai()
        destroy_run_instances(run_name, store)
    if args.purge_r2 or args.purge_r2_only:
        n = store.delete_prefix(run_name)
        print(f"purged {n} R2 objects under joe/{run_name}/", flush=True)
    else:
        print("R2 prefix kept (pass --purge-r2 to delete it)", flush=True)


def _last_metrics(store, run_name):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "metrics.jsonl")
        try:
            store.download_run_file(run_name, "logs/metrics.jsonl", path)
        except FileNotFoundError:
            return None
        text = Path(path).read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def cmd_status(args):
    load_dotenv()
    run_name = validate_run_name(args.run_name)
    store = R2Store.from_env()
    launch = store.read_launch(run_name)
    latest = store.resolve_latest(run_name)
    hb = store.read_heartbeat(run_name)
    boot = store.read_boot(run_name)
    rec = store.read_instance(run_name)
    metrics = _last_metrics(store, run_name)

    print(f"run {run_name}", flush=True)
    if launch:
        print(f"  git {launch.get('git_sha', '')[:12]} "
              f"engine {str(launch.get('engine_sha', ''))[:12]} "
              f"smoke={launch.get('smoke')} "
              f"gpu={launch.get('gpu_names')}", flush=True)
    else:
        print("  no launch.json", flush=True)
    if latest:
        st = latest["state"]
        print(f"  latest.json global_step={st['global_step']} "
              f"stage={st['curriculum_stage']} "
              f"last_eval_wr={st['last_eval_wr']}", flush=True)
    else:
        print("  no latest.json (written at the first checkpoint, "
              "not at boot)", flush=True)
    if boot:
        age = time.time() - float(boot["time"])
        print(f"  boot.json phase={boot.get('phase')} age={age:.0f}s "
              f"python={boot.get('python_version', '')}", flush=True)
    else:
        print("  no boot.json (onstart has not reached R2; "
              "first-boot jax pip can take 10-20 min on the old script)",
              flush=True)
    if hb:
        age = time.time() - float(hb["time"])
        print(f"  heartbeat instance={hb['instance_id']} age={age:.0f}s",
              flush=True)
        if age > 300 and latest is None and boot is None:
            print("  note: launch writes this heartbeat; vast_boot "
                  "refreshes it only after pip + unpack", flush=True)
    else:
        print("  no heartbeat", flush=True)
    if rec:
        print(f"  recorded instance {rec.get('instance_id')}", flush=True)
    if metrics:
        step = metrics.get("step")
        wr = metrics.get("eval/win_rate")
        extra = f" eval/win_rate={wr}" if wr is not None else ""
        print(f"  last metrics step={step}{extra}", flush=True)
    else:
        print("  no metrics.jsonl", flush=True)

    try:
        vastai_bin()
    except SystemExit:
        print("  vastai CLI not found; skip instance list", flush=True)
        return
    _ids, matched = _instance_ids_for_run(run_name, store)
    if not matched:
        print("  no matching vast.ai instance", flush=True)
        return
    for inst in matched:
        iid = inst.get("id") or inst.get("instance_id")
        status = inst.get("actual_status") or inst.get("status")
        gpu = inst.get("gpu_name")
        print(f"  instance {iid} status={status} gpu={gpu} "
              f"label={inst.get('label')}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        description="Joe vast.ai launcher (interruptible, R2-backed)")
    sub = p.add_subparsers(dest="cmd", required=True)

    launch = sub.add_parser(
        "launch", help="pack, upload, create (or adopt) an instance")
    launch.add_argument("--tier", default="",
                        help="S or M (default M; S when --smoke)")
    launch.add_argument("--run-name", default="",
                        help="R2 prefix identity; stamped if omitted")
    launch.add_argument("--gpu", default="",
                        help="h100, 4090, or comma-separated gpu_name list")
    launch.add_argument("--num-gpus", type=int, default=1)
    launch.add_argument("--bid", type=float, default=None,
                        help="interruptible bid $/hr (default: offer min_bid)")
    launch.add_argument("--disk", type=int, default=None)
    launch.add_argument("--image", default="",
                        help=f"docker image (default {DEFAULT_IMAGE})")
    launch.add_argument("--smoke", action="store_true",
                        help="S-config-sized few-iter run on a 4090-class GPU")
    launch.add_argument("--overrides", default="",
                        help="JSON object of Config field overrides")
    launch.add_argument("--pack-only", action="store_true",
                        help="upload code + launch.json; do not call vastai")
    launch.add_argument("--dry-run", action="store_true",
                        help="pack, upload, search offers; do not create")
    launch.add_argument("--force", action="store_true",
                        help="overwrite an existing launch.json")
    launch.add_argument("--instance-id", default="",
                        help="adopt this existing, running vast.ai instance "
                             "instead of searching offers and renting one")
    launch.set_defaults(func=cmd_launch)

    status = sub.add_parser("status", help="instance + R2 heartbeat + metrics")
    status.add_argument("--run-name", required=True)
    status.set_defaults(func=cmd_status)

    resume = sub.add_parser(
        "resume",
        help="destroy-before-create (or destroy-before-adopt with "
             "--instance-id) a replacement for the same run")
    resume.add_argument("--run-name", required=True)
    resume.add_argument("--bid", type=float, default=None)
    resume.add_argument("--instance-id", default="",
                        help="adopt this existing, running vast.ai instance "
                             "instead of renting a replacement")
    resume.set_defaults(func=cmd_resume)

    destroy = sub.add_parser("destroy", help="tear down the instance")
    destroy.add_argument("--run-name", required=True)
    destroy.add_argument("--purge-r2", action="store_true",
                         help="also delete the R2 prefix")
    destroy.add_argument("--purge-r2-only", action="store_true",
                         help="delete the R2 prefix; do not call vastai")
    destroy.set_defaults(func=cmd_destroy)

    sync = sub.add_parser(
        "sync-env",
        help="copy local R2_* into vast.ai account env vars (encrypted)")
    sync.set_defaults(func=cmd_sync_env)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not ONSTART.is_file():
        raise SystemExit(f"onstart script missing: {ONSTART}")
    args.func(args)


if __name__ == "__main__":
    main()
