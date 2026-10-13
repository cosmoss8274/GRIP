"""Encode a synthetic input or score one JSON record with a trained GRIP model."""

import argparse
import hashlib
import json
from pathlib import Path


# Format demonstration only: these are not an experimental peptide or receptor.
SYNTHETIC_RECORD = {"epitope_sequence": "AAAAAAAA", "mhc": "ACDEFGHIK"}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    from grip_models import load_config, prepare_record

    if args.record is None:
        record = dict(SYNTHETIC_RECORD)
    else:
        with args.record.open(encoding="utf-8") as stream:
            record = json.load(stream)

    if args.command == "encode":
        config = load_config(args.config)
        batch = prepare_record(record, config)
        return {
            "input_source": "synthetic_example" if args.record is None else "record_file",
            "component_lengths": {
                name.removesuffix("_lengths"): int(value.item())
                for name, value in batch.items() if name.endswith("_lengths")
            },
            "tensor_count": len(batch),
        }

    from grip_models import BaseEnergyModel, load_weights, score_record

    digest = file_sha256(args.checkpoint)
    config = load_config(args.config)
    model = BaseEnergyModel(config)
    load_weights(
        model, args.checkpoint, sha256=digest, state_dict_key=args.state_dict_key,
    )
    model.to(args.device).eval()
    return score_record(model, record, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode", help="encode a synthetic or supplied record without weights")
    score = commands.add_parser("score", help="score a supplied record with a local checkpoint")
    for command in (encode, score):
        command.add_argument("--record", type=Path, required=command is score,
                             help="JSON file containing one peptide/context record")
        command.add_argument("--config", type=Path,
                             help="matching configuration JSON; omitted uses core defaults")
    score.add_argument("--checkpoint", type=Path, required=True, help="local trained checkpoint")
    score.add_argument("--state-dict-key", help="top-level state-dictionary key, if the file is wrapped")
    score.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0 (default: cpu)")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
