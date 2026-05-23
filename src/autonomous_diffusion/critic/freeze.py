"""CLI: freeze the current contract for the locked test.

Writes a YAML freeze record. The locked-test runner refuses to start if the
contract's locked_test_freeze fields have drifted from this snapshot.
"""
from __future__ import annotations

import click
import yaml

from .guards import freeze_contract


@click.command()
@click.option("--contract", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", required=True, type=click.Path(dir_okay=False))
def main(contract: str, out: str) -> None:
    with open(contract) as fh:
        c = yaml.safe_load(fh)
    record = freeze_contract(c)
    with open(out, "w") as fh:
        yaml.safe_dump(record, fh, sort_keys=False)
    click.echo(f"frozen {len(record['frozen_fields'])} fields; sha256={record['sha256'][:12]}")
    click.echo(f"wrote {out}")


if __name__ == "__main__":
    main()
