# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ConfRover Command Line Interface"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import argparse

from confrover.data.msa.mmseq2_colab import add_args as _add_msa_args
from confrover.data.msa.mmseq2_colab import cli as _msa_cli
from confrover.data.pretrain_repr.openfold.make_openfold_repr import (
    add_args as _add_openfold_args,
)
from confrover.data.pretrain_repr.openfold.make_openfold_repr import (
    cli as _openfold_cli,
)
from confrover.inference import add_args as _add_generate_args
from confrover.inference import cli as _generate_cli
from confrover.utils import get_pylogger

log = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Components
# =============================================================================


def build_parser():
    main_parser = argparse.ArgumentParser(
        prog="confrover",
        description="See below for sub-commands",
        usage="%(prog)s <command> [args]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = main_parser.add_subparsers(dest="command", help="")
    subparsers.required = True

    # Query MSA
    msa_parser = subparsers.add_parser(
        "query_msa",
        help="Query MSA using ColabFold's MMSeq2 server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    msa_parser = _add_msa_args(msa_parser)
    msa_parser.set_defaults(func=_msa_cli)

    # Generate Openfold repr
    openfold_repr_parser = subparsers.add_parser(
        "openfold_repr",
        help="Make OpenFold representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    openfold_repr_parser = _add_openfold_args(openfold_repr_parser)
    openfold_repr_parser.set_defaults(func=_openfold_cli)

    # Run ConfRover generation
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate ConfRover samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate_parser = _add_generate_args(generate_parser)
    generate_parser.set_defaults(func=_generate_cli)

    return main_parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
