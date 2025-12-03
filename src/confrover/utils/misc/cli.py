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

from __future__ import annotations

import argparse


def str2bool(v: str) -> bool:
    """A argparse 'type' function to convert string to boolean value.

    Usage:
        parser.add_argument("--flag", type=str2bool, default=False)

        function.py --flag True/true/1/yes/y

    Args:
        v (str): String value to be converted.

    Returns:
        bool: Boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def group_values(
    parser: argparse.ArgumentParser, args: argparse.Namespace, group: str | None = None
):
    """Return {group_title: {dest: value}} for all groups on this parser."""
    out = {}
    for grp in parser._action_groups:  # private API but standard in practice
        gd = {}
        for act in grp._group_actions:
            if (
                act.dest
                and act.dest is not argparse.SUPPRESS
                and hasattr(args, act.dest)
            ):
                gd[act.dest] = getattr(args, act.dest)
        out[grp.title] = gd
    if group is None:
        return out
    else:
        return out[group]
