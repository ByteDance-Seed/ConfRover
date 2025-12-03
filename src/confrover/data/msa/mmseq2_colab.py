#!/usr/bin/env python3

# All Bytedance's Modifications are Copyright (2024) Bytedance Ltd. and/or its affiliates.
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
# Original file was released under MIT, with the full license text available below.
#
# This modified file is released under the same license.
#
# ----------------
#
# MIT License
#
# Copyright (c) 2024 Bowen Jing, Bonnie Berger, Tommi Jaakkola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ----------------
#
# MIT License
#
# Copyright (c) 2021 Sergey Ovchinnikov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""mmseq2 MSA query using ColabFold'server modified from:
  - https://github.com/bjing2016/alphaflow/blob/master/scripts/mmseqs_query.py
  - https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py

----------------

"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import argparse
import os
import random
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, List, Tuple, Union

import requests
from tqdm import tqdm

from confrover.env import CONFROVER_VERSION, CachePaths
from confrover.utils import get_pylogger
from confrover.utils.misc import unique_dir
from confrover.utils.misc.cli import str2bool

logger = get_pylogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_PATH = CachePaths()

DEFAULT_AGENT = f"confrover/{CONFROVER_VERSION}"

COLABFOLD_URL = "https://api.colabfold.com"

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)

# =============================================================================
# Functions
# =============================================================================


def run_mmseqs2(
    x,
    prefix,
    use_env=True,
    use_filter=True,
    use_templates=False,
    filter=None,
    use_pairing=False,
    pairing_strategy="greedy",
    host_url=COLABFOLD_URL,
    user_agent: str = DEFAULT_AGENT,
) -> Union[Tuple[List[str], List[Any]], List[str]]:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    headers = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent (e.g., 'toolname/version contact@email') to help us debug in case of problems. This warning will become an error in the future."
        )

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        while True:
            error_count = 0
            try:
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching status from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching result from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        use_templates = False
        use_env = False
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # OLDTODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.info(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    raise Exception(
                        "MMseqs2 API is undergoing maintenance. Please try again in a few minutes."
                    )

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.info(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)
                    # if TIME > 900 and out["status"] != "COMPLETE":
                    #  # something failed on the server side, need to resubmit
                    #  N += 1
                    #  break

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. "
                        "If error persists, please try again an hour later."
                    )

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # templates
    if use_templates:
        logger.warning("use_templates=True has not been tested. Use with Caution.")

        templates = {}
        # print("seq\tpdb\tcid\tevalue")
        for line in open(f"{path}/pdb70.m8", "r"):
            p = line.rstrip().split()
            M, pdb, qid, e_value = p[0], p[1], p[2], p[10]
            M = int(M)
            if M not in templates:
                templates[M] = []
            templates[M].append(pdb)
            # if len(templates[M]) <= 20:
            #  print(f"{int(M)-N}\t{pdb}\t{qid}\t{e_value}")

        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = f"{prefix}_{mode}/templates_{k}"
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ",".join(TMPL[:20])
                response = None
                while True:
                    error_count = 0
                    try:
                        # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                        # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                        response = requests.get(
                            f"{host_url}/template/{TMPL_LINE}",
                            stream=True,
                            timeout=6.02,
                            headers=headers,
                        )
                    except requests.exceptions.Timeout:
                        logger.warning(
                            "Timeout while submitting to template server. Retrying..."
                        )
                        continue
                    except Exception as e:
                        error_count += 1
                        logger.warning(
                            f"Error while fetching result from template server. Retrying... ({error_count}/5)"
                        )
                        logger.warning(f"Error: {e}")
                        time.sleep(5)
                        if error_count > 5:
                            raise
                        continue
                    break
                with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:  # type:ignore
                    tar.extractall(path=TMPL_PATH)
                os.symlink("pdb70_a3m.ffindex", f"{TMPL_PATH}/pdb70_cs219.ffindex")
                with open(f"{TMPL_PATH}/pdb70_cs219.ffdata", "w") as f:
                    f.write("")
            template_paths[k] = TMPL_PATH

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    # return results

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
                # print(f"{n-N}\tno_templates_found")
            else:
                template_paths_.append(template_paths[n])

    if use_templates:
        return a3m_lines, template_paths_
    else:
        return a3m_lines


# =============================================================================
# Main
# =============================================================================


def batch_query(
    seqres_index_pairs: List[Tuple[str, str]],
    output_dir,
    max_query_size: int = 64,
    clean_tmp_dir: bool = True,
    tmp_dir: str | Path = "",
    deduplicate: bool = True,
) -> List[Tuple[str, str]]:
    """Batch query MMseqs2 server.

    Args:
        seqres_index_pairs (pd.DataFrame | List[Tuple[str, str]]): a DataFrame contains 'seqres' and 'index' columns, or a List of seqres-index pairs.
        output_dir (str): Output directory.
        max_query_size (int, optional): Maximum batch size. Defaults to 64.
        clean_tmp_dir (bool, optional): Clean temporary directory after query. Defaults to True.

    Returns:
        List[Tuple[str, str]]: List of seqres-index pairs.
    """
    # check duplicates in seqres
    if deduplicate:
        before = len(seqres_index_pairs)
        seqres_index_pairs = list({k: v for k, v in seqres_index_pairs}.items())
        after = len(seqres_index_pairs)
        logger.info(f"Deduplicated seqres: {before} -> {after} sequences")
    job_list = seqres_index_pairs
    logger.info(
        f"Querying ColabFold's MMSeqs2 server with {len(job_list)} sequences..."
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tmp_dir == "":
        tmp_dir = output_dir / ".tmp"
    else:
        tmp_dir = Path(tmp_dir)

    # Split seqres_list in batches
    if max_query_size:
        job_batches = [
            job_list[i : i + max_query_size]
            for i in range(0, len(job_list), max_query_size)
        ]
    else:
        job_batches = [job_list]

    # Process query job in batches
    updated_seqres_index_pairs = []

    for batch_ix, job_batch in enumerate(job_batches):
        tmp_dir_batch = tmp_dir / f"batch_{batch_ix}"
        tmp_dir_batch.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[Batch {batch_ix}] Querying {len(job_batch)} seqres, tmp_dir: {tmp_dir_batch}"
        )

        seqres_list = [seqres for seqres, _ in job_batch]
        msas = run_mmseqs2(
            seqres_list,
            prefix=tmp_dir_batch,
            user_agent=DEFAULT_AGENT,
            use_templates=False,
        )

        # Save it in AlphaFold compatible format
        # NOTE: we always use index[:2] as a level of subdirectory
        for (seqres, index), msa in zip(job_batch, msas):
            new_index = unique_dir(output_dir / index[:2] / index).stem
            if new_index != index:
                logger.warning(f"Index {index} already exists. Using {new_index}")
                index = new_index
            a3m_dir = output_dir / index[:2] / index / "a3m"
            a3m_dir.mkdir(parents=True, exist_ok=True)
            with open(a3m_dir / f"{index}.a3m", "w") as f:
                f.write(str(msa))
            updated_seqres_index_pairs.append((seqres, index))
        logger.info(f"[Batch {batch_ix}] Finished")
        if clean_tmp_dir:
            shutil.rmtree(tmp_dir_batch)  # clean up after each batch

    if clean_tmp_dir:
        shutil.rmtree(tmp_dir)

    return updated_seqres_index_pairs


def cli(args):
    from confrover.data.msa.msa_loader import MSALoader, _load_seqres_index_pairs

    logger.info(f"Will save MSA to: {args.msa_root}. Wait for 5 seconds...")
    time.sleep(5)
    msa_loader = MSALoader(msa_root=args.msa_root)

    seqres_index_pairs = [
        tuple(row) for row in _load_seqres_index_pairs(args.input_csv).to_numpy()
    ]

    msa_loader.query_msa(
        seqres_index_pairs=seqres_index_pairs,
        max_query_size=args.max_query_size,
        clean_tmp_dir=args.cleanup,
        overwrite=args.overwrite,
    )
    logger.info(f"Finished querying MSA. Saved to: {args.msa_root}")


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input_csv",
        type=str,
        metavar="<path>",
        required=True,
        default=argparse.SUPPRESS,
        help="An input csv file containing a set of seqres to query. Must include a 'seqres' column and an 'index' column.",
    )
    parser.add_argument(
        "--msa_root",
        type=str,
        metavar="<path>",
        default=str(DEFAULT_PATH.msa),
        help="Path to save MSA.",
    )
    parser.add_argument(
        "--max_query_size",
        type=int,
        metavar="<int>",
        default=64,
        help="Maximum number of sequences in each query. Reduce the number if encounter Timeout.",
    )
    parser.add_argument(
        "--cleanup",
        type=str2bool,
        metavar="true|false",
        nargs="?",
        const=True,
        default=True,
        help="Remove temp directory content (`{msa_root}/.tmp`) after each query.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        metavar="true|false",
        nargs="?",
        const=True,
        default=False,
        help="Overwrite existing MSA.",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query MSA using MMseqs2 server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = add_args(parser)
    args = parser.parse_args()
    cli(args)
