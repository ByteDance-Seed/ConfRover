#!/bin/bash

# CLI run for example forward simulation
confrover generate --job_config examples/example_fwd.json --output examples/output/cli_example_fwd --model ConfRover-base-20M-v1.0

# CLI run for example independent ensemble sampling
confrover generate --job_config examples/example_iid.json --output examples/output/cli_example_iid --model ConfRover-base-20M-v1.0

# CLI run for example interpolating two conformations
confrover generate --job_config examples/example_interp.json --output examples/output/cli_example_interp --model ConfRover-interp-20M-v1.0