import argparse
import subprocess
import glob
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

WORKFLOW_NAME = "RETROSPECTIVE_SIMULATION"


def get_script_location():
    return os.path.dirname(os.path.realpath(__file__))


def get_nextflow_dir():
    return os.path.join(get_script_location(), "..")


def get_args():
    parser = argparse.ArgumentParser(description="Run retrospective experiment")
    parser.add_argument(
        "--unmasked-screen",
        type=str,
        required=True,
        help="Path to unmasked experiment",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--n-dist-chunks",
        type=int,
        required=True,
        help="Number chunks to parallelize pairwise distance computation over",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        required=True,
        help="Number of chains to use for MCMC sampling",
    )
    parser.add_argument(
        "--nextflow-config-file",
        type=str,
        required=True,
        help="Path to nextflow config file",
    )
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


def validate_output_dir_and_get_result_files_as_dict(output_dir):
    experiment_tracker = glob.glob(
        os.path.join(output_dir, "*", "experiment_tracker_output.json")
    )
    advanced_experiment_glob = glob.glob(
        os.path.join(output_dir, "*", "advanced_experiment.h5")
    )
    n_remaining_plates = glob.glob(os.path.join(output_dir, "*", "n_remaining_plates"))

    advanced_experiment = list(advanced_experiment_glob)[0]
    experiment_tracker = list(experiment_tracker)[0]

    n_remaining_plates = list(n_remaining_plates)[0]

    with open(n_remaining_plates, "r") as f:
        n_remaining_plates = int(f.read().strip())

    return {
        "experiment_tracker": experiment_tracker,
        "advanced_experiment": advanced_experiment,
        "n_remaining_plates": n_remaining_plates,
    }


def run_nextflow_step(
    output_dir,
    unmasked_experiment,
    nextflow_config_file,
    n_chunks,
    n_chains,
    extra_args,
):
    experiment_name, _ = os.path.splitext(os.path.basename(unmasked_experiment))
    # list all directories in output directory
    contents_of_output_directory = glob.glob(output_dir + "/*")
    # filter to directories
    output_dirs = [
        x
        for x in contents_of_output_directory
        if os.path.isdir(x) and os.path.basename(x).isdigit()
    ]

    workflow_path = os.path.join(
        get_nextflow_dir(),
        "workflows",
        "nf-core",
        "batchie",
        "retrospective_simulation",
        "main.nf",
    )

    next_output_dir = os.path.join(output_dir, str(len(output_dirs) + 1))
    current_iteration = len(output_dirs) + 1
    # create next output directory
    os.mkdir(next_output_dir)

    logger.info(f"Running iteration {current_iteration}")

    if len(output_dirs) == 0:
        subprocess.check_call(
            [
                "nextflow",
                "run",
                workflow_path,
                "-entry",
                WORKFLOW_NAME,
                "--experiment_name",
                experiment_name,
                "--unmasked_experiment",
                unmasked_experiment,
                "--n_chunks",
                str(n_chunks),
                "--n_chains",
                str(n_chains),
                "--outdir",
                next_output_dir,
                "-c",
                nextflow_config_file,
            ]
            + extra_args,
            cwd=next_output_dir,
        )
        return True

    else:
        latest_result = max(output_dirs, key=lambda x: int(os.path.basename(x)))

        latest_result_files = validate_output_dir_and_get_result_files_as_dict(
            latest_result
        )

        if latest_result_files["n_remaining_plates"] == 0:
            logger.info("No remaining plates, exiting")
            return False
        else:
            logger.info(f"Running iteration {current_iteration}")
            logger.info(f"{latest_result_files['n_remaining_plates']} remaining plates")

        subprocess.check_call(
            [
                "nextflow",
                "run",
                workflow_path,
                "-entry",
                WORKFLOW_NAME,
                "--experiment_name",
                experiment_name,
                "--experiment_tracker",
                latest_result_files["experiment_tracker"],
                "--masked_experiment",
                latest_result_files["advanced_experiment"],
                "--unmasked_experiment",
                unmasked_experiment,
                "--n_chunks",
                str(n_chunks),
                "--n_chains",
                str(n_chains),
                "--outdir",
                next_output_dir,
                "-c",
                nextflow_config_file,
            ]
            + extra_args,
            cwd=next_output_dir,
        )
        return True


def main():
    args, remaining_args = get_args()
    while True:
        should_run_again = run_nextflow_step(
            output_dir=os.path.abspath(args.output_dir),
            unmasked_experiment=os.path.abspath(args.unmasked_experiment),
            nextflow_config_file=os.path.abspath(args.nextflow_config_file),
            n_chunks=args.n_dist_chunks,
            n_chains=args.n_chains,
            extra_args=remaining_args,
        )

        if not should_run_again:
            break


if __name__ == "__main__":
    main()
