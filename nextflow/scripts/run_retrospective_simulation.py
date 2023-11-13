import argparse
import subprocess
import glob
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_script_location():
    return os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def get_nextflow_dir():
    return os.path.abspath(os.path.join(get_script_location(), ".."))


def get_base_config():
    return os.path.abspath(os.path.join(get_nextflow_dir(), "..", "nextflow.config"))


def get_repository_root():
    return os.path.abspath(os.path.join(get_script_location(), "..", ".."))


def get_main_nf_file():
    return os.path.abspath(os.path.join(get_repository_root(), "main.nf"))


def get_args():
    parser = argparse.ArgumentParser(
        description="Run retrospective simulation, all arguments are passed to nextflow"
    )
    parser.add_argument(
        "--screen",
        type=str,
        required=True,
        help="Path to screen",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to output directory"
    )
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


def validate_output_dir_and_get_result_files_as_dict(output_dir):
    simulation_tracker = list(
        glob.glob(os.path.join(output_dir, "*", "simulation_tracker_output.json"))
    )
    advanced_screen_glob = list(
        glob.glob(os.path.join(output_dir, "*", "advanced_screen.h5"))
    )
    test_screen_glob = list(
        glob.glob(os.path.join(output_dir, "..", "1", "*", "test.screen.h5"))
    )
    n_remaining_plates = list(
        glob.glob(os.path.join(output_dir, "*", "n_remaining_plates"))
    )

    if (
        len(simulation_tracker) == 0
        or len(advanced_screen_glob) == 0
        or len(n_remaining_plates) == 0
    ):
        return None

    advanced_screen = advanced_screen_glob[0]
    simulation_tracker = simulation_tracker[0]

    n_remaining_plates = n_remaining_plates[0]
    test_screen = test_screen_glob[0]

    with open(n_remaining_plates, "r") as f:
        n_remaining_plates = int(f.read().strip())

    return {
        "simulation_tracker": simulation_tracker,
        "advanced_screen": advanced_screen,
        "test_screen": test_screen,
        "n_remaining_plates": n_remaining_plates,
    }


def run_nextflow_step(
    output_dir,
    screen,
    extra_args,
):
    os.makedirs(output_dir, exist_ok=True)

    experiment_name, _ = os.path.splitext(os.path.basename(screen))
    # list all directories in output directory
    contents_of_output_directory = glob.glob(output_dir + "/*")
    # filter to directories
    output_dirs = [
        x
        for x in contents_of_output_directory
        if os.path.isdir(x) and os.path.basename(x).isdigit()
    ]

    current_iteration = 0
    latest_result_files = None

    if len(output_dirs) > 0:
        latest_results_in_order = sorted(
            output_dirs, key=lambda x: int(os.path.basename(x)), reverse=True
        )

        for latest_result in latest_results_in_order:
            latest_result_files = validate_output_dir_and_get_result_files_as_dict(
                latest_result
            )
            if latest_result_files is not None:
                current_iteration = int(os.path.basename(latest_result))
                break

    current_iteration = current_iteration + 1

    next_output_dir = os.path.join(output_dir, str(current_iteration))

    if latest_result_files is None:
        os.makedirs(next_output_dir, exist_ok=True)
        logger.info(f"Running iteration {current_iteration}")

        subprocess.check_call(
            [
                "nextflow",
                "run",
                get_main_nf_file(),
                "--mode",
                "retrospective",
                "--screen",
                screen,
                "--name",
                experiment_name,
                "--outdir",
                next_output_dir,
                "-work-dir",
                os.path.join(next_output_dir, "work"),
            ]
            + extra_args,
            cwd=get_repository_root(),
        )
        return True

    else:
        if latest_result_files["n_remaining_plates"] == 0:
            logger.info("No remaining plates, exiting")
            return False
        else:
            logger.info(f"Running iteration {current_iteration}")
            logger.info(f"{latest_result_files['n_remaining_plates']} remaining plates")

        os.makedirs(next_output_dir, exist_ok=True)

        subprocess.check_call(
            [
                "nextflow",
                "run",
                get_main_nf_file(),
                "--mode",
                "retrospective",
                "--simulation_name",
                experiment_name,
                "--simulation_tracker",
                latest_result_files["simulation_tracker"],
                "--training_screen",
                latest_result_files["advanced_screen"],
                "--test_screen",
                latest_result_files["test_screen"],
                "--outdir",
                next_output_dir,
                "-work-dir",
                os.path.join(next_output_dir, "work"),
            ]
            + extra_args,
            cwd=get_repository_root(),
        )
        return True


def main():
    args, remaining_args = get_args()
    while True:
        should_run_again = run_nextflow_step(
            output_dir=os.path.abspath(args.outdir),
            screen=os.path.abspath(args.screen),
            extra_args=remaining_args,
        )

        if not should_run_again:
            break


if __name__ == "__main__":
    main()
