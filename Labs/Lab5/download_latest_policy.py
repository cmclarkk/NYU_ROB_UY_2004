import argparse
import pathlib
import shutil

import wandb


def download_latest_model(
    project_name,
    entity_name=None,
    run_number: int = None,
    model_dir="/home/pi/pupperv3-monorepo/ros2_ws/src/neural_controller/launch",
    model_name="test_policy.json",
):
    """
    Downloads the latest model from a W&B project.

    :param project_name: The name of the W&B project.
    :param entity_name: The W&B entity (username or team). If None, fetches from logged-in user.
    :param run_number: The run number to download. If None, downloads the latest run.
    :param model_dir: The directory where the model will be downloaded.
    :param model_name: The name to copy the model as.
    :return: None
    """

    # Initialize the API
    api = wandb.Api()
    
    # If entity_name is not provided, fetch from logged-in user
    if entity_name is None:
        try:
            # Get the default entity for the logged-in user
            entity_name = api.default_entity
            print(f"Using entity from logged-in user: {entity_name}")
        except Exception as e:
            print(f"ERROR: Could not fetch entity name from wandb. Are you logged in?")
            print(f"       Run 'wandb login' first.")
            print(f"       Error: {e}")
            return

    # Fetch the latest run
    runs = api.runs(f"{entity_name}/{project_name}")

    # Check if there are any runs
    if not runs:
        print("No runs found in the project.")
        return

    # find the run whose names ends in -run_number
    if run_number is not None:
        runs = [run for run in runs if run.name.endswith(f"-{run_number}")]
        if not runs:
            print(f"No runs found with the number {run_number}.")
            return
        run = runs[0]
        print("Using run: ", run.name)
    else:
        # Get the latest run (assuming runs are sorted by start time by default)
        # sort runs by the number at the end of the name
        runs = sorted(runs, key=lambda run: int(run.name.split("-")[-1]))
        run = runs[-1]
        print(f"Latest run: {run.name}")

    # get the artifact with the name that contains .json
    artifacts = [art for art in run.logged_artifacts() if ".json" in art.name]
    if not artifacts:
        print("ERROR: No model .json files found in the run.")
        return
        
    art = artifacts[0]
    print("Using artifact: ", art.name)

    # remove the :[version] from the name
    base_name = art.name.split(":")[0]
    print("Base name: ", base_name)

    # get folder of this script
    script_dir = pathlib.Path(__file__).parent
    model_dir = pathlib.Path(model_dir)

    downloaded_filepath = script_dir / model_dir / pathlib.Path(base_name)
    save_filepath = script_dir / model_dir / model_name

    # Download the file
    art.download(root=script_dir / model_dir)
    print(f"Model downloaded to: {downloaded_filepath}")
    model_name = pathlib.Path(model_name)
    shutil.copyfile(downloaded_filepath, save_filepath)
    print(f"Model copied to: {save_filepath}")


if __name__ == "__main__":
    # Define your project name
    project_name = "pupperv3-mjx-rl"

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--run_number", type=int, default=None, 
                          help="Run number to download. If not specified, downloads the latest run.")
    argparser.add_argument("--entity", type=str, default=None,
                          help="W&B entity name. If not specified, uses the logged-in user's entity.")
    args = argparser.parse_args()

    # Call the function to download the latest model
    # entity_name will be auto-fetched from logged-in user if not specified
    download_latest_model(
        project_name,
        entity_name=args.entity,
        run_number=args.run_number,
    )