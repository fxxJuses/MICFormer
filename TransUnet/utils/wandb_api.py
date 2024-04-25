import os
import wandb
import shutil
import torch
import subprocess
import pandas as pd

from tqdm import tqdm


def get_wandb_run_data(entity, project, run_id):
    """
    Retrieve the logged data from a WandB run.
    
    Args:
        entity (str): 
            The entity name.
        project (str):
            The project name.
        run_id (str):
            The run id.

    Returns:
        history_df (pandas.DataFrame):
            The logged data as a pandas DataFrame.
    """
    # define the project and run you want to retrieve data for
    run_path = os.path.join(entity, project, run_id)

    # initialize the wandb run object
    api = wandb.Api()
    run = api.run(run_path)

    # get the history of the run
    history = run.history()

    # convert the history to a pandas dataframe
    history_df = pd.DataFrame(history)

    # remove useless columns
    history_df = history_df[[col for col in history_df.columns if col.startswith("train") or col.startswith("val")]]

    # remove all rows with nan values
    history_df = history_df.dropna()

    # reset index to start at 0
    history_df = history_df.reset_index(drop=True)

    return history_df


def sync_offline_runs(folder_path, delete=False):
    """
    Sync offline wandb runs to the cloud.
    
    Args:
        folder_path (str):
            The path to the folder containing the offline wandb runs.
        delete (bool):
            Whether to delete the offline runs after syncing.
    """
    # get all wandb folders
    wandb_folders = [os.path.join(folder_path, folder) for folder in sorted(os.listdir(folder_path)) if folder.startswith("offline-run-")]

    # sync each wandb folder
    for wandb_folder in tqdm(wandb_folders):
        # check if run is finished
        if os.path.exists(os.path.join(wandb_folder, "files", "wandb-summary.json")):
            # sync wandb folder
            subprocess.run(["wandb", "sync", wandb_folder])

            # delete wandb folder
            if delete:
                shutil.rmtree(wandb_folder)

def download_weights_wandb(username, project_name, artifact_name, artifact_version, output_dir):
    '''
    Download model weights from wandb

    Args:
        username: str
            Username of the wandb account
        project_name: str
            Name of the wandb project
        artifact_name: str
            Name of the wandb artifact
        artifact_version: str
            Version of the wandb artifact
        output_dir: str
            Path to the directory where the weights should be saved
    '''
    # set up the api instance
    artifact_path = os.path.join(username, project_name, f"{artifact_name}:{artifact_version}")

    # set up the weights artifact
    api = wandb.Api()

    # setup output directory
    output_dir = os.path.join(output_dir, artifact_name, project_name, artifact_version)

    # check if the artifact exists on wandb
    try:
        # get the artifact
        artifact = api.artifact(artifact_path)
        # create the output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # download the artifact
        weights_path = artifact.download(root=output_dir)
        print(f"----- Model weights downloaded to {weights_path}")
    except:
        print(f"----- Artifact {artifact_name} does not exist in the project {project_name}")