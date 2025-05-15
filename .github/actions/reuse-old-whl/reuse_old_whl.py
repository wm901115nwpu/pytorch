import argparse
import os
import subprocess
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import requests


FORCE_REBUILD_LABEL = "ci-force-rebuild"


@lru_cache
def get_merge_base() -> str:
    merge_base = subprocess.check_output(
        ["git", "merge-base", "HEAD", "origin/main"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return merge_base


@lru_cache
def get_head_sha() -> str:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return sha


def query_github_api(url: str) -> Any:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    }
    response = requests.get(url, headers=headers)
    return response.json()


def ok_changed_file(file: str) -> bool:
    # Return true if the file is in the list of allowed files to be changed to
    # reuse the old whl
    if (
        file.startswith("torch/")
        and file.endswith(".py")
        and not file.startswith("torch/csrc/")
    ):
        return True
    if file.startswith("test/") and file.endswith(".py"):
        return True
    return False


def is_main_branch() -> bool:
    print(
        f"Checking if we are on main branch: merge base {get_merge_base()}, head {get_head_sha()}"
    )
    return get_merge_base() == get_head_sha()


def check_labels_for_pr() -> bool:
    # Check if the current commit is part of a PR and if it has the
    # FORCE_REBUILD_LABEL
    head_sha = get_head_sha()
    url = f"https://api.github.com/repos/pytorch/pytorch/commits/{head_sha}/pulls"
    response = query_github_api(url)
    print(f"Found {len(response)} PRs for commit {head_sha}")
    for pr in response:
        labels = pr.get("labels", [])
        for label in labels:
            if label["name"] == FORCE_REBUILD_LABEL:
                print(f"Found label {FORCE_REBUILD_LABEL} in PR {pr['number']}.")
                return True
    return False


def check_changed_files() -> bool:
    # Return true if all the changed files are in the list of allowed files to
    # be changed to reuse the old whl
    merge_base = get_merge_base()
    changed_files = (
        subprocess.check_output(
            ["git", "diff", "--name-only", merge_base, "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        .strip()
        .split()
    )

    for file in changed_files:
        if not ok_changed_file(file):
            print(f"File {file} is not allowed to be changed.")
            return False
        else:
            print(f"File {file} is allowed to be changed.")
    return True


def find_old_whl(workflow_id: str, build_environment: str) -> bool:
    # Find the old whl on s3 and download it to artifacts.zip
    if build_environment is None:
        print("BUILD_ENVIRONMENT is not set.")
        return False
    merge_base = get_merge_base()
    print(f"Merge base: {merge_base}, workflow_id: {workflow_id}")

    workflow_runs = query_github_api(
        f"https://api.github.com/repos/pytorch/pytorch/actions/workflows/{workflow_id}/runs?head_sha={merge_base}&branch=main&per_page=100"
    )
    if workflow_runs.get("total_count", 0) == 0:
        print("No workflow runs found.")
        return False
    for run in workflow_runs.get("workflow_runs", []):
        # Look in s3 for the old whl
        run_id = run["id"]
        try:
            url = f"https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/{run_id}/{build_environment}/artifacts.zip"
            print(f"Checking for old whl at {url}")
            response = requests.get(
                url,
            )
            if response.status_code == 200:
                with open("artifacts.zip", "wb") as f:
                    f.write(response.content)
                    print(f"Found old whl file from s3: {url}")
                    return True
        except requests.RequestException as e:
            print(f"Error checking for old whl: {e}")
            continue
    return False


def unzip_artifact_and_replace_files() -> None:
    # Unzip the artifact and replace files
    subprocess.check_output(
        ["unzip", "-o", "artifacts.zip", "-d", "artifacts"],
    )
    os.remove("artifacts.zip")

    # Rename wheel into zip
    wheel_path = Path("artifacts/dist").glob("*.whl")
    print(wheel_path)
    for path in wheel_path:
        new_path = path.with_suffix(".zip")
        os.rename(path, new_path)
        print(f"Renamed {path} to {new_path}")
        # Unzip the wheel
        subprocess.check_output(
            ["unzip", "-o", new_path, "-d", f"artifacts/dist/{new_path.stem}"],
        )
        # Copy python files into the artifact
        subprocess.check_output(
            ["rsync", "-avz", "torch", f"artifacts/dist/{new_path.stem}"],
        )

        # Zip the wheel back
        subprocess.check_output(
            [
                "zip",
                "-r",
                f"artifacts/dist/{new_path.stem}.zip",
                f"artifacts/dist/{new_path.stem}",
            ],
        )

        # Reame back to whl
        os.rename(new_path, path)

        # Remove the extracted folder
        subprocess.check_output(
            ["rm", "-rf", f"artifacts/dist/{new_path.stem}"],
        )

    # Rezip the artifact
    subprocess.check_output(["zip", "-r", "artifacts.zip", "."], cwd="artifacts")
    subprocess.check_output(
        ["mv", "artifacts/artifacts.zip", "."],
    )
    return None


def set_output() -> None:
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print("reuse=true", file=env)
    else:
        print("::set-output name=reuse::true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check for old whl files.")
    parser.add_argument("--workflow-id", type=str, required=True, help="Workflow ID")
    parser.add_argument(
        "--build-environment", type=str, required=True, help="Build environment"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    can_use_old_whl = (
        not is_main_branch() and check_changed_files() and not check_labels_for_pr()
    )
    # if not can_use_old_whl:
    #     print(f"Cannot use old whl, because we are on main branch or changed files.")
    #     set_build_output()
    if not find_old_whl(args.workflow_id, args.build_environment):
        print("No old whl found.")
        sys.exit(0)
    unzip_artifact_and_replace_files()
    set_output()
