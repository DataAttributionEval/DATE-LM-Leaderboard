import json
from pathlib import Path
import sys
import os

def create_data_entry(submitted_json):
    entry =  {
        "Rank": 0,
    }

    submitted_dict = dict(submitted_json)
    del submitted_dict["Metadata Path"]
    del submitted_dict["Leaderboard"]
    del submitted_dict["Date of Submission"]
    
    entry.update(submitted_dict)

    return submitted_json.get("Leaderboard"), entry

def merge_metadata(metadata_path):
    file_map = {"Pre-Training (10K)": ['data', 'DataSelection', 'pythia1b-10k-lambada.json'],
            "Pre-Training (30K)": ['data', 'DataSelection', 'pythia1b-30k-lambada.json'], 
            "Fine-Tuning": ['data', 'DataSelection', 'finetune.json'],
            "Homogeneous": ['data', 'Applications', 'toxicity-homogeneous.json'],
            "Heterogeneous": ['data', 'Applications', 'toxicity-heterogeneous.json'],
            "Factual Attribution": ['data', 'Applications', 'factual.json']
           }

    with open(metadata_path, "r") as f:
        metadata_json = json.load(f)
        leaderboard, entry = create_data_entry(metadata_json)
        target_file = file_map.get(leaderboard)
        repo_root = os.environ.get('GITHUB_WORKSPACE', os.getcwd())

        target_path = os.path.join(repo_root, *target_file)
        
        if target_path is not None:
            with open(target_path, "r") as f:
                existing = json.load(f)
        else:
            # existing data should not be empty
            raise ValueError("Could not retreive file content.")

        existing.append(entry)

        # Save updated list
        with open(target_path, "w") as f:
            json.dump(existing, f, indent=2)

if __name__ == "__main__":
    file_path = sys.argv[1]
    merge_metadata(file_path)
