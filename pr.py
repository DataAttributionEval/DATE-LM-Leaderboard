from github import Github
from datetime import datetime, timezone
import os
import json
import nameMapping

###################### Push Up to Github #################################

REPO_NAME = "DataAttributionEval/DATE-LM-Leaderboard"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
reviewer = "cathyjiao"

def submit_and_open_PR(selected_leaderboard, *new_entry):
    # Unpack data
    (method_name, method_dropdown, model_name, model_size, paper_link, scores,
     pre_avg, pre_sciq, pre_arc_easy, pre_arc_chall, pre_logiqa, 
     pre_boolq, pre_hellaswag, pre_piqa, pre_wino, pre_open, 
     fine_mmlu, fine_gsm, fine_bbh, 
     tox_toxicChat, tox_xsTest, tox_jbb, tox_auprc, 
     fac_recall, fac_mrr) = new_entry

    # Save metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch_name = f"{timestamp}-{''.join(method_name.split())}-{method_name}"

    submission_data = {
        "Metadata Path": f"submissions/{branch_name}/metadata.json",
        "Leaderboard": selected_leaderboard,
        "Date of Submission": timestamp,
        "Method": method_name,
        "Attribution Method Type": method_dropdown,
        "Model": model_name,
        "Model Size": model_size,
        "Paper/Code/Contact Link": paper_link
    }

    nameMap = nameMapping.DROPDOWN_NAME_MAPPING
    if selected_leaderboard in nameMap['pretrain']:
        fields = ["avg", "sciq", "arc_easy", "arc_challenge", "logiqa", \
                  "boolq", "hellaswag", "piqa", "winogrande", "openbookqa"]
        vals = [pre_avg, pre_sciq, pre_arc_easy, pre_arc_chall, pre_logiqa, \
                pre_boolq, pre_hellaswag, pre_piqa, pre_wino, pre_open]
        submission_data.update(dict(zip(fields, vals)))
    elif selected_leaderboard in nameMap['finetune']:
        submission_data["MMLU"] = fine_mmlu
        submission_data["GSM8K"] = fine_gsm
        submission_data["BBH"] = fine_bbh
    elif selected_leaderboard in nameMap['toxicity']:
        submission_data["ToxicChat"] = tox_toxicChat
        submission_data["XSTest-response"] = tox_xsTest
        submission_data["JailBreakBench"] = tox_jbb
        submission_data["AUPRC"] = tox_auprc
    elif selected_leaderboard in nameMap['factual']:
        submission_data["Recall@50"] = fac_recall
        submission_data["MRR"] = fac_mrr

    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)

    # Create a unique branch name
    base = repo.get_branch("main")
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)

    # Upload score files
    with open(scores.name, "rb") as f:
        repo.create_file(
            path=f"submissions/{branch_name}/{os.path.basename(scores.name)}",
            message=f"Uploaded Scores File",
            content=f.read(),
            branch=branch_name
    )

    # Add PR metadata.json file
    repo.create_file(
        path=f"submissions/{branch_name}/metadata.json",
        message="Submission Form Metadata",
        content=json.dumps(submission_data, indent=2),
        branch=branch_name
    )

    # Create pull request
    pr = repo.create_pull(
        title=f"[HF Leaderboard Submission] {method_name} for {selected_leaderboard}",
        body=f"Auto-Generated Leaderboard Submission PR from HF Space\n{json.dumps(submission_data, indent=4)}",
        head=branch_name,
        base="main"
    )
    pr.add_to_labels("leaderboard-submission")
    pr.create_review_request(reviewers=[reviewer])

    return f"✅ PR created: {pr.html_url}"