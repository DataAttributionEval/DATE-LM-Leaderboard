import gradio as gr
from gradio_leaderboard import Leaderboard
import pandas as pd
import json
import os

import filePaths 
import nameMapping
import pr

##################### Leaderboard Paths + Variables #####################

pathLst = filePaths.PATHLIST
pretrain_10K, pretrain_30K, finetune = pathLst[0], pathLst[1], pathLst[2]
toxicity_homogeneous, toxicity_heterogeneous, factual = pathLst[3], pathLst[4], pathLst[5]

import nameMapping
leaderboard_names = nameMapping.LEADERBOARD_NAMES
trainingNamesSet = nameMapping.TRAINING_LEADERBOARDS

########################## Data Loading ###########################

def load_leaderboard_data(file_path):
    """
    Load leaderboard data from JSON file.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def add_ranking_column(data, id):
    """
    Add ranking column dynamically for display based on selected metric aggregation.
    """
    if id == 'toxicity': # Toxicity: AUPRC
        key_fn = lambda x: x["AUPRC"]
    elif id == 'factual': # Factual: Avg of Recall@50 and MRR
        key_fn = lambda x: (x["Recall@50"]+x["MRR"])/2
    elif id == 'pretrain': # Pretrain: Avg
        key_fn = lambda x: x["avg"]
    else: # FineTune: Avg of Metrics
        key_fn = lambda x: (x["MMLU"]+x["GSM8K"]+x["BBH"])/3

    sorted_data = sorted(data, key=key_fn, reverse=True)
    for index, entry in enumerate(sorted_data):
        entry["Rank"] = index + 1 
    return sorted_data

def load_data(filePath, id):
    """
    Load initial leaderboard data.
    """
    return pd.DataFrame(add_ranking_column(load_leaderboard_data(filePath), id))

pretrain_10K_data = load_data(pretrain_10K, "pretrain")
pretrain_30K_data = load_data(pretrain_30K, "pretrain")
finetune_data = load_data(finetune, "finetune")
homogeneous_data = load_data(toxicity_homogeneous, "toxicity")
heterogeneous_data = load_data(toxicity_heterogeneous, "toxicity")
factual_data = load_data(factual, "factual")

########################## Leaderboard Columns + Helpers ###########################

def get_leaderboard_columns(leaderboard_name):
    """
    Returns the Expected Columns for Leaderboard
    """
    leaderboardNameMap = nameMapping.DROPDOWN_NAME_MAPPING
    if leaderboard_name in leaderboardNameMap["toxicity"]:
        return nameMapping.TOXICITY_COLS
    elif leaderboard_name in leaderboardNameMap["factual"]:
        return nameMapping.FACTUAL_COLS
    elif leaderboard_name in leaderboardNameMap["finetune"]:    
        return nameMapping.FINETUNE_COLS
    else: # pretrain
        return nameMapping.PRETRAIN_COLS

def get_model_sizes(leaderboard_name):
    """
    Returns Model Sizes for Applications Leaderboards
    """
    nameFileMapping = {"Homogeneous": toxicity_homogeneous, 
                       "Heterogeneous": toxicity_heterogeneous,
                       "Factual Attribution": factual}
    leaderboardJson = load_leaderboard_data(nameFileMapping[leaderboard_name])
    modelSizes = set()

    for row in leaderboardJson:
        modelSizes.add(row["Model Size"])
    
    return ['All'] + list(modelSizes)

################### Submission Helper Functions #############################

def update_fields(leaderboard):
    """
    Determine visibility of group / display additional metrics in submission area.
    """
    nameMap = nameMapping.DROPDOWN_NAME_MAPPING
    return {
        pretrain_group: gr.update(visible=(leaderboard in nameMap['pretrain'])),
        finetune_group: gr.update(visible=(leaderboard in nameMap['finetune'])),
        toxicity_group: gr.update(visible=(leaderboard in nameMap['toxicity'])),
        factual_group: gr.update(visible=(leaderboard in nameMap['factual']))
    }

def validate_inputs(*inputFields):
    (leaderboard_dropdown, method_name, method_dropdown, model_name, model_size, paper_link, scores, 
     pre_avg, pre_sciq, pre_arc_easy, pre_arc_chall, pre_logiqa, 
     pre_boolq, pre_hellaswag, pre_piqa, pre_wino, pre_open, 
     fine_mmlu, fine_gsm, fine_bbh, 
     tox_toxicChat, tox_xsTest, tox_jbb, tox_auprc, 
     fac_recall, fac_mrr) = inputFields
    
    if not all([leaderboard_dropdown, model_name, method_name, method_dropdown, model_size]):
        raise gr.Error("All fields must be filled out and with the correct type.")
    
    if not paper_link:
        raise gr.Error("Please fill in out the Paper/Code/Contact Link info.")
    
    if not scores:
        raise gr.Error("Please upload data attribution scores in .pt file.")
    
    # Check Metrics Non-Empty
    nameMap = nameMapping.DROPDOWN_NAME_MAPPING
    # nameMap['pretrain'] nameMap['finetune'] nameMap['finetune'] nameMap['factual']
    if leaderboard_dropdown in nameMap['pretrain']:
        metricsList = [pre_avg, pre_sciq, pre_arc_easy, pre_arc_chall, pre_logiqa, pre_boolq, pre_hellaswag, pre_piqa, pre_wino, pre_open]
    elif leaderboard_dropdown in nameMap['finetune']:
        metricsList = [fine_mmlu, fine_gsm, fine_bbh]
    elif leaderboard_dropdown in nameMap['toxicity']:
        metricsList = [tox_toxicChat, tox_xsTest, tox_jbb, tox_auprc]
    elif leaderboard_dropdown in nameMap['factual']:
        metricsList = [fac_recall, fac_mrr]

    if not all(metricsList):
        raise gr.Error("Metrics must be filled out.")
    if not all(metric > 0 for metric in metricsList):
        raise gr.Error("Metrics must be positive.")

    
######## Dynamically Update Ranking when Filtering on Model Size ###############

def update_rankings(filtered_df, id):
    df_with_rank = filtered_df.copy() # create copy to avoid modifying original
    
    if id == 'toxicity': # Toxicity: AUPRC
        df_with_rank = df_with_rank.sort_values(by="AUPRC", ascending=False)
    elif id == 'factual': # Factual: Avg of Recall@50 and MRR
        average_scores  = df_with_rank[["Recall@50", "MRR"]].mean(axis=1)
        sorted_index = average_scores .sort_values(ascending=False).index
        df_with_rank = df_with_rank.loc[sorted_index]
    
    df_with_rank["Rank"] = range(1, len(df_with_rank) + 1) # Add rank column
    
    return df_with_rank

def filter_and_rank(df, filter_value, id):
    if filter_value == "All":
        filtered_df = df
    else:
        filtered_df = df[df["Model Size"] == filter_value]
    return update_rankings(filtered_df, id)

def rerank_leaderboard(filter_value, dfPath, idNum):
    df = load_data(dfPath, idNum)
    filtered_ranked_df = filter_and_rank(df, filter_value, idNum)
    return filtered_ranked_df

#################### Leaderboards Code ##############################

with gr.Blocks(css="""
    body, .gradio-container {
        font-family: 'roboto';
    }
""") as demo:
    gr.Markdown("""
    # Data Attribution Methods Leaderboards
    """)
    gr.Markdown(f"""
    Survey and ranking of data attribution methods on data selection and 
                downstream application tasks for the Date-LM Evaluation paper.

    **Leaderboard Submission**:                        
    - To submit your team's scores, click on the "Submit Scores" tab.
                
    **Data Attribution Method Categories**: 
    - Gradient (ex. GradDot, GradSim, LESS, DataInf, EKFAC)
    - Similarity (ex. RepSim)
    - Modeling (ex. MATES)
    - Lexical (ex. BM25)
    - Baseline (ex. GradSafe, OpenAI Moderation, LLM Classifiers)
    - Other                                

    **Search Feature**: 
    - Input the name of the method you would like to search / filter for, and
                then press "Enter". The original row from the leaderboard table will be displayed.   
    """
    )

    with gr.Tabs():
        with gr.TabItem("Training Data Selection"):
            with gr.Tabs():  # Subtabs container
                with gr.TabItem("Pre-Training (10K)"):  # Subtab
                    gr.Markdown("""DATE-LM Task Description: Trained pythia-1B model on Fineweb using 
                                Lambada reference dataset. Testing results conducted on 10K step checkpoint.
                                
                                Ranking Metric: highest score in **avg** column""") # description
                    l1 = Leaderboard(
                            value=pd.DataFrame(pretrain_10K_data),
                            select_columns=get_leaderboard_columns("Pre-Training (10K)"),
                            search_columns=['Method'],
                            filter_columns=["Attribution Method Type", "Method", "avg"],
                        )
                with gr.TabItem("Pre-Training (30K)"):
                    gr.Markdown("""DATE-LM Task Description: Trained pythia-1B model on Fineweb using 
                                Lambada reference dataset. Testing results conducted on 30K step checkpoint.
                                
                                Ranking Metric: highest score in **avg** column""")
                    l2 = Leaderboard(
                        value=pd.DataFrame(pretrain_30K_data),
                        select_columns=get_leaderboard_columns("Pre-Training (30K)"),
                        search_columns=["Method"],
                        filter_columns=["Attribution Method Type", "Method", "avg"],
                    )
                with gr.TabItem("Fine-Tuning"):
                    gr.Markdown("""DATE-LM Task Description: Targeted instruction tuning setting.
                                 Given a diverse instruction set and a eval dataset, we select data that would yield 
                                optimal performance on the eval data. For this task, the training data pool is 
                                Tulu3 (unfiltered) and the eval data is MMLU, GSM8K, and BBH. 
                                
                                Ranking Metric: average of the **MMLU**, **GSM8K**, and **BBH** scores""")
                    l3 = Leaderboard(
                        value=pd.DataFrame(finetune_data),
                        select_columns=get_leaderboard_columns("Fine-Tuning"),
                        search_columns=["Method"],
                        filter_columns=["Attribution Method Type", "MMLU", "GSM8K", "BBH"],
                    )
        with gr.TabItem("Applications"):
            with gr.Tabs():
                with gr.TabItem("Toxicity/Bias"):
                    with gr.Tabs(): 
                        with gr.TabItem("Homogeneous"):
                            gr.Markdown("""DATE-LM Task Description: This leaderboard presents detection AUPRC results of baseline methods and data attribution methods in the homogenous setting 
                                        (i.e., detecting small amount of toxic/biased data embedded into larger benign data).
                                        
                                        Ranking Metric: **AUPRC** (an average of ToxicChat, XSTest-response, JailBreakBench)""")
                            category_filter4 = gr.Dropdown(
                                choices=get_model_sizes("Homogeneous"),
                                value="All",
                                label="Filter Model Size"
                            ) # ensures page placement above leaderboard
                            l4 = Leaderboard(
                                value=pd.DataFrame(homogeneous_data),
                                select_columns=get_leaderboard_columns("Homogeneous"),
                                search_columns=["Method"],
                                filter_columns=["Attribution Method Type", "Model", "AUPRC"],
                            )
                            data_path4 = gr.Textbox(value=toxicity_homogeneous, visible=False)
                            id_str4 = gr.Textbox(value="toxicity", visible=False)
                            category_filter4.change(
                                fn=rerank_leaderboard,
                                inputs=[category_filter4, data_path4, id_str4],
                                outputs=[l4]
                            )
                        with gr.TabItem("Heterogeneous"):
                            gr.Markdown("""DATE-LM Task Description: This leaderboard presents detection AUPRC results of baseline methods and data attribution methods in the heterogeneous setting 
                                        (i.e., safety-aligned examples that resemble unsafe data in format but contain safe responses).
                                        
                                        Ranking Metric: **AUPRC** (an average of ToxicChat, XSTest-response, JailBreakBench)""")
                            category_filter5 = gr.Dropdown(
                                choices=get_model_sizes("Heterogeneous"),
                                value="All",
                                label="Filter Model Size"
                            )
                            l5 = Leaderboard(
                                value=pd.DataFrame(heterogeneous_data),
                                select_columns=get_leaderboard_columns("Heterogeneous"),
                                search_columns=["Method"],
                                filter_columns=["Attribution Method Type", "Model", "AUPRC"]
                            )
                            data_path5 = gr.Textbox(value=toxicity_heterogeneous, visible=False)
                            id_str5 = gr.Textbox(value="toxicity", visible=False)
                            category_filter5.change(
                                fn=rerank_leaderboard,
                                inputs=[category_filter5, data_path5, id_str5],
                                outputs=[l5]
                            )
                with gr.TabItem("Factual Attribution"):
                    gr.Markdown("""DATE-LM Task Description: Identifying the specific training examples that support a model's generated facts.
                                        
                                   Ranking Metric: average of **Recall@50** and **MRR**""")
                    category_filter6 = gr.Dropdown(
                        choices=get_model_sizes("Factual Attribution"),
                        value="All",
                        label="Filter Model Size"
                    )
                    l6 = Leaderboard(
                        value=pd.DataFrame(factual_data),
                        select_columns=get_leaderboard_columns("Factual Attribution"),
                        search_columns=["Method"],
                        filter_columns=["Attribution Method Type", "Model", "Recall@50", "MRR"],
                    )
                    data_path6 = gr.Textbox(value=factual, visible=False)
                    id_str6 = gr.Textbox(value="factual", visible=False)
                    category_filter6.change(
                        fn=rerank_leaderboard,
                        inputs=[category_filter6, data_path6, id_str6],
                        outputs=[l6]
                    )
        with gr.TabItem("Submit Scores 🚀"):
            with gr.Column():
                gr.Markdown("""### Submit Your Score to a Leaderboard
                            
                Note: Please first select the leaderboard you would like to submit to. This will display the fields for the 
                            corresponding metrics that are needed. 
                """)

                leaderboard_dropdown = gr.Dropdown(
                    label="Select Leaderboard",
                    choices=nameMapping.LEADERBOARD_NAMES,
                    value=None
                )

                method_name = gr.Textbox(label="Method Name")
                method_dropdown = gr.Dropdown(
                    label="Method Type",
                    choices=["Gradient", "Similarity", "Representation-Based", "Modeling", "Baseline", "Lexical", "Other"],
                    value=None
                )

                # model_size = gr.Dropdown(
                #     label="Model Size",
                #     choices=["400M", "1B", "3B", "7B"],
                #     value=None
                # ) 
                model_name = gr.Textbox(label="Model Name")
                model_size = gr.Textbox(label="Model Size (ex. 410M, 1B, 8B)")

                paper_link = gr.Textbox(label="Paper/Code/Contact Link") 
                
                scores = gr.File(label='Upload Data Attribution Scores File (.pt)', height=150, file_types=[".pt"])
                
                # Dynamically Display Needed Fields for Each Leaderboard Type

                with gr.Column(visible=False) as pretrain_group:
                    pre_avg = gr.Number(label="Avg")
                    pre_sciq = gr.Number(label="sciq")
                    pre_arc_easy = gr.Number(label="arc_easy")
                    pre_arc_chall = gr.Number(label="arc_challenge")
                    pre_logiqa = gr.Number(label="logiqa")
                    pre_boolq = gr.Number(label="boolq")
                    pre_hellaswag = gr.Number(label="hellaswag")
                    pre_piqa = gr.Number(label="piqa")
                    pre_wino = gr.Number(label="winogrande")
                    pre_open = gr.Number(label="openbookqa")

                with gr.Column(visible=False) as finetune_group:
                    fine_mmlu = gr.Number(label="MMLU")
                    fine_gsm = gr.Number(label="GSM8K")
                    fine_bbh = gr.Number(label="BBH")

                with gr.Column(visible=False) as toxicity_group: 
                    tox_toxicChat = gr.Number(label="ToxicChat")
                    tox_xsTest = gr.Number(label="XSTest-response")
                    tox_jbb = gr.Number(label="JailBreakBench")
                    tox_auprc = gr.Number(label="AUPRC") 

                with gr.Column(visible=False) as factual_group:
                    fac_recall = gr.Number(label="Recall@50")
                    fac_mrr = gr.Number(label="MRR")

                # with gr.Group(visible=False) as training_group:
                #     acc = gr.Number(label="Accuracy")
                    
                # applications_group = gr.Column(visible=False)
                # with applications_group:
                #     f1_score = gr.Number(label="F1")
                #     auprc_score = gr.Number(label="AUPRC")
                #     acc1 = gr.Number(label="Accuracy")
                
                # Submit button
                submit_button = gr.Button("Submit")

                leaderboard_dropdown.change(update_fields, inputs=[leaderboard_dropdown], outputs=[pretrain_group, finetune_group, toxicity_group, factual_group])
                
                # information lists
                inputsList = [leaderboard_dropdown, method_name, method_dropdown, model_name, model_size, paper_link, scores, \
                              pre_avg, pre_sciq, pre_arc_easy, pre_arc_chall, pre_logiqa, pre_boolq, pre_hellaswag, pre_piqa, pre_wino, pre_open, \
                              fine_mmlu, fine_gsm, fine_bbh, \
                              tox_toxicChat, tox_xsTest, tox_jbb, tox_auprc, \
                              fac_recall, fac_mrr]
                
                submit_button.click(
                    validate_inputs, inputs=inputsList, outputs=[]
                ).success(fn=pr.submit_and_open_PR, inputs=inputsList, outputs=[gr.Textbox(label="Opened PR on Github")])
                
if __name__ == "__main__":
    demo.launch(debug=True)
