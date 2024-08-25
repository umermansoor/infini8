from autogen.agentchat import ConversableAgent

class Visualizer():
    def __init__(self):
        self._system_prompt = """
        You are a data visualization and Python expert who can create meaningful and insightful visualizations based on a dataset. You are given a list of insights that need to be visualized for a dataset. You will also be provided a dataset summary, link to the dataset and visualization library to use.

        You will output a well formatted Jupyter notebook containing the visualizations for each insight. The notebook should include the following sections:
        - Introduction (Markdown): Briefly introduce the dataset and the insights that are visualized.
        - Setup (Code): Import the necessary libraries and load the dataset. If using `pip install`, use the `-qqq` flag to suppress output.
        - Insight #1 (Markdown): Describe the insight and the visualization.
        - Insight #1 (Code): Generate the visualization for Insight #1.
        ...

        You may also receive feedback from a code reviewer, which you should review and address.
        
        You MUST only output the Jupyter notebook without any additional information.
        """
        pass

    def visualize(self, dataset_summary, insights, llm_config):
        programmer = ConversableAgent(
            name="Programmer",
            is_termination_msg=lambda x: x.get("content", "").find("LGTM") >= 0,
            llm_config=llm_config,
            human_input_mode="NEVER",
            system_message="""
                You are an expert Python programmer.
            """,
        )

        code_reviewer = ConversableAgent(
            name="CodeReviewer",
            is_termination_msg=lambda x: x.get("content", "").find("LGTM") >= 0,
            llm_config=llm_config,
            human_input_mode="NEVER",
            system_message=f"""
                ## Objective
                You are an expert Python engineer.
                Your job is to review the provided Python and provide constructive feedback to fix any issues or improve the code.

                Instructions:

                - Focus on identifying issues or offering suggestions for improvement. 
                - Keep your feedback concise and specific. Don't call out positive aspects of the code. Focus on areas that need improvement.
                - Review the dataset summary below and the insights code to make sure the code is refering to the correct columns and fields.
                - Make sure the descriptions and visualizations match.
                - If the code is ready to merge, conclude your review with "LGTM".
                - Do not rewrite the entire Jupyter Notebook.
                
                ## Dataset Summary
                Here's a summary of the dataset for which the Jupyter notebook insights were generated. 
                {dataset_summary}

                ## Insights Asked to Visualize
                Here are the insights that will be visualized in the Jupyter notebook:
                {insights}
            """,
        )

        res = code_reviewer.initiate_chat(
            recipient=programmer,
            message=f"""
                ## Objective
                {self._system_prompt}

                ## CSV File Path
                This is the file path to the dataset that you will be working with:
                {dataset_summary["file_path"]}

                ## Dataset Summary
                Here's a summary of the dataset for which the insights were generated:
                {dataset_summary}
            """,
            max_turns=2,
        )

        print(programmer.last_message()["content"])

        codeblock =  sanitize_codeblock(programmer.last_message()["content"])
        return codeblock


def sanitize_codeblock(codeblock) -> str:
    codeblock = codeblock.strip()
    codeblock = codeblock.replace("```python", "").replace("```", "")

    return codeblock





        




