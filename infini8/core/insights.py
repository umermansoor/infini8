import pandas as pd
from autogen.agentchat import AssistantAgent
import json

class InsightsGenerator:
     def __init__(self) -> None:
        """Initialize"""
        self._system_prompt = """
        You are an experienced data analyst who suggest the best visualizations for a dataset when provided with a summary of the dataset and a specified persona. The visualizations you recommend must adhere to BEST PRACTICES e.g., use bar charts instead of pie charts for comparing quantities. You must suggest visualizations that must be meaningful insightful and interesting to the relevant persona.

        Each visualization suggestion must include:

        1. **Expected Insights**: 
            - **Question**: A specific, actionable question that the visualization will help answer.
            - **Insight**: Describe what insight or key takeaway the visualization is expected to provide.
  
        2. **Visualization**:
            - **Type**: Specify the type of visualization (e.g., bar chart, scatter plot, heatmap, etc.).
            - **Axes and Labels**: Clearly state which variables should be plotted on the X-axis and Y-axis, and provide any necessary labels or units.
            - **Additional Recommendations**: Include suggestions for styling, color choices, use of legends, and any other relevant design elements that enhance clarity or focus.

        3. **Data Handling**:
            - **Fields to Use**: Explicitly mention which fields or columns from the dataset should be used for the visualization.
            - **Aggregations**: Describe any necessary aggregations (e.g., sum, average, median) that should be applied to the data before visualizing it.
            - **Transformations**: Suggest any transformations (e.g., logarithmic scales, normalization, binning) that are needed to prepare the data for visualization.
            - **Filtering**: Recommend any filters that should be applied to the dataset (e.g., date ranges, specific categories) to focus the analysis.
            - **Common Issues**: Warn about any common pitfalls or issues (e.g., misleading scales, overplotting) that should be avoided in the visualization and mitigation strategies.

        Only output the JSON representation of the insights, without any preamble or additional information.
        ```
        {
            "insights": [
                {
                "index": 0,
                "expected_insight": {
                    "question": "How have monthly sales trended over the past year across different regions?",
                    "insight": "The line chart should reveal whether there are any seasonal trends or significant variations in sales across different regions. It will help identify which regions are consistently performing well and which might require additional attention."
                },
                "visualization": {
                    "type": "Line Chart",
                    "axes_and_labels": {
                    "x_axis": "Date (Month-Year format)",
                    "y_axis": "Total Sales",
                    "additional_labels": [
                        "Region"
                    ]
                    },
                    "additional_recommendations": {
                    "styling": "Use different colors for each region to clearly distinguish trends.",
                    "legend": "Include a legend to identify each region.",
                    "gridlines": "Enable gridlines for better readability of trends."
                    }
                },
                "data_handling": {
                    "fields_to_use": [
                        "date",
                        "region_name",
                        "sales"
                    ],
                    "aggregations": {
                        "sales": "Sum by month and region"
                    },
                    "transformations": {
                        "date": "Convert to Month-Year format"
                    },
                    "filtering": {
                        "date_range": "Last 12 months"
                    },
                    "common_issues": {
                        "issue": "Overplotting may occur if there are are over 1000 regions.",
                        "mitigation_strategy": "Consider focusing on top-performing regions or using a separate chart for each region."
                    }
                }
                },
                ...
            ]
        }
        ```
        """
        pass

     def generate_insights(self, dataset_summary: dict, llm_config, num_insights = 5, persona: str = ""):
        """
        Generate insights based on the dataset summary and specified persona.

        Parameters:
        - dataset_summary (dict): A summary of the dataset containing information about the columns.
        - persona (str): The persona for which insights are being generated.

        Returns:
        - list[dict]: A list of dictionaries, each containing an insight with a question, visualization type, and rationale.
        """

        if not persona:
            persona = "You are a principle data analyst you will come up with a meaningful insights that can be derived from the dataset"


        assistant = AssistantAgent("assistant", llm_config=llm_config)
        res = assistant.generate_reply(
            messages=[
                {
                    "content": f"""
                        ## Objective
                        {self._system_prompt}
                        You will generate {num_insights} insights.

                        ## Persona
                        The generated insights should be tailored to the following persona:
                        {persona}

                        ## Dataset Summary
                        Here's a summary of the dataset:
                        {json.dumps(dataset_summary, indent=4)}
                    """,
                    "role": "user",
                }
            ]
        )

        return res

        

    