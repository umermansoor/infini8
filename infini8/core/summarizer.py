import pandas as pd
from autogen.agentchat import AssistantAgent
import json
import os


class Summarizer:
    def __init__(self) -> None:
        """Initialize the Summarizer class."""
        self._system_prompt = """
        You are a highly skilled data analyst that can annotate datasets. Here are the instructions you should follow when annotating the dataset, you must:

        1. Provide the name of the dataset in `dataset_name`. This is a short name that accurately describes the dataset.
        2. Provide a brief description of the dataset in `dataset_description`. This should be a one or two sentence summary of the dataset.
        3. For each column in the dataset, provide a description of the data in the column in the `description` field.
        4. For each column in the dataset, provide a single-word `llm_type` that describes the column based on its values e.g., longitude, email, ip_address, phone_number, company_name, os_version, browser_type, purchase_amount, is_mobile, referral_code, etc.
        5. Highlight important variables or features, including their relationships, distributions, and roles (e.g., target variable, key predictor) in the `key_variables` field.
        You must only return JSON without any extra information.
        """
        self.summary = None

    @staticmethod
    def _cast_to_serializable_type(dtype: str, value):
        """Cast value to the appropriate type to ensure it is JSON serializable."""
        if pd.api.types.is_float_dtype(dtype):
            return float(value)
        elif pd.api.types.is_integer_dtype(dtype):
            return int(value)
        return value

    def _extract_column_metadata(
        self, df: pd.DataFrame, num_samples: int = 3
    ) -> list[dict]:
        """
        Extract detailed information for each column in a pandas DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame from which to extract detailed column information.
        - num_samples (int): The number of sample values to include for each column.

        Returns:
        - list[dict]: A list of dictionaries, where each dict contains detailed metadata for a single column.
        """

        def get_numerical_properties(column: pd.Series) -> dict:
            return {
                "dtype": "number",
                "standard_deviation": self._cast_to_serializable_type(column.dtype, column.std()),
                "min": self._cast_to_serializable_type(column.dtype, column.min()),
                "max": self._cast_to_serializable_type(column.dtype, column.max()),
                "num_unique_values": column.nunique(),
            }

        def get_date_properties(column: pd.Series) -> dict:
            try:
                min_date = column.min()
                max_date = column.max()
            except TypeError:
                cast_date_col = pd.to_datetime(column, errors="coerce")
                min_date = cast_date_col.min()
                max_date = cast_date_col.max()
            return {"dtype": "date", "min": min_date, "max": max_date}

        def get_string_properties(column: pd.Series) -> dict:
            try:
                pd.to_datetime(column, errors="raise")
                return {"dtype": "date"}
            except ValueError:
                if column.nunique() / len(column) < 0.5:
                    return {"dtype": "category"}
                return {"dtype": "string"}

        def get_column_properties(column: pd.Series) -> dict:
            if pd.api.types.is_numeric_dtype(column):
                return get_numerical_properties(column)
            elif pd.api.types.is_bool_dtype(column):
                return {"dtype": "boolean"}
            elif pd.api.types.is_categorical_dtype(column):
                return {"dtype": "category"}
            elif pd.api.types.is_datetime64_any_dtype(column):
                return get_date_properties(column)
            elif pd.api.types.is_object_dtype(column):
                return get_string_properties(column)
            return {"dtype": str(column.dtype)}

        def add_common_properties(column: pd.Series, properties: dict) -> dict:
            non_null_values = column[column.notnull()].unique()
            n_samples = min(num_samples, len(non_null_values))
            samples = (
                pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
            )

            properties.update(
                {
                    "samples": samples,
                    "num_unique_values": column.nunique(),
                    "llm_type": "",
                    "description": "",
                }
            )
            return properties

        properties_list = []
        for column_name in df.columns:
            column = df[column_name]
            properties = get_column_properties(column)
            properties = add_common_properties(column, properties)
            properties_list.append({"column": column_name, "properties": properties})

        return properties_list

    def summarize(self, csv_path: str, llm_config, dataset_details = ""):
        """
        Summarize the dataset and annotate it using the LLM.

        Parameters:
        - csv_path (str): The file path of the CSV to summarize.
        - llm_config: configuration for LLM (passed to autogen)

        Returns:
        - The LLM annotated dataset
        """
        df = pd.read_csv(csv_path)
        column_metadata = self._extract_column_metadata(df)

        base_summary = {
            "dataset_name": "",
            "dataset_description": "",
            "key_variables": "",
            "column_metadata": column_metadata,
        }

        if dataset_details:
            base_summary["user_provided_details"] = dataset_details

        assistant = AssistantAgent("assistant", llm_config=llm_config)

        reply = assistant.generate_reply(
            messages=[
                {
                    "content": f"""
                {self._system_prompt}

                Here's the summary you need to annotate:
                {base_summary}
                """,
                    "role": "user",
                }
            ]
        )

        if isinstance(reply, str):
            reply = json.loads(reply)

        # Append the file_path to the dataset_summary so it can be used later
        if os.path.isabs(csv_path):
            reply["file_path"] = csv_path
        else:
            reply["file_path"] = os.path.abspath(csv_path)

        # Append number of rows and columns to the dataset_summary
        reply["num_rows"] = df.shape[0]
        reply["num_columns"] = df.shape[1]

        return reply