import nbformat
from nbformat import read

from nbconvert.preprocessors import ExecutePreprocessor

class Executor:
    def __init__(self):
        self._system_prompt = """
        You are a Python expert. Your task is to review the provided Jupyter notebook and an associated errors.
        You will rewrite the entire notebook to fix the error and pass it back to the user.

        """

    def _execute_notebook(self, notebook_path: str) -> str:
        """
        Execute a Jupyter notebook and returns the output as a string. 
        The output is a empty string is the notebook execution was successful, including warning.
        The output contains the error message if the notebook execution failed.
        """
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': './'}})
        except Exception as e:
            return str(e)

        return ""
    
    def _execute_notebook_cells(self, notebook_path: str) -> dict:
        """
        Executes one cell at a time in a Jupyter notebook. It stops at the first error encountered and returns the error message, cell code, and cell output.
        """
        with open(notebook_path) as f:
            nb = read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        errors = {}
        
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':  # Only execute code cells
                try:
                    ep.preprocess(nb, {'metadata': {'path': './'}}, cell_index=index)
                except Exception as e:
                    errors[index] = {
                        "error": str(e),
                        "cell_code": cell.source,
                    }
                    break

        return errors
    
    def execute_with_llm(self, notebook_path: str) -> str:
        """
        Execute a Jupyter notebook, which may contain some errors. Errors are passed to the LLM for correction.
        """
        errors =  self._execute_notebook_cells(notebook_path)

        print ("Errors: ", errors)

        return ""
    
    # run the execute_with_llm function with this: /Users/umermansoor/Documents/GitHub/infini8/notebooks/output/notebook.ipynb.

executor = Executor()
executor.execute_with_llm("/Users/umermansoor/Documents/GitHub/infini8/notebooks/output/notebook.ipynb")

        





