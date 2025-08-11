#%%
"""
This script generates R documentation for Python distributions using the `reticulate` package.
It extracts docstrings from Python classes and methods, converts them into `roxygen2` format, and saves the documentation in a JSON file for later use.
"""
from np_dists import UnifiedDist
import ollama
dist_classes = {}
for name in dir(UnifiedDist):
    obj = getattr(UnifiedDist, name)
    if (
        obj and not name.startswith("_")
        and name not in ["Distribution", "ExpandedDistribution", "TransformedDistribution", "IndependentDistribution"]
    ):
        dist_classes[name] = obj
dist_classes

#%%
doc = dist_classes['binomial'].__doc__
def build_roxygen2_docstring(docstring):
    prompt = rf"""
    **You are an expert R programmer specializing in creating documentation for R packages that wrap Python     libraries using the `reticulate` package. Your task is to convert a given Python docstring into a   comprehensive and user-friendly `roxygen2` documentation block for its corresponding R wrapper function.  **

    **Core Context & Conversion Rules:**

    1.  **Wrapper Functions:** Assume the R function is a wrapper that calls the Python function. The   primary goal of the documentation is to be clear for an *R user* so R oibject types are converted to  Python object types within the function.

    2. Latex formulas need to be wrapped in
        #' '''{{latex}}
        #' \[ 
        ....
        #' \]
        ' '''  
        to be rendered correctly in R documentation.

    3.  **Argument Type Translation:** Your primary responsibility is to translate the description of   Python types into their R equivalents.
        *   **Python `tuple`:** This is a critical conversion. Arguments that expect a Python `tuple` (e.   g., `shape=(10,)`) are handled in the R wrapper by accepting a numeric vector (e.g., `c(10)`). The     `@param` documentation *must* reflect this, instructing the R user to provide a vector.
        *   **Python `None`:** This is translated to `reticulate::py_none()` in R. Mention this in the  parameter description where applicable.
        *   **Python `bool`:** `True` and `False` correspond to `TRUE` and `FALSE` in R. Describe the   parameter as `Logical`.
        *   **Arrays (`numpy.ndarray`, `jax.Array`):** Describe these in R terms, such as "a numeric    vector, matrix, or array."

    4.  **Docstring Structure Mapping:** Convert the structure of the Python docstring (typically NumPy or  Google style) to `roxygen2` tags:
        *   **Title/Summary:** The first line of the docstring becomes the `@title`. The subsequent     paragraph(s) form the `@description`.
        *   **`Args:` section:** Each argument becomes a separate `@param` tag. Ensure the name and     description are accurate for the R wrapper.
        *   **`Returns:` section:** This becomes the `@return` tag, describing the object that the R    function will return.
        *   **`Example*` section:** This becomes the `@examples` tag. **You must rewrite the Python code    examples into valid R code**,with the following structure: 
            ```r
            library(BI)
            m = importBI('cpu')
            bi.dist.distributionName(required_args_with_defaults, sample=True)
            ```
            Adapt the distribution name by the correct one.

        *   **Mathematical Notation:** Convert any `.. math::` blocks into the `roxygen2` `\deqn` format.
        *   **Source/Reference Links:** If the docstring contains a "Wrapper of" or "See Also" section with     a URL, place it in an `@seealso` tag using the `\url` macro.

    5.  **Standard Roxygen Tags:** Always include these standard tags in your generated output:
        *   `@export`: To make the function available to package users.
        *   `@importFrom reticulate py_none tuple`: To properly import the necessary `reticulate` functions     that are commonly used in the wrappers.
    6. Do not return documentation between triple quotes.

    **Your Task:**

    Based on the principles and rules outlined above, please convert the following Python docstring into    the `roxygen2` format. **Return only the roxygen2 documentation, without any additional text or    explanations.**

    **{docstring}**
    """


    response = ollama.chat(
        model='gemma3:12b',
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ],
        options={
        'temperature': 0  # lower = more deterministic
        }
    )
    return response['message']['content']

#%%
import os
import json

file_path = "Doc_R.json"

if os.path.exists(file_path):
    try:
        with open(file_path, "r") as f:
            DOC = json.load(f)
    except (json.JSONDecodeError, IOError):
        DOC = {}
else:
    DOC = {}


for name, doc in dist_classes.items():

    if name not in DOC.keys():
        print(f"Generating roxygen2 documentation for {name}-----------------------------------------")
        doc = build_roxygen2_docstring(dist_classes[name].__doc__)
        print(doc)
        DOC[name] = doc


# %%
import json
with open('Doc_R.json', 'w') as f:
    json.dump(DOC, f)