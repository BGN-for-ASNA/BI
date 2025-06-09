import re
from IPython.display import display, Latex, Math

greek_symbols = {
    'alpha': '\\alpha',
    'beta': '\\beta',
    'gamma': '\\gamma',
    'delta': '\\delta',
    'epsilon': '\\epsilon',
    'zeta': '\\zeta',
    'eta': '\\eta',
    'theta': '\\theta',
    'iota': '\\iota',
    'kappa': '\\kappa',
    'lambda': '\\lambda',
    'mu': '\\mu',
    'nu': '\\nu',
    'xi': '\\xi',
    'omicron': 'o',  # No direct LaTeX symbol for omicron, using "o"
    'pi': '\\pi',
    'rho': '\\rho',
    'sigma': '\\sigma',
    'tau': '\\tau',
    'upsilon': '\\upsilon',
    'phi': '\\phi',
    'chi': '\\chi',
    'psi': '\\psi',
    'omega': '\\omega'
}

def convert_to_greek(var_name):
    # Convert variable name to lowercase for case-insensitive matching
    var_name_lower = var_name.lower()
    # Check if the variable name has a corresponding Greek symbol
    if var_name_lower in greek_symbols:
        return greek_symbols[var_name_lower]
    else:
        return var_name

def extract_latex(command):
    # Define a regular expression pattern to match the desired parts of the command
    pattern = r"(\w+)\s*=\s*(\w+)\([^,]+,\s*[^,]+,\s*(.*)\)"
    match = re.match(pattern, command)
    
    if match:
        var_name = match.group(1)
        func_name = match.group(2)
        params = match.group(3)
        # Convert var_name to Greek symbol if applicable
        var_name_latex = convert_to_greek(var_name)
        # Construct the desired LaTeX text
        latex_text = f"{var_name_latex} = {func_name}({params})"
        return latex_text
    else:
        return None
# Example usage
command = "Sigma_i = exponential('Sigma_individual', [ni], 1)"
latex_text = extract_latex(command)
display(Latex(f'''${latex_text}$'''))



def display_se_kernel_latex():
    """Display the Squared Exponential Kernel in LaTeX format."""
    latex_code = r"""
    k_{\text{SE}}(x, x') = \sigma^2 \exp \left( -\frac{(x - x')^2}{2 \ell^2} \right)
    """
    display(Math(latex_code))

def display_periodic_kernel_latex():
    """Display the Periodic Kernel in LaTeX format."""
    latex_code = r"""
    k_{\text{Periodic}}(x, x') = \sigma^2 \exp \left( -\frac{2 \sin^2 \left( \frac{\pi |x - x'|}{p} \right)}{\ell^2} \right)
    """
    display(Math(latex_code))


def display_local_periodic_kernel_latex():
    """Display the Local Periodic Kernel in LaTeX format."""
    latex_code = r"""
    k_{\text{LocalPer}}(x, x') = k_{\text{Per}}(x, x') k_{\text{SE}}(x, x') = \sigma^2 \exp \left( -\frac{2 \sin^2 \left( \frac{\pi |x - x'|}{p} \right)}{\ell^2} \right) \exp \left( -\frac{(x - x')^2}{2 \ell^2} \right)
    """
    display(Math(latex_code))