import ast
import re
import inspect
from IPython.display import Latex

# Greek symbols mapping
greek_symbols = {
    'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma', 'delta': '\\delta',
    'epsilon': '\\epsilon', 'zeta': '\\zeta', 'eta': '\\eta', 'theta': '\\theta',
    'iota': '\\iota', 'kappa': '\\kappa', 'lambda_': '\\lambda', 'mu': '\\mu',
    'nu': '\\nu', 'xi': '\\xi', 'omicron': 'o', 'pi': '\\pi', 'rho': '\\rho',
    'sigma': '\\sigma', 'tau': '\\tau', 'upsilon': '\\upsilon', 'phi': '\\phi',
    'chi': '\\chi', 'psi': '\\psi', 'omega': '\\omega'
}

# LaTeX accents mapping
latex_accents = {
    'bar': '\\bar', 'hat': '\\hat', 'tilde': '\\tilde', 'vec': '\\vec',
    'dot': '\\dot', 'ddot': '\\ddot'
}

def extract_lines(code_str):
    return [line.rstrip() for line in code_str.split("\n") if line.strip() != ""]

def convert_to_greek(var_name):
    return greek_symbols.get(var_name.lower(), var_name)

# MODIFICATION: This function is updated to be more robust.
def format_latex_var(var_name):
    """
    Formats a Python variable name into a LaTeX string, handling Greek symbols,
    accents (e.g., 'bar_alpha'), and subscripts with escaped underscores.
    """
    if var_name in greek_symbols:
        return greek_symbols[var_name]

    if '_' in var_name:
        parts = var_name.split('_', 1)
        part1, part2 = parts[0], parts[1]

        if part1 in latex_accents:
            accent_cmd = latex_accents[part1]
            inner_var_latex = format_latex_var(part2)
            return f"{accent_cmd}{{{inner_var_latex}}}"
        else:
            base = convert_to_greek(part1)
            # MODIFICATION 1: Escape underscores in the subscript part to prevent KaTeX errors.
            subscript = part2.replace('_', r'\_')
            return f"{base}_{{{subscript}}}"

    return convert_to_greek(var_name)

def convert_line_names(line):
    tokens = re.split(r'(\W)', line)
    return ''.join([format_latex_var(t) if re.match(r'^[A-Za-z_]\w*$', t) else t for t in tokens])

def extract_latex_line_final(line):
    leading_spaces = len(line) - len(line.lstrip())
    stripped_line = line.lstrip()

    def ast_to_latex(node):
        """
        Recursively convert expressions to LaTeX. This is now more robust and
        handles tuples, slices, and complex subscripts.
        """
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return format_latex_var(node.id)
        elif isinstance(node, ast.Subscript):
            value_latex = ast_to_latex(node.value)
            slice_latex = ast_to_latex(node.slice)
            return f"{value_latex}[{slice_latex}]"
        elif isinstance(node, ast.BinOp):
            left = ast_to_latex(node.left)
            right = ast_to_latex(node.right)
            op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(node.op)]
            return f"{left} {op} {right}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                func_name_str = node.func.attr
            else:
                func_name_str = node.func.id
            func_name_latex = f"\\text{{{func_name_str.capitalize()}}}"
            args = [ast_to_latex(a) for a in node.args]
            kwargs = [f"{format_latex_var(kw.arg)}={ast_to_latex(kw.value)}" for kw in node.keywords]
            args_str = ", ".join(args + kwargs)
            return f"{func_name_latex}({args_str})"
        # MODIFICATION 2: Add handlers for tuples and slices to parse unpacking.
        elif isinstance(node, ast.Tuple):
            return ", ".join(ast_to_latex(e) for e in node.elts)
        elif isinstance(node, ast.Slice):
            lower = ast_to_latex(node.lower)
            upper = ast_to_latex(node.upper)
            step = ast_to_latex(node.step)
            if step:
                return f"{lower}:{upper}:{step}"
            return f"{lower}:{upper}"
        else:
            return convert_line_names(ast.unparse(node))

    try:
        tree = ast.parse(stripped_line)
        node = tree.body[0]

        def process_dist_call(func_call_node):
            # ... (this helper function remains the same)
            if isinstance(func_call_node.func, ast.Attribute):
                func_name_str = func_call_node.func.attr
            else:
                func_name_str = func_call_node.func.id
            func_name = "".join([part.capitalize() for part in func_name_str.split('_')])
            obs_var = None
            pos_args = [ast_to_latex(a) for a in func_call_node.args]
            kw_args = []
            for kw in func_call_node.keywords:
                if kw.arg == 'obs':
                    obs_var = format_latex_var(kw.value.id) if isinstance(kw.value, ast.Name) else ast_to_latex(kw.value)
                elif kw.arg != 'name':
                    key = format_latex_var(kw.arg)
                    if '_' in kw.arg:
                        key = f'\\text{{{key.replace("_", " ")}}}'
                    value = ast_to_latex(kw.value)
                    kw_args.append(f"{key}={value}")
            args_str = ", ".join(pos_args + kw_args)
            return func_name, args_str, obs_var

        is_dist_assignment = (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                              and 'dist' in ast.unparse(node.value.func))
        is_dist_expression = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                              and 'dist' in ast.unparse(node.value.func))

        if is_dist_assignment:
            func_name, args_str, obs_var = process_dist_call(node.value)
            lhs_var = node.targets[0].id
            lhs_latex = obs_var if obs_var else format_latex_var(lhs_var)
            return " " * leading_spaces + f"{lhs_latex} \\sim \\text{{{func_name}}}({args_str})"
        elif is_dist_expression:
            func_name, args_str, obs_var = process_dist_call(node.value)
            if obs_var is not None:
                return " " * leading_spaces + f"{obs_var} \\sim \\text{{{func_name}}}({args_str})"
        # MODIFICATION 3: Handle tuple unpacking on the left side of an assignment.
        elif isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Tuple):
                lhs_parts = [format_latex_var(t.id) for t in node.targets[0].elts]
                lhs = ", ".join(lhs_parts)
            else:
                lhs = format_latex_var(node.targets[0].id)
            rhs = ast_to_latex(node.value)
            return " " * leading_spaces + f"{lhs} = {rhs}"
        else:
            return " " * leading_spaces + convert_line_names(stripped_line)

    except Exception:
        return " " * leading_spaces + convert_line_names(stripped_line)

def to_latex(model):
    code = inspect.getsource(model)
    lines = extract_lines(code)
    lines_clean = [line.lstrip() for line in lines][1:]
    latex_lines = [extract_latex_line_final(line) for line in lines_clean]
    latex_str = "\\begin{align*}\n"
    latex_str += " \\\\\n".join(filter(None, reversed(latex_lines)))
    latex_str += "\\\\\n"
    latex_str += "\\end{align*}\n"
    display(Latex(latex_str))
    return latex_str