import ast
import re
import inspect
try:
    from IPython.display import display, Latex
except ImportError:
    display = print
    Latex = str

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
    # Drop blank lines AND comment-only lines: a `# note` line otherwise reached
    # the LaTeX output verbatim and broke the render.
    return [line.rstrip() for line in code_str.split("\n")
            if line.strip() != "" and not line.strip().startswith("#")]

def convert_to_greek(var_name):
    # Case-sensitive first, so Sigma (a covariance matrix) does not collapse
    # onto sigma (a scalar sd).
    if var_name in greek_symbols:
        return greek_symbols[var_name]
    lowered = var_name.lower()
    if lowered in greek_symbols and var_name[:1].isupper():
        sym = greek_symbols[lowered]
        # Only some Greek letters have a distinct uppercase command.
        if sym.lstrip("\\") in ("gamma", "delta", "theta", "lambda", "xi",
                                "pi", "sigma", "upsilon", "phi", "psi", "omega"):
            return "\\" + sym.lstrip("\\").capitalize()
        return sym
    return greek_symbols.get(lowered, var_name)

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

def _callee_str(func_node):
    try:
        return ast.unparse(func_node)
    except Exception:
        return ""


# NumPyro primitives whose SITE NAME is the first positional argument.
_NUMPYRO_SITE_FNS = ("numpyro.sample", "sample",
                     "numpyro.deterministic", "deterministic")


def _is_dist_callee(func_node):
    """True for m.dist.xxx(...) or numpyro.sample(...)-style sampling calls.

    Matching the bare substring 'dist' also fired on np.distance(...) and
    dist_matrix(...), rendering them as sampling statements.
    """
    try:
        name = ast.unparse(func_node)
    except Exception:
        return False
    parts = name.split(".")
    return any(p == "dist" for p in parts) or name in (
        "numpyro.sample", "sample", "numpyro.deterministic", "deterministic")


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
            if isinstance(node.op, ast.Div):
                return f"\\frac{{{left}}}{{{right}}}"
            if isinstance(node.op, ast.Pow):
                return f"{left}^{{{right}}}"
            # A bare dict lookup raised KeyError for **, %, //, @ -- swallowed
            # by the outer except into an unformatted fallback.
            op = {ast.Add: '+', ast.Sub: '-', ast.Mult: r'\cdot',
                  ast.Mod: r'\bmod', ast.FloorDiv: '//',
                  ast.MatMult: r'\cdot'}.get(type(node.op))
            if op is None:
                return convert_line_names(ast.unparse(node))
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

    # Block headers (`with numpyro.plate("i", N):`, `for i in range(N):`) are
    # not parseable as standalone statements, so they used to fall through to
    # the raw-source fallback. Render them as an explicit scope annotation.
    if stripped_line.endswith(":"):
        header = stripped_line[:-1].strip()
        if header.startswith("with "):
            inner = header[len("with "):].split(" as ")[0].strip()
            try:
                call = ast.parse(inner, mode="eval").body
                if isinstance(call, ast.Call) and "plate" in ast.unparse(call.func):
                    args = ", ".join(ast_to_latex(a) for a in call.args)
                    return " " * leading_spaces + f"\\text{{for }} {args} \\text{{:}}"
            except Exception:
                pass
            return " " * leading_spaces + f"\\text{{{header}}}"
        if header.startswith(("for ", "if ", "elif ", "else", "while ")):
            return " " * leading_spaces + f"\\text{{{header}}}"

    try:
        tree = ast.parse(stripped_line)
        node = tree.body[0]

        def process_dist_call(func_call_node):
            # numpyro.sample("mu", dist.Normal(0, 1), obs=y): the site name is
            # the first positional argument and the DISTRIBUTION is the second.
            # Rendering the wrapper itself gave "\text{Sample}(mu, Normal(...))".
            site_name = None
            if (_callee_str(func_call_node.func) in _NUMPYRO_SITE_FNS
                    and func_call_node.args
                    and isinstance(func_call_node.args[0], ast.Constant)
                    and isinstance(func_call_node.args[0].value, str)):
                site_name = func_call_node.args[0].value
                inner = next((a for a in func_call_node.args[1:]
                              if isinstance(a, ast.Call)), None)
                if inner is not None:
                    name, args_str, obs = process_dist_call(inner)
                    for kw in func_call_node.keywords:
                        if kw.arg == "obs":
                            obs = (format_latex_var(kw.value.id)
                                   if isinstance(kw.value, ast.Name)
                                   else ast_to_latex(kw.value))
                    return name, args_str, (obs or format_latex_var(site_name))

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
                              and _is_dist_callee(node.value.func))
        is_dist_expression = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                              and _is_dist_callee(node.value.func))

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

def _body_lines(code):
    """Source lines of the model function BODY, indentation preserved.

    The old ``[line.lstrip() for line in lines][1:]`` did two damaging things:
    lstrip() flattened `with numpyro.plate(...)` bodies into the top level so no
    block structure survived, and the [1:] assumed the `def` occupied exactly
    one line, silently eating a real statement for a decorated or multi-line
    signature. Locate the body with ast instead.
    """
    import textwrap
    code = textwrap.dedent(code)
    try:
        tree = ast.parse(code)
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except Exception:
        return [l.lstrip() for l in extract_lines(code)][1:]

    src_lines = code.split("\n")
    body_stmts = list(func.body)
    # Drop a leading docstring; it is prose, not a model equation.
    if (body_stmts and isinstance(body_stmts[0], ast.Expr)
            and isinstance(body_stmts[0].value, ast.Constant)
            and isinstance(body_stmts[0].value.value, str)):
        body_stmts = body_stmts[1:]
    if not body_stmts:
        return []

    start = body_stmts[0].lineno - 1          # 1-based -> 0-based
    end = max(getattr(s, "end_lineno", s.lineno) for s in body_stmts)
    body = src_lines[start:end]

    # Strip the body's common indent but keep RELATIVE indentation, so nested
    # blocks stay distinguishable.
    return [l.rstrip() for l in textwrap.dedent("\n".join(body)).split("\n")
            if l.strip() and not l.strip().startswith("#")]


def to_latex(model):
    code = inspect.getsource(model)
    lines_clean = _body_lines(code)
    latex_lines = [extract_latex_line_final(line) for line in lines_clean]
    latex_str = "\\begin{align*}\n"
    latex_str += " \\\\\n".join(filter(None, reversed(latex_lines)))
    latex_str += "\\\\\n"
    latex_str += "\\end{align*}\n"
    display(Latex(latex_str))
    return latex_str