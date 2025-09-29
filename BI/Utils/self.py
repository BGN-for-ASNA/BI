import re
import os

def wrap_functions_in_class(input_file_path, output_file_path, class_name, indent_spaces=4):
    """
    Reads a Python file of functions, wraps them in a class, adds 'self'
    to their definitions, and indents the entire content.

    Args:
        input_file_path (str): The path to the source Python file.
        output_file_path (str): The path where the modified class file will be saved.
        class_name (str): The name for the new class.
        indent_spaces (int): The number of spaces to use for one level of indentation.
    """
    # Regex to find function definitions, same as before.
    # It correctly handles existing indentation on the 'def' line.
    pattern = re.compile(r"^\s*def\s+(\w+)\s*\((.*)\):$")
    indent = ' ' * indent_spaces

    # The replacer function to add 'self'.
    def add_self_to_def(match):
        # We preserve any original indentation from the matched line.
        # Although the whole line will be re-indented later.
        function_name = match.group(1)
        original_args = match.group(2).strip()

        if not original_args:
            return f"def {function_name}(self):"
        else:
            return f"def {function_name}(self, {original_args}):"

    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            # 1. Write the class definition line at the top of the new file.
            outfile.write(f"class {class_name}:\n")

            # 2. Process every line from the source file.
            for line in infile:
                # Remove trailing newline/whitespace to work with a clean line.
                original_line = line.rstrip()

                # If the line is blank, just write a blank line. No indentation needed.
                if not original_line.strip():
                    outfile.write('\n')
                    continue

                # First, check if the line is a function definition and add 'self'.
                modified_line = pattern.sub(add_self_to_def, original_line)

                # Then, add one level of indentation to the (potentially modified) line.
                indented_line = indent + modified_line

                # Write the final, indented line to the output file.
                outfile.write(indented_line + '\n')

        print(f"✅ Success! Functions from '{input_file_path}' are now methods in class '{class_name}'")
        print(f"   -> Saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


# --- HOW TO USE ---

if __name__ == '__main__':
    # --- CONFIGURE YOUR SCRIPT HERE ---

    # 1. The name of the file containing your functions.
    input_filename = 'np_dists.py'  # <--- CHANGE THIS

    # 2. The desired name for your new class.
    new_class_name = 'UnifiedDist'     # <--- CHANGE THIS

    # 3. The name for the new file that will be created.
    output_filename = 'np_dists2.py' # <--- CHANGE THIS (or leave as is)

    # ------------------------------------

    if os.path.exists(input_filename):
        wrap_functions_in_class(input_filename, output_filename, new_class_name)
    else:
        print(f"Error: Input file '{input_filename}' not found.")
        print("Please update the 'input_filename' variable in the script.")