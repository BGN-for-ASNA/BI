import re

def count_words(text):
    # 1. Remove lstlisting environments
    # We use \\\\ to match a literal backslash in regex
    text = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', '', text, flags=re.DOTALL)
    
    # 2. Remove citations \cite{...}, \citep{...}, \citet{...}
    text = re.sub(r'\\cite[pt]?\{.*?\}', '', text)
    
    # 3. Remove LaTeX commands but keep their arguments (like section titles)
    # Remove \command
    text = re.sub(r'\\\w+', '', text)
    
    # 4. Remove comments
    text = re.sub(r'%.*', '', text)
    
    # 5. Remove braces and other LaTeX special chars
    text = re.sub(r'[{}]', '', text)
    
    # 6. Count words
    words = re.findall(r'\w+', text)
    return len(words)

with open('Manuscript/Manuscript.tex', 'r') as f:
    full_content = f.read()

# Extract Abstract
abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', full_content, re.DOTALL)
abstract_text = abstract_match.group(1) if abstract_match else ''

# Extract Main Text (from Introduction to end of Discussion)
intro_start = full_content.find('\\section{Introduction}')
data_avail_start = full_content.find('\\section*{DATA AVAILABILITY STATEMENT}')

if intro_start != -1 and data_avail_start != -1:
    main_text = full_content[intro_start:data_avail_start]
else:
    # Fallback if section names changed
    main_text = ''

print(f'Abstract word count (refined): {count_words(abstract_text)}')
print(f'Main text word count (refined): {count_words(main_text)}')
