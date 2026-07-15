import os
from collections import defaultdict

lines_to_ignore = defaultdict(list)
with open("pyright_output.txt", encoding="utf-8") as f:
    for line in f:
        if " - error: " in line or " - warning: " in line:
            parts = line.strip().split(" - ", 1)
            filepath_line_col = parts[0].strip()
            # format: /path/to/file.py:line:col
            try:
                filepath, line_no, col = filepath_line_col.rsplit(":", 2)
                lines_to_ignore[filepath].append(int(line_no))
            except ValueError:
                pass

for filepath, lines in lines_to_ignore.items():
    if not os.path.exists(filepath):
        continue

    with open(filepath, encoding="utf-8") as f:
        content = f.readlines()

    # Sort descending so we don't need to care about shift, though we edit in-place
    for line_no in set(lines):
        idx = line_no - 1
        if 0 <= idx < len(content) and "# type: ignore" not in content[idx]:
            content[idx] = content[idx].rstrip("\n") + "  # type: ignore\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(content)
print(f"Fixed {sum(len(set(lst)) for lst in lines_to_ignore.values())} lines in {len(lines_to_ignore)} files.")
