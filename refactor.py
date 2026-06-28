import os

with open('train_test.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []

# Find line ranges
# execution block 1: line index 119 to 452
# function definitions: line index 453 to 525
# execution block 2 (export): line index 526 to end

# Write all header and class/function defs (0 to 118)
for i in range(119):
    new_lines.append(lines[i])

new_lines.append("\ndef main():\n")

# Write execution block 1, indented
for i in range(119, 453):
    line = lines[i]
    if line.strip() == "":
        new_lines.append(line)
    else:
        new_lines.append("    " + line)

new_lines.append("\n    # --- Export logic ---\n")
# Write execution block 2, indented
for i in range(526, len(lines)):
    line = lines[i]
    if line.strip() == "":
        new_lines.append(line)
    else:
        new_lines.append("    " + line)

new_lines.append("\n# --- Function Definitions ---\n")
# Write function definitions, NOT indented
for i in range(453, 526):
    new_lines.append(lines[i])

new_lines.append("\nif __name__ == '__main__':\n    main()\n")

with open('train_test_fixed.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Done!")
