import re, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('face_recognition_module.py', 'rb') as f:
    raw = f.read().decode('utf-8', errors='replace')
positions = [m.start() for m in re.finditer('\"\"\"', raw)]
lines_text = raw.split('\n')
def pos_to_line(pos, text):
    return text[:pos].count('\n') + 1
in_string = False
for i, p in enumerate(positions):
    ln = pos_to_line(p, raw)
    state = "CLOSE" if in_string else "OPEN"
    in_string = not in_string
    if 260 <= ln <= 285:
        print(f'Triple-quote #{i+1} at line {ln}: {state}')
print('Final in_string state:', in_string)
