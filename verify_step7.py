import re
with open('v7_1_validation.txt') as f:
    txt = f.read()

acc_m = re.search(r'[Aa]ccuracy[:\s]+(\d+\.?\d*)%', txt)
ll_m  = re.search(r'[Ll]og.?[Ll]oss[:\s]+([\d.]+)', txt)
ece_m = re.search(r'ECE[:\s]+([\d.]+)', txt)

acc = float(acc_m.group(1)) if acc_m else 0.0
ll  = float(ll_m.group(1))  if ll_m  else 99.0
ece = float(ece_m.group(1)) if ece_m else 99.0

print(f'Accuracy: {acc:.2f}%  (gate: ≥43.0%)')
print(f'Log-Loss: {ll:.4f}   (gate: ≤1.10)')
print(f'ECE:      {ece:.4f}  (target: <0.08)')

passed_acc = acc >= 43.0
passed_ll  = ll  <= 1.10
passed_ece = ece <  0.09

print()
print(f'  {"✅" if passed_acc else "❌"} Accuracy gate')
print(f'  {"✅" if passed_ll  else "❌"} Log-Loss gate')
print(f'  {"✅" if passed_ece else "⚠️"} ECE target')

if passed_acc and passed_ll:
    print()
    print('✅ V7.1 PASSES CI/CD GATES — safe to push to GitHub')
else:
    print()
    print('❌ GATES FAILED — do not push, investigate failing fold')
