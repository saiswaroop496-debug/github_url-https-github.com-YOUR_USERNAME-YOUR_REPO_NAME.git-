# Read train_test.py and check the Kalman iteration order
with open('train_test.py') as f:
    src = f.read()

# The freeze guard: store MUST appear before update in the source
freeze_idx  = src.find('kalman_system.get_strength')   # or .get_velocity or .get_or_init
if freeze_idx == -1:
    freeze_idx = src.find('kalman_system.get_velocity')
if freeze_idx == -1:
    freeze_idx = src.find('kalman_system.get_or_init')
update_idx  = src.find('kalman_system.update_match')

if freeze_idx == -1:
    print('❌ FAIL: kalman_system.get_strength() not found in train_test.py')
elif update_idx == -1:
    print('❌ FAIL: kalman_system.update_match() not found in train_test.py')
elif freeze_idx < update_idx:
    print('✅ PASS: Kalman freeze BEFORE update — no leakage')
else:
    print('❌ CRITICAL LEAKAGE: update_match() called BEFORE storing state')
    print(f'   get_strength at char {freeze_idx}, update_match at char {update_idx}')
