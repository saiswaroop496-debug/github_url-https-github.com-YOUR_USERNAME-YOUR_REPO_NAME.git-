with open('train_test.py') as f:
    src = f.read()
with open('inference.py') as f:
    inf = f.read()

glicko_in_train = 'home_glicko' in src and 'compute_glicko' in src
glicko_in_inf   = 'home_glicko' in inf or 'glicko' in inf
kalman_in_train = 'kalman' in src.lower()
kalman_in_inf   = 'kalman' in inf.lower()

print(f'Glicko kept in train_test.py:  {"✅" if glicko_in_train else "❌"}')
print(f'Glicko kept in inference.py:   {"✅" if glicko_in_inf else "❌"}')
print(f'Kalman added in train_test.py: {"✅" if kalman_in_train else "❌"}')
print(f'Kalman added in inference.py:  {"✅" if kalman_in_inf else "❌"}')

if not glicko_in_train:
    print('⚠️  WARNING: Glicko was replaced entirely — this will hurt accuracy in early folds')
