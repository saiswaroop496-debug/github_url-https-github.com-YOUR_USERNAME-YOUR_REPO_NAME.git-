import json
from inference import run_inference
try:
    res = run_inference("Argentina", "France", 1.0, "group", None, None, None)
    print("SUCCESS")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
