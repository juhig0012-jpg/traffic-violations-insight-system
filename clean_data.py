import pandas as pd
import numpy as np
print("Cleaning Traffic Violations data...")
df = pd.read_csv('raw_traffic.csv')
# Add all cleaning steps from my Step 2 response
df.to_csv('cleaned_traffic.csv', index=False)
print("✅ Cleaning complete!")
