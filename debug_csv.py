# it's give CSV's column names! 

import pandas as pd
csv_file = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
df = pd.read_csv(csv_file)
print("Columns:", list(df.columns))
print("\nLast column sample:", df.iloc[:, -1].unique()[:10])
print("\nShape:", df.shape)
print("\nHead:\n", df.head(2))