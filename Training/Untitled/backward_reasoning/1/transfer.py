import pyarrow.parquet as pq

table = pq.read_table('train.parquet')
df = table.to_pandas()

df.to_csv("train.csv",index=False)
