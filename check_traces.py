import csv
import glob
unique_trace_SN = set()
unique_service_name_SN = set()
unique_trace_TT = set()
unique_service_name_TT = set()

with open(
    "../parsed_output/SN_Dataset/trace/SN.2022-04-17T183729D2022-04-17T190100_trace.csv",
    "r",
) as file:
    reader = csv.DictReader(file)
    for row in reader:
        # print(row)  # Each row is a dictionary with column names as keys
        unique_trace_SN.add(row["trace_id"])
        unique_service_name_SN.add(row["service_name"])


print(f"Number of unique trace_ids in SN dataset: {len(unique_trace_SN)}")
print(f"Number of unique service_ids in SN dataset: {len(unique_service_name_SN)}")


# files = glob.glob("../parsed_output/TT_Dataset/trace/*.csv")
# for i,filename in enumerate(files):
#     print(f"FILE {i+1}: {filename}")
# filename = "../parsed_ouput/TT_Dataset/trace/TT.2022-04-18T121515D2022-04-18T140256_trace.csv"
with open("../parsed_output/TT_Dataset/trace/TT.2022-04-21T153246D2022-04-21T174753_trace.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        # print(row)  # Each row is a dictionary with column names as keys
        unique_trace_TT.add(row["trace_id"])
        unique_service_name_TT.add(row["service_name"])

print(f"Number of unique trace_ids in TT dataset: {len(unique_trace_TT)}")
print(f"Number of unique trace_ids in TT dataset: {len(unique_service_name_TT)}")


# ------- TOTAL ------

total_traces = len(unique_trace_TT) + len(unique_trace_SN)
print(f"Total traces: {total_traces}")
print(f"Percentage of SN: {len(unique_trace_SN) / total_traces:.2%}")
print(f"Percentage of TT: {len(unique_trace_TT) / total_traces:.2%}")