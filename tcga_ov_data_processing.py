import pandas as pd
import requests
import io

# Function to download and process TCGA-OV WGS and Clinical data
def DownloadAndProcessTCGA_OVData():
    # Request TCGA-OV WGS data
    endpoint = "https://api.gdc.cancer.gov/files"
    filters = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-OV"]}},
                {"op": "in", "content": {"field": "files.data_format", "value": ["BAM"]}},
                {"op": "in", "content": {"field": "files.experimental_strategy", "value": ["WGS"]}}
            ]
        },
        "format": "TSV",
        "fields": "file_id,file_name,data_category,data_type,md5sum,file_size",
        "size": "1000"
    }

    response = requests.post(endpoint, json=filters, headers={"Content-Type": "application/json"})
    df_wgs = pd.read_csv(io.StringIO(response.text), sep="\t")

    # Rename columns for WGS data
    df_wgs.columns = ["File_ID", "File_Name", "Data_Category", "Data_Type", "MD5Sum", "File_Size"]

    # Filter for WGS data (Aligned Reads)
    df_wgs = df_wgs[df_wgs["Data_Type"] == "Aligned Reads"]
    df_wgs.to_csv("TCGA-OV_WGS_files.csv", index=False)

    # Download each WGS file
    for _, row in df_wgs.iterrows():
        file_url = f"https://api.gdc.cancer.gov/data/{row['File_ID']}"
        file_response = requests.get(file_url)
        with open(row["File_Name"], "wb") as f:
            f.write(file_response.content)
        print(f"Downloaded: {row['File_Name']}")

    # Request Clinical Data
    endpoint_clinical = "https://api.gdc.cancer.gov/clinical"
    filters_clinical = {
        "filters": {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-OV"]}},
        "format": "TSV",
        "fields": "case_id,patient_id,diagnosis,stage,age_at_diagnosis",
        "size": "1000"
    }

    response_clinical = requests.post(endpoint_clinical, json=filters_clinical, headers={"Content-Type": "application/json"})
    df_clinical = pd.read_csv(io.StringIO(response_clinical.text), sep="\t")

    # Rename columns for clinical data
    df_clinical.columns = ["Case_ID", "Patient_ID", "Diagnosis", "Stage", "Age_at_Diagnosis"]

    # Save clinical data
    df_clinical.to_csv("TCGA-OV_clinical_data.csv", index=False)

    return "TCGA-OV WGS and Clinical data downloaded and processed successfully."

# Main execution
if __name__ == "__main__":
    print(DownloadAndProcessTCGA_OVData())
