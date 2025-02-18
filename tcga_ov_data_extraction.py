import requests
import json
import pandas as pd
import os

# Define the base URL for the GDC API
GDC_API = "https://api.gdc.cancer.gov/"

# Step 1: Define the query for TCGA OV data
def query_gdc(project="TCGA-OV", data_type="Gene Expression Quantification", workflow_type="HTSeq - Counts"):
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": [data_type]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": [workflow_type]
                }
            }
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "format": "json",
        "size": "1000",  # Limit results to 1000 entries
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id,data_type"
    }

    response = requests.get(GDC_API + "files", params=params)
    
    if response.status_code == 200:
        print("Query successful!")
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Step 2: Parse and save the metadata
def save_metadata(metadata, output_filename="gdc_tcga_ov_metadata.csv"):
    data = [
        {
            "file_id": file_entry["file_id"],
            "file_name": file_entry["file_name"],
            "case_id": file_entry["cases"][0]["case_id"],
            "submitter_id": file_entry["cases"][0]["submitter_id"],
            "data_type": file_entry["data_type"]
        }
        for file_entry in metadata["data"]["hits"]
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(output_filename, index=False)
    print(f"Metadata saved to {output_filename}")
    return df

# Step 3: Download the files based on file IDs
def download_files(file_ids, download_dir="gdc_tcga_ov_data"):
    os.makedirs(download_dir, exist_ok=True)
    
    for file_id in file_ids:
        download_url = f"{GDC_API}data/{file_id}"
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(download_dir, f"{file_id}.tar.gz")
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {file_id} to {file_path}")
        else:
            print(f"Failed to download file {file_id}")

# Main function to execute the steps
def main():
    metadata = query_gdc()
    if metadata:
        df_metadata = save_metadata(metadata)
        file_ids = df_metadata["file_id"].tolist()
        download_files(file_ids)

if __name__ == "__main__":
    main()
