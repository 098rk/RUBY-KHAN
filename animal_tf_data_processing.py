import pandas as pd
import urllib.request
import gzip

# Function to download and process AnimalTF data
def DownloadAndProcessAnimalTFData(filename="AnimalTFData.csv"):
    url = "https://example.com/AnimalTFData.gz"
    output_file = "AnimalTFData.gz"

    # Downloading the file
    urllib.request.urlretrieve(url, output_file)

    # Extract and process data
    with gzip.open(output_file, 'rt') as f:
        data = [line.strip().split("\t") for line in f.readlines()]

    # Convert data into a pandas DataFrame with correct column names
    df = pd.DataFrame(data, columns=["Species", "Symbol", "Ensembl", "Family", "Protein", "Entrez_ID"])

    # Save to CSV
    df.to_csv(filename, index=False)

    return "AnimalTF data downloaded and processed successfully."

# Main execution
if __name__ == "__main__":
    print(DownloadAndProcessAnimalTFData(filename="D://ProjectFiles//AnimalTFData.csv"))
