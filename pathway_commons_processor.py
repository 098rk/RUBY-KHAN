import pandas as pd
import urllib.request
import gzip
import os

class PathwayDataLoader:
    """
    A class to download and process PathwayCommons data.
    """
    def __init__(self, filename="PathwayCommons.csv", 
                 url="http://www.pathwaycommons.org/archives/PC2/v12/PathwayCommons12.All.BINARY_SIF.gz"):
        self.filename = filename
        self.url = url
        self.output_file = "PathwayCommons.gz"

    def download_data(self):
        """Downloads the PathwayCommons data file."""
        print("Downloading data...")
        urllib.request.urlretrieve(self.url, self.output_file)
        print("Download complete.")

    def process_data(self):
        """Extracts and processes the downloaded data into a structured format."""
        print("Processing data...")
        
        with gzip.open(self.output_file, 'rt') as f:
            data = [line.strip().split("\t") for line in f.readlines()]
        
        df = pd.DataFrame(data, columns=[
            "Participant A", "Interaction Type", "Participant B", 
            "Source", "PubMed ID", "Pathway Names"
        ])

        # Filter relevant interaction types
        interaction_types = {"controls-state-change-of", "controls-phosphorylation-of", "controls-expression-of"}
        df_filtered = df[df["Interaction Type"].isin(interaction_types)]

        # Select only necessary columns
        df_filtered = df_filtered[["Participant A", "Participant B", "Interaction Type"]]

        # Save to CSV
        df_filtered.to_csv(self.filename, index=False)
        print(f"Data saved to {self.filename}")

    def run(self):
        """Executes the full pipeline: downloading and processing."""
        self.download_data()
        self.process_data()
        print("Process completed successfully.")
        
if __name__ == "__main__":
    loader = PathwayDataLoader(filename="PathwayCommons.csv")
    loader.run()
