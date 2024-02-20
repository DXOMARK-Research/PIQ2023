import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import assets.assets_23_PIQ as my_asset

def main():
    
    my_asset.print_hello()

    # Load the images.csv file
    images_df = pd.read_csv(r'./images.csv')

    # Add a 'SCORE' column with a default value of 0.0
    images_df['SCORE'] = 0.0

    # Save the modified DataFrame to result_23_PIQ.csv in the results folder
    images_df.to_csv('./results/result_23_PIQ.csv', index=False, sep=',')

if __name__ == "__main__":
    main()