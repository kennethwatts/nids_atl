import csv
import sys

def prepend_header(header_file, data_file, output_file):
    """
    Prepends the header from header_file to data_file and writes the result to output_file.

    Args:
        header_file (str): Path to the CSV file containing the header.
        data_file (str): Path to the CSV file containing the raw data.
        output_file (str): Path to the output CSV file.
    """
    try:
        with open(header_file, 'r', newline='', encoding='utf-8') as hf, \
             open(data_file, 'r', newline='', encoding='utf-8') as df, \
             open(output_file, 'w', newline='', encoding='utf-8') as of:

            header_reader = csv.reader(hf)
            data_reader = csv.reader(df)
            output_writer = csv.writer(of)

            # Read and write the header
            for row in header_reader:
                output_writer.writerow(row)

            # Read and write the raw data
            for row in data_reader:
                output_writer.writerow(row)

        print(f"Header from '{header_file}' prepended to '{data_file}' and written to '{output_file}'.")

    except FileNotFoundError:
        print("Error: One or more input files not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python prepend_header.py <header_file.csv> <data_file.csv> <output_file.csv>")
    else:
        header_file = sys.argv[1]
        data_file = sys.argv[2]
        output_file = sys.argv[3]
        prepend_header(header_file, data_file, output_file)
