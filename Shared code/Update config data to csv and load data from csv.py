import csv
import time
import os

# Example data
Config_data = [
    [1, 1, "DY08P1S1", 20, 30, 300, 200],
    [1, 2, "DY08P1S2", 320, 220, 300, 200],
    # Add more data as needed
]

# File name where the data will be saved
csv_file_name = 'config_data.csv'

def save_to_csv(data, file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Main box', 'Sub box', 'Model', 'x', 'y', 'w', 'h'])  # Header
        writer.writerows(data)

def load_from_csv(file_name):
    if not os.path.exists(file_name):
        return []
    with open(file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        data = []
        for row in reader:
            data.append([int(row[0]), int(row[1]), row[2], int(row[3]), int(row[4]), int(row[5]), int(row[6])])
        return data

def create_initial_csv_if_not_exists(data, file_name):
    if not os.path.exists(file_name):
        save_to_csv(data, file_name)

def main():
    global Config_data

    # Create the initial CSV file if it does not exist
    create_initial_csv_if_not_exists(Config_data, csv_file_name)

    # Load data from CSV file if it exists
    Config_data = load_from_csv(csv_file_name)

    # Simulate data changes and periodic saving
    try:
        while True:
            # Simulate data update (replace with your actual data update logic)
            print("Current Config_data:", Config_data)
            if Config_data:
                Config_data[0][3] += 1  # Example of changing 'x' value of first entry
                Config_data[0][4] += 1  # Example of changing 'y' value of first entry

            # Save the current state to the CSV file
            save_to_csv(Config_data, csv_file_name)

            # Wait for some time before the next update
            time.sleep(5)  # Save every 5 seconds (adjust as needed)

    except KeyboardInterrupt:
        print("Program interrupted. Saving data before exit...")
        save_to_csv(Config_data, csv_file_name)
        print("Data saved. Exiting program.")

if __name__ == "__main__":
    main()
