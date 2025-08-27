import requests
import csv
import io

url = "https://iv.ipie.me/ivr_all"
# Disable SSL verification to handle certificate issues
response = requests.get(url, verify=False)
data = response.text

print(f"Response status code: {response.status_code}")
print(f"Data length: {len(data)}")
print("="*50)

# The data is already in a structured text format, not HTML
# Let's parse it directly
lines = data.strip().split('\n')

# Find the header line (starts with "Ticker")
header_line = None
data_start = 0

for i, line in enumerate(lines):
    if line.strip().startswith('Ticker'):
        header_line = line
        data_start = i + 1
        break

if header_line:
    print("Found header:", header_line)
    print("="*50)
    
    # Parse the data into CSV format
    csv_output = []
    csv_output.append(header_line)
    
    # Add data rows
    for line in lines[data_start:]:
        line = line.strip()
        if line and not line.startswith('Updated') and not line.startswith('Total'):
            csv_output.append(line)
    
    # Print the CSV data
    print("Extracted CSV data:")
    for row in csv_output[:10]:  # Show first 10 rows
        print(row)
    
    if len(csv_output) > 10:
        print(f"... and {len(csv_output) - 10} more rows")
    
    # Save to file
    with open('iv_data.csv', 'w', newline='', encoding='utf-8') as f:
        for row in csv_output:
            f.write(row + '\n')
    
    print(f"\nData saved to 'iv_data.csv' with {len(csv_output)} rows")
else:
    print("Could not find header line in the data")
    print("Raw data preview:")
    print(data[:1000])
