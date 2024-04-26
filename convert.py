import json

def convert_jsonl_to_json(jsonl_file, json_file):
    # Open the JSONL file for reading and JSON file for writing
    with open(jsonl_file, 'r', encoding='utf-8') as infile, open(json_file, 'w', encoding='utf-8') as outfile:

        # Initialize an empty list to store JSON objects
        json_list = []
        
        # Read each line in the JSONL file
        for line in infile:
            # Parse the JSON object from the line
            json_obj = json.loads(line)
            
            # Add the parsed JSON object to the list
            json_list.append(json_obj)
        
        # Write the list of JSON objects to the JSON file
        json.dump(json_list, outfile, indent=4)

# Example usage
convert_jsonl_to_json('train_phi.jsonl', 'intents.json')
