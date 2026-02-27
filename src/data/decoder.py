import json
import os

def load_reverse_mapping():
    """Loads encoder and flips it to { "ID_String": "Category_Name" }."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_path = os.path.abspath(os.path.join(base_dir, '../../json_files/encoder/encoder.json'))
    
    with open(encoder_path, 'r') as f:
        # Original: {"Bib-Shorts": 0} -> Target: {"0": "Bib-Shorts"}
        original_map = json.load(f)
        return {str(v): k for k, v in original_map.items()}

def decode_json(input_data):
    """Replaces numeric IDs with Category Names using reversed mapping."""
    mapping = load_reverse_mapping()
    decoded_output = {}

    for key, value in input_data.items():
        # Case 1: Objects with nested 'item_id'
        if isinstance(value, dict) and 'item_id' in value:
            item_id = str(value['item_id'])
            item_name = mapping.get(item_id, f"Unknown_ID_{item_id}")
            
            new_entry = value.copy()
            new_entry['item_name'] = item_name
            decoded_output[item_name] = new_entry

        # Case 2: ID as the primary key
        else:
            item_name = mapping.get(str(key), f"Unknown_ID_{key}")
            decoded_output[item_name] = value

    return decoded_output

def run_decoder(input_file_path, output_file_path):
    """Processes the file and saves the decoded version."""
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    decoded_data = decode_json(data)
    
    with open(output_file_path, 'w') as f:
        json.dump(decoded_data, f, indent=4)
    
    return decoded_data