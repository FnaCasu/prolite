import configparser # to modify ini files
import json
import numpy as np
import ast

def write_ini(file_out, param_dic, file_in=None):
    # Import parameters form Generic Configuration
    config_parser = configparser.ConfigParser()
    config_parser.optionxform = str # Keep the original case of parameter keys
    if not(file_in is None):
        # TBD: check if the file exist
# TODO: this not raise an error if the file does not exist
        config_parser.read(file_in)  
        # Add parameters from the header to generic parameter
        for (sec, opt, val) in param_dic:
            config_parser.set(sec, opt, _format_out_ini(val))
    else:
        for (sec, opt, val) in param_dic:
            # Create the section if not exits
            if sec not in config_parser:
                config_parser[sec] = {}
            # Add data with correct format
            config_parser[sec][opt] = _format_out_ini(val)

    # Write updated config to temporary .ini file
    with open(file_out, "w") as configfile:
      config_parser.write(configfile)

def _format_out_ini(val):
    if isinstance(val, np.ndarray):
        return "[" + ", ".join(map(str, val.tolist())) + "]"

    elif isinstance(val, (list, tuple)):
        return "[" + ", ".join(map(str, val)) + "]"

    else:
        return str(val)


def _format_in_ini(val):
    # Remove space at start and end of the string
    val = val.strip()
    # Try to parse Python literal (e.g. "[1, 2]", "3.14", "True")
    try:
        parsed = ast.literal_eval(val)
        # If the parsed value is a list or tuple convert to np.array
        if isinstance(parsed, (list, tuple)):
            return np.array(parsed)
        return parsed
    except Exception:
        # Not a literal, return raw string
        return val


def read_ini(file_in):
## TODO: CONTROLLARE CHE IL FILE ESISTA PRIMA
    # Create the Parsser
    config_parser = configparser.ConfigParser()
    config_parser.optionxform = str # Keep the original case of parameter keys
    # Read file
    config_parser.read(file_in)

    # Define an empty list
    data = []
    # Unpack and cast all elements
    for sec in config_parser.sections():
        for key, raw_val in config_parser[sec].items():
            data.append((sec, key, _format_in_ini(raw_val)))
    # Return the list

    parsed = {}
    for sec in config_parser.sections():
        parsed[sec] = {}
        for key, raw_val in config_parser.items(sec):
            parsed[sec][key] = _format_in_ini(raw_val)
    
    return parsed




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def write_json(file_out, data):
    # Write file
    with open(file_out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4,  cls=NumpyEncoder)

def read_json(file_in):
    with open(file_in) as f:
        data = json.load(f)
    return data
