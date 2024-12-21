import torch
import json
import argparse
from typing import Dict, Any, Optional
from torch.nn import Module

def describe_pth_model(file_path: str) -> Dict[str, Any]:
    try:
        data = torch.load(file_path, map_location="cpu")
    except Exception as e:
        return {"error": f"Failed to load .pth file: {str(e)}"}
    
    description = {
        "file_path": file_path,
        "keys": list(data.keys()) if isinstance(data, dict) else ["state_dict"],
        "details": {},
        "summary": {}
    }
    
    # Check for common keys and analyze their contents
    if isinstance(data, dict):
        if "state_dict" in data:
            description["details"]["state_dict"] = analyze_state_dict(data["state_dict"])
        else:
            description["details"]["state_dict"] = "Not found"

        if "optimizer" in data:
            description["details"]["optimizer"] = {
                "keys": list(data["optimizer"].keys()),
            }
        else:
            description["details"]["optimizer"] = "Not found"
        
        if "params_ema" in data:
            description["details"]["params_ema"] = analyze_state_dict(data["params_ema"])
        elif "params" in data:
            description["details"]["params"] = analyze_state_dict(data["params"])

        for key, value in data.items():
            if key not in ["state_dict", "optimizer", "params", "params_ema"]:
                if isinstance(value, (int, float, str)):
                    description["details"][key] = value
                elif isinstance(value, dict):
                    description["details"][key] = f"Dictionary with {len(value)} keys"
                else:
                    description["details"][key] = f"Type: {type(value).__name__}"

    else:
        description["details"]["model"] = describe_torch_model(data)
    
    description["summary"] = {
        "num_keys": len(description["keys"]),
        "contains_params_ema": "params_ema" in description["keys"],
        "contains_params": "params" in description["keys"],
        "contains_optimizer": "optimizer" in description["keys"]
    }
    
    return description

def analyze_state_dict(state_dict: Any) -> Dict[str, Any]:
    if not isinstance(state_dict, (dict, torch.nn.ModuleDict)):
        return {"error": "Invalid format. Expected a dictionary-like object."}
    
    details = {
        "num_parameters": 0,
        "parameter_shapes": {}
    }
    try:
        details["num_parameters"] = sum(p.numel() for p in state_dict.values())
        details["parameter_shapes"] = {k: tuple(v.shape) for k, v in state_dict.items()}
    except Exception as e:
        details["error"] = f"Failed to analyze state_dict: {str(e)}"
    return details

def describe_torch_model(model: Module) -> Dict[str, Any]:
    details = {
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "layer_names": [],
        "num_layers": 0,
        "module_hierarchy": []
    }

    for name, layer in model.named_modules():
        details["module_hierarchy"].append({"name": name, "type": type(layer).__name__})
        if len(list(layer.children())) == 0:  # Count only leaf layers
            details["layer_names"].append(name)
            details["num_layers"] += 1

    return details

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a PyTorch .pth model file and output its structure and metadata in JSON format."
    )
    parser.add_argument(
        "model_path", 
        type=str, 
        help="Path to the .pth model file to analyze."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Optional path to save the JSON output. If not specified, output is printed to the console."
    )
    
    args = parser.parse_args()
    
    model_description = describe_pth_model(args.model_path)
    
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(model_description, f, indent=4)
            print(f"Model description saved to {args.output}")
        except Exception as e:
            print(f"Failed to save output to {args.output}: {str(e)}")
    else:
        print(json.dumps(model_description, indent=4))

if __name__ == "__main__":
    main()
