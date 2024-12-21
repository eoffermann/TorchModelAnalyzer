# https://github.com/eoffermann/TorchModelAnalyzer
import torch
import json
import argparse
from typing import Dict, Any

def describe_pth_model(file_path: str) -> Dict[str, Any]:
    """
    Analyzes a .pth file and returns a JSON-friendly dictionary description.
    
    Args:
        file_path (str): Path to the .pth model file.
        
    Returns:
        Dict[str, Any]: A dictionary describing the model file.
    """
    try:
        data = torch.load(file_path, map_location="cpu")
    except Exception as e:
        return {"error": f"Failed to load .pth file: {str(e)}"}
    
    description = {
        "file_path": file_path,
        "keys": list(data.keys()) if isinstance(data, dict) else ["state_dict"],
        "details": {}
    }
    
    if isinstance(data, dict):
        if "state_dict" in data:
            description["details"]["state_dict"] = {
                "num_parameters": sum(p.numel() for p in data["state_dict"].values()),
                "parameter_shapes": {k: tuple(v.shape) for k, v in data["state_dict"].items()},
            }
        else:
            description["details"]["state_dict"] = "Not found"

        if "optimizer" in data:
            description["details"]["optimizer"] = {
                "keys": list(data["optimizer"].keys()),
            }
        else:
            description["details"]["optimizer"] = "Not found"

        for key, value in data.items():
            if key not in ["state_dict", "optimizer"]:
                description["details"][key] = type(value).__name__
    else:
        description["details"]["state_dict"] = {
            "num_parameters": sum(p.numel() for p in data.values()),
            "parameter_shapes": {k: tuple(v.shape) for k, v in data.items()},
        }
    
    return description

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
    
    # Analyze the model
    model_description = describe_pth_model(args.model_path)
    
    # Output the result
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
