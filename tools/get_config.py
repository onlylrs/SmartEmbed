#!/usr/bin/env python3
"""
Configuration helper for shell scripts.
Reads unified configuration and outputs shell variables.

Usage in shell scripts:
  eval $(python tools/get_config.py --section runtime)
  echo "GPUs: $DEFAULT_GPUS"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from jina.utils.config_manager import load_config


def main():
    parser = argparse.ArgumentParser(description="Get configuration values for shell scripts")
    parser.add_argument("--section", required=True, help="Configuration section to output")
    parser.add_argument("--format", choices=['shell', 'json'], default='shell', help="Output format")
    
    args = parser.parse_args()
    
    try:
        config = load_config()
        
        # Handle special case for slurm - it's under runtime section
        if args.section == 'slurm':
            if 'runtime' not in config:
                print(f"Error: Runtime section not found in configuration", file=sys.stderr)
                sys.exit(1)
            
            runtime_config = config['runtime']
            # Export only slurm-related keys
            for key, value in runtime_config.items():
                if key.startswith('slurm_'):
                    if isinstance(value, (str, int, float, bool)):
                        shell_key = key.upper()
                        if isinstance(value, bool):
                            shell_value = "true" if value else "false"
                        else:
                            shell_value = str(value)
                        print(f'export {shell_key}="{shell_value}"')
        else:
            if args.section not in config:
                print(f"Error: Section '{args.section}' not found in configuration", file=sys.stderr)
                sys.exit(1)
            
            section_config = config[args.section]
            
            if args.format == 'shell':
                # Output shell variable assignments
                for key, value in section_config.items():
                    if isinstance(value, (str, int, float, bool)):
                        shell_key = key.upper()
                        if isinstance(value, bool):
                            shell_value = "true" if value else "false"
                        else:
                            shell_value = str(value)
                        print(f'export {shell_key}="{shell_value}"')
                    elif isinstance(value, list):
                        shell_key = key.upper()
                        shell_value = ",".join(str(item) for item in value)
                        print(f'export {shell_key}="{shell_value}"')
            
            elif args.format == 'json':
                import json
                print(json.dumps(section_config, indent=2))
    
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
