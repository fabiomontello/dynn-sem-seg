import argparse
from typing import Any, Dict

from omegaconf import OmegaConf
from prettytable import PrettyTable


def print_omegaconf(cfg):
    """
    Print an OmegaConf configuration in a table format.

    :param cfg: OmegaConf configuration object.
    """
    # Flatten the OmegaConf configuration to a dictionary
    flat_config = OmegaConf.to_container(cfg, resolve=True)

    # Create a table with PrettyTable
    table = PrettyTable()

    # Define the column names
    table.field_names = ["Key", "Value"]

    # Recursively go through the items and add rows
    def add_items(items, parent_key=""):
        for k, v in items.items():
            current_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                # If the value is another dict, recursively add its items
                add_items(v, parent_key=current_key)
            else:
                # If it's a leaf node, add it to the table
                table.add_row([current_key, v])

    # Start adding items from the top-level configuration
    add_items(flat_config)

    # Print the table
    print(table)


def load_config_from_path_arg(print_config: bool = False) -> Dict[str, Any]:
    """
    Load and process a configuration file for training a Sensor MAE model.

    This function parses command-line arguments to retrieve the path to a YAML
    configuration file and other optional settings. It loads the configuration,
    merges it with a base configuration (if specified), and integrates
    command-line arguments. Additional processing is performed to ensure the
    configuration is complete and consistent.

    Args:
        print_config (bool): Whether to print the final configuration to stdout.

    Returns:
        Dict[str, Any]: A dictionary representing the merged configuration.
    """
    parser = argparse.ArgumentParser(description="Train a Sensor MAE model")

    # Define command-line arguments
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument("--load_from", type=str, help="Checkpoint file path to load")

    # Parse command-line arguments
    args = parser.parse_args()

    # Load the primary configuration from the specified YAML file
    config = OmegaConf.load(args.config)

    # If a base configuration is specified, merge it with the primary configuration
    if config.get("base_config", None):
        print("Loading base config from " + config.base_config)
        base_config = OmegaConf.load(config.base_config)
        config = OmegaConf.merge(base_config, config)

    # Convert parsed command-line arguments into a dictionary
    cli_args = {k: v for k, v in vars(args).items()}

    # Merge command-line arguments into the configuration
    config = OmegaConf.merge(config, cli_args)

    # Optionally print the final configuration
    if print_config:
        print_omegaconf(config)

    return config
