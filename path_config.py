import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Path Configuration Utility')
    parser.add_argument('--data-path', type=str, help='Path to data directory', required=True)
    parser.add_argument('--output-path', type=str, help='Path to output directory', required=True)
    return parser.parse_args()

def get_data_path():
    args = parse_arguments()
    return args.data_path

def get_output_path():
    args = parse_arguments()
    return args.output_path
