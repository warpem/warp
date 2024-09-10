import os
import subprocess
from pathlib import Path

def generate_docs(command, subcommands, output_file):
    output_file = Path(output_file)
    output_dir = output_file.parent
    output_dir.mkdir(exist_ok=True, parents=True)
       
    output_file.unlink(missing_ok=True)
    if subcommands is None:
        result = subprocess.run([f'{command}', '--help'], capture_output=True, text=True)
        help_text = result.stdout
        
        with open(output_file, 'a') as f:
            f.write(f"## {command}\n\n")
            f.write("```\n")
            f.write(help_text)
            f.write("```\n\n\n")
        
    for subcommand in subcommands:
        result = subprocess.run([f'{command}', f'{subcommand}', '--help'], capture_output=True, text=True)
        help_text = result.stdout

        with open(output_file, 'a') as f:
            f.write(f"## {subcommand}\n\n")
            f.write("```\n")
            f.write(help_text)
            f.write("```\n\n\n")


cli_programs_mtools = [
    "create_population",
    "create_source",
    "create_species",
    "rotate_species",
    "shift_species",
    "expand_symmetry",
    "resample_trajectories",
    "update_mask",
    "list_species",
    "list_sources",
    "add_source",
    "remove_species",
    "remove_source",
]

generate_docs('MTools', subcommands=cli_programs_mtools, output_file="docs/reference/mtools/api/mtools.md")
generate_docs('MCore', subcommands=None, output_file="docs/reference/mtools/api/mcore.md")