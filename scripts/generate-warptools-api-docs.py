import os
import subprocess
from pathlib import Path

def generate_docs(command, subcommands, output_file):
    output_file = Path(output_file)
    output_dir = output_file.parent
    output_dir.mkdir(exist_ok=True, parents=True)
       
    output_file.unlink(missing_ok=True)
    
    for subcommand in subcommands:
        result = subprocess.run([f'{command}', f'{subcommand}', '--help'], capture_output=True, text=True)
        help_text = result.stdout

        with open(output_file, 'a') as f:
            f.write(f"## {subcommand}\n\n")
            f.write("```\n")
            f.write(help_text)
            f.write("```\n\n\n")


cli_programs_general = [
    "create_settings", 
    "move_data",
    "filter_quality",
    "change_selection",
    "threshold_picks",
    "helpgpt",
]

cli_programs_frameseries = [
    "fs_motion_and_ctf",
    "fs_motion",
    "fs_ctf",
    "fs_boxnet_infer",
    "fs_boxnet_train",
    "fs_export_micrographs",
    "fs_export_particles",
]

cli_programs_tiltseries = [
    "ts_import",
    "ts_stack",
    "ts_aretomo",
    "ts_etomo_fiducials",
    "ts_etomo_patches",
    "ts_import_alignments",
    "ts_defocus_hand",
    "ts_ctf",
    "ts_reconstruct",
    "ts_template_match",
    "ts_export_particles",
    "ts_eval_model"
]

generate_docs('WarpTools', cli_programs_general, "docs/reference/warptools/api/general.md")
generate_docs('WarpTools', cli_programs_frameseries, "docs/reference/warptools/api/frame_series.md")
generate_docs('WarpTools', cli_programs_tiltseries, "docs/reference/warptools/api/tilt_series.md")
