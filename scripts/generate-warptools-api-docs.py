import os
import subprocess
from pathlib import Path

def generate_docs(cli_programs, output_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
       
    with open(output_file, 'w+') as f:
        f.write(f'# {Path(output_file).stem}\n')

    for program in cli_programs:
        result = subprocess.run(['WarpTools', f'{program} --help'], capture_output=True, text=True)
        help_text = result.stdout

        with open(output_file, 'w+') as f:
            f.write(f"## {program}\n")
            f.write("```\n")
            f.write(help_text)
            f.write("```\n")

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
    "fs_ctf"
    "fs_boxnet_infer",
    "fs_boxnet_train",
    "fs_export_micrographs",
    "fs_export_particles",
]

cli_programs_tiltseries =[
    "ts_import",
    "ts_stack",
    "ts_aretomo",
    "ts_etomo_fiducials",
    "ts_etomo_patches",
    "ts_ctf",
    "ts_reconstruct",
    "ts_template_match"
    "ts_import_alignments",
    "ts_eval_model"
]

generate_docs(cli_programs_general, "docs/reference/warptools/api/general.md")
generate_docs(cli_programs_frameseries, "docs/reference/warptools/api/frame_series.md")
generate_docs(cli_programs_tiltseries, "docs/reference/warptools/api/tilt_series.md")