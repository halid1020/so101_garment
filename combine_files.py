import glob
import os


def combine_files_recursively(directory_path, output_filename="combined_code.txt"):
    """
    Recursively finds all .py, .json, and .yaml/.yml files in a directory
    and its subdirectories, combining them into a single text file.
    Ignores any files located inside 'venv' or '.venv' directories.
    """
    # Define the file extensions you want to capture
    extensions = ["*.py", "*.json", "*.yaml", "*.yml", "*.md"]
    matched_files = []

    # Define directory names to ignore exactly
    ignore_dirs = {"venv", ".venv"}

    # Search for each extension recursively
    for ext in extensions:
        # The "**" tells glob to look in all subdirectories
        search_pattern = os.path.join(directory_path, "**", ext)

        # We must explicitly set recursive=True for the "**" to work
        found_files = glob.glob(search_pattern, recursive=True)

        for file_path in found_files:
            # Normalize slashes and split the path into its individual folders/components
            path_parts = file_path.replace("\\", "/").split("/")

            # Only add the file if it is NOT inside any of the ignored directories
            if not any(ignored in path_parts for ignored in ignore_dirs):
                matched_files.append(file_path)

    # Check if any files were found
    if not matched_files:
        print(
            f"No target files found in '{directory_path}' or its subdirectories (excluding venv)."
        )
        return

    try:
        # Open the output file in write mode
        with open(output_filename, "w", encoding="utf-8") as outfile:
            for file_path in matched_files:
                # We use the full relative path instead of just the basename
                # This helps you know exactly which subfolder the code came from!
                filename = os.path.relpath(file_path, directory_path)

                # Write a clear visual header for each file
                outfile.write(f"{'='*50}\n")
                outfile.write(f"FILE: {filename}\n")
                outfile.write(f"{'='*50}\n\n")

                # Read the contents of the current file and write it
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"# Error reading {filename}: {e}\n")

                # Add some blank space before the next file begins
                outfile.write("\n\n\n")

        print(f"Success! Combined {len(matched_files)} files into '{output_filename}'.")

    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")


# --- Execution ---
if __name__ == "__main__":
    # The directory you want to scan.
    # "." means the current directory where this script is located.
    target_directory = "."

    # The name of the text file you want to generate
    output_file = "all_my_code_recursive.txt"

    combine_files_recursively(target_directory, output_file)
