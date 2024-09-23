import subprocess


def run_script(content):
    script_path = "run_finetune.sh"

    with open(script_path, "w") as file:
        file.write(content)

    subprocess.run(['chmod', '+x', script_path])

    # Step 3: Run the shell script
    result = subprocess.run(["./" + script_path], shell=True, capture_output=True, text=True)

    # Output the result
    print("Output:", result.stdout)
    print("Errors:", result.stderr)
