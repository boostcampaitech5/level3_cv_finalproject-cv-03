import yaml
import subprocess
import argparse


def quote_arg(arg):
    # Function to quote arguments with spaces
    return f'"{arg}"' if " " in arg else arg


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the training script with config.yaml settings."
    )
    parser.add_argument("config_file", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()

    # Load the configuration from the config.yaml file
    config = load_config(args.config_file)

    token = config["TOKEN_NAME"]
    class_name = config["CLASS_NAME"]
    exp_name = config["EXP_NAME"]

    # Prepare the command string
    command = "accelerate launch train.py"
    command += f" --pretrained_model_name_or_path={quote_arg(config['MODEL_NAME'])}"
    command += f" --pretrained_vae_model_name_or_path={quote_arg(config['PRETRAINED_VAE_MODEL_NAME_OR_PATH'])}"
    command += f" --instance_data_dir=/opt/ml/stable-diffusion-xl/data/tokens/{token}"
    command += (
        f" --class_data_dir=/opt/ml/stable-diffusion-xl/data/generated/{class_name}"
    )
    command += f" --output_dir=/opt/ml/stable-diffusion-xl/weights/{token}/{exp_name}"
    command += " --with_prior_preservation"
    command += f" --prior_loss_weight={config['PRIOR_LOSS_WEIGHT']}"
    command += f" --mixed_precision={config['MIXED_PRECISION']}"
    command += f" --instance_prompt={quote_arg(config['INSTANCE_PROMPT'])}"
    command += f" --class_prompt={quote_arg(config['CLASS_PROMPT'])}"
    command += f" --resolution={config['RESOLUTION']}"
    command += f" --train_batch_size={config['TRAIN_BATCH_SIZE']}"
    command += f" --gradient_accumulation_steps={config['GRADIENT_ACCUMULATION_STEPS']}"
    command += " --gradient_checkpointing"
    command += f" --learning_rate={config['LEARNING_RATE']}"
    command += f" --lr_scheduler={config['LR_SCHEDULER']}"
    command += f" --lr_warmup_steps={config['LR_WARMUP_STEPS']}"
    command += f" --max_train_steps={config['MAX_TRAIN_STEPS']}"
    command += f" --num_class_images={config['NUM_CLASS_IMAGES']}"
    command += f" --sample_batch_size={config['SAMPLE_BATCH_SIZE']}"
    command += f" --seed={config['SEED']}"
    if config["PUSH_TO_HUB"]:
        command += " --push_to_hub"
        command += f" --hub_model_id={config['HUB_MODEL_ID']}"
    if config["TRAIN_TEXT_ENCODER"]:
        command += " --train_text_encoder"

    print(command)

    # Execute the generated command
    subprocess.run(command, shell=True)
