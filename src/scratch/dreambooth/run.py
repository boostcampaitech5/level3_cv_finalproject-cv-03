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
    parser.add_argument("--config-file", type=str, help="Path to the config.yaml file.")
    parser.add_argument("--token", type=str, help="user's token.")
    parser.add_argument("--user-gender", type=str, help="user's gender for class")
    parser.add_argument("--seed", type=int, help="random seeds")
    args = parser.parse_args()

    # Load the configuration from the config.yaml file
    config = load_config(args.config_file)

    token = args.token
    class_name = args.user_gender

    # Prepare the command string
    command = "accelerate launch train_dreambooth.py"
    command += f" --pretrained_model_name_or_path={quote_arg(config['MODEL_NAME'])}"
    command += f" --pretrained_vae_model_name_or_path={quote_arg(config['PRETRAINED_VAE_MODEL_NAME_OR_PATH'])}"
    command += f" --instance_data_dir=/opt/ml/level3_cv_finalproject-cv-03/src/scratch/dreambooth/data/users/{token}"
    command += f" --output_dir=/opt/ml/level3_cv_finalproject-cv-03/src/scratch/dreambooth/weights/{token}"
    command += f" --mixed_precision={config['MIXED_PRECISION']}"
    command += f" --instance_prompt='A photo of a {token} {class_name}'"
    command += f" --resolution={config['RESOLUTION']}"
    command += f" --train_batch_size={config['TRAIN_BATCH_SIZE']}"
    command += f" --gradient_accumulation_steps={config['GRADIENT_ACCUMULATION_STEPS']}"
    command += " --gradient_checkpointing"
    command += f" --learning_rate={config['LEARNING_RATE']}"
    command += f" --lr_scheduler={config['LR_SCHEDULER']}"
    command += f" --lr_warmup_steps={config['LR_WARMUP_STEPS']}"
    command += f" --max_train_steps={config['MAX_TRAIN_STEPS']}"
    command += f" --sample_batch_size={config['SAMPLE_BATCH_SIZE']}"
    command += f" --seed={int(args.seed)}"

    if config["WITH_PRIOR_PRESERVATION"]:
        command += " --with_prior_preservation"
        command += f" --prior_loss_weight={config['PRIOR_LOSS_WEIGHT']}"
        command += f" --num_class_images={config['NUM_CLASS_IMAGES']}"
        command += f" --class_data_dir=/opt/ml/level3_cv_finalproject-cv-03/src/scratch/dreambooth/data/generated/{class_name}"
        command += f" --class_prompt='A photo of {class_name}'"

    if config["PUSH_TO_HUB"]:
        command += " --push_to_hub"
        command += f" --hub_model_id={config['HUB_MODEL_ID']}"

    if config["TRAIN_TEXT_ENCODER"]:
        command += " --train_text_encoder"

    print(command)
