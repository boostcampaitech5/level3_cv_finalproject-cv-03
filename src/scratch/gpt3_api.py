# Python built-in modules
import os

# OpenAI API
import openai

# Built-in modules
from utils import load_yaml


def get_description(
    lyrics: str, artist_name: str, album_name: str, song_names: str
) -> str:
    gpt_config = load_yaml(
        os.path.join(
            "/opt/ml/level3_cv_finalproject-cv-03/src/scratch/config", "private.yaml"
        ),
        "gpt",
    )

    # OpenAI API key
    # https://platform.openai.com/
    openai.api_key = gpt_config["api_key"]

    # -- 공백, 줄바꿈 제거
    lyrics = lyrics.strip()
    lyrics = lyrics.replace("\n\n", " ")
    lyrics = lyrics.replace("\n", " ")

    # message
    message = [
        f"Describe the atmosphere or vibe of these lyrics into 5 different words seperated with comma. They should be optimal for visualizing a atmosphere. \n\n{lyrics}",
        f"Also describe a atmosphere using the following Artist name, Album name and Song names into 5 different words seperated with comma. They should be optimal for visualizing a atmosphere. Artist name : {artist_name} \n Album name : {album_name} \n Song names : {song_names}",
    ]

    # Set up the API call
    responses = []
    for idx in range(len(message)):
        response = openai.ChatCompletion.create(
            model=gpt_config["model"],
            messages=[
                {
                    "role": gpt_config["role"],
                    "content": message[idx],
                }
            ],
            max_tokens=gpt_config[
                "max_tokens"
            ],  # Adjust the value to control the length of the generated description
            temperature=gpt_config[
                "temperature"
            ],  # Adjust the temperature to control the randomness of the output
            n=gpt_config["n_response"],  # Generate a single response
            stop=gpt_config["stop"],  # Stop generating text at any point
        )
        responses.append(response["choices"][0]["message"]["content"])

    return ",".join(responses)


def get_dreambooth_prompt(
    lyrics: str,
    album_name: str,
    song_names: str,
    gender: str,
    genre: str,
    artist_name: str,
) -> str:
    gpt_config = load_yaml(
        os.path.join(
            "/opt/ml/level3_cv_finalproject-cv-03/src/scratch/config", "private.yaml"
        ),
        "gpt",
    )
    openai.api_key = gpt_config["api_key"]

    lyrics = lyrics.strip()
    lyrics = lyrics.replace("\n\n", " ")
    lyrics = lyrics.replace("\n", " ")

    message = [
        f"read a \n\n'{lyrics}', \n\n'{song_names}'. \n and, give a just one good prompt for generate album cover images that matching with above texts.\
                prompt should start with 'A image of {gender} ~' and should not over 75 tokens."
    ]

    # Set up the API call
    responses = []
    for idx in range(len(message)):
        response = openai.ChatCompletion.create(
            model=gpt_config["model"],
            messages=[
                {
                    "role": gpt_config["role"],
                    "content": message[idx],
                }
            ],
            max_tokens=gpt_config[
                "max_tokens"
            ],  # Adjust the value to control the length of the generated description
            temperature=gpt_config[
                "temperature"
            ],  # Adjust the temperature to control the randomness of the output
            n=gpt_config["n_response"],  # Generate a single response
            stop=gpt_config["stop"],  # Stop generating text at any point
        )
        responses.append(response["choices"][0]["message"]["content"])

    return ",".join(responses)
