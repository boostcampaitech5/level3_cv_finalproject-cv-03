import openai


def get_description(lyrics, artist_name, album_name, season, song_names):
    # OpenAI API key
    # https://platform.openai.com/
    openai.api_key = "Your_API_Key"

    # -- 공백, 줄바꿈 제거
    lyrics = lyrics.strip()
    lyrics = lyrics.replace("\n\n", " ")
    lyrics = lyrics.replace("\n", " ")

    # Set up the API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Describe the atmosphere or vibe of these lyrics into 5 different words seperated with comma. They should be optimal for visualizing a atmosphere. \n\n{lyrics}",
            }
        ],
        max_tokens=50,  # Adjust the value to control the length of the generated description
        temperature=0.5,  # Adjust the temperature to control the randomness of the output
        n=1,  # Generate a single response
        stop=None,  # Stop generating text at any point
    )

    response2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Also describe a atmosphere using the following Artist name, Album name, Season and Song names into 5 different words seperated with comma. They should be optimal for visualizing a atmosphere. Artist name : {artist_name} \n Album name : {album_name} \n Season : {season}, Song names : {song_names}",
            }
        ],
        max_tokens=50,
        temperature=0.5,
        n=1,
        stop=None,
    )

    # Get the generated description
    description1 = response["choices"][0]["message"]["content"]
    description2 = response2["choices"][0]["message"]["content"]

    return description1 + "," + description2
