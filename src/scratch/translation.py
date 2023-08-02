# Genre Translation


def translate_genre_to_english(genre):
    # Define the translation dictionary
    genre_translation = {
        "발라드": "ballad",
        "댄스": "dance",
        "트로트": "trot",
        "랩/힙합": "rap&hiphop",
        "인디음악": "indie-music",
        "록/메탈": "rock&metal",
        "포크/블루스": "folk&blues"
        # Add more genre translations here if needed
    }

    # Use the get() method to handle cases where the genre is not in the dictionary
    return genre_translation.get(genre, genre)
