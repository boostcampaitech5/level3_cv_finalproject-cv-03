import streamlit as st
import pandas as pd
import requests
import base64
import io
from PIL import Image


def main():
    st.title("Make a Album Cover")

    # 1. Input album information
    st.header("Input Your Album Information")

    song_names = st.text_input("Input Your Song Names", placeholder="Your Song")
    artist_name = st.text_input(
        "Input Your Artist Name", placeholder="Your Artist Name"
    )
    genre_kr = st.selectbox(
        "Select the Genre",
        [
            "Ballad",
            "Dance",
            "Rap/Hiphop",
            "R&B",
            "Pop",
            "Soul",
            "Indie",
            "Rock",
            "Metal",
            "Trot",
            "Folk",
            "Blues",
            "Jazz",
        ],
    )
    album_name = st.text_input("Input Your Album Name", placeholder="Your Album Name")
    release = st.date_input("Select The Release Date")
    lyric = st.text_area("Input the Lyrics", placeholder="The Lyrics for Your Song")

    year = release.year
    month = release.month
    day = release.day
    # version - 1
    season = f"{year}-{month}-{day}"
    # version - 2
    if month > 11 or month < 3:
        season = "winter"
    elif 3 <= month < 6:
        season = "spring"
    elif 6 <= month < 9:
        season = "summer"
    elif 9 <= month < 12:
        season = "fall"

    info = {
        "song_names": song_names,
        "artist_name": artist_name,
        "genre": genre_kr,
        "album_name": album_name,
        "release": f"{year}-{month}-{day}",
        "lyric": lyric,
    }

    # Translation
    genre_t = {
        "Ballad": "Ballad",
        "Dance": "Dance",
        "Rap/Hiphop": "Hip-Hop",
        "R&B": "R&B",
        "Pop": "Pop",
        "Soul": "Soul",
        "Indie": "Indie",
        "Rock": "Rock",
        "Metal": "Metal",
        "Trot": "Trot",
        "Folk": "Folk",
        "Blues": "Blues",
        "Jazz": "Jazz",
    }
    info["genre"] = genre_t[genre_kr]

    info_df = pd.DataFrame(
        list(info.values()), index=list(info.keys()), columns=["Input"]
    )
    info_table = st.dataframe(info_df, use_container_width=True)

    # 3. Inference
    st.header("Generate Album Cover")
    if st.button("Generate Album Cover", use_container_width=True):
        with st.spinner("Wait for it..."):
            # Call the FastAPI server to generate the album cover
            response = requests.post("http://localhost:8000/generate_cover", json=info)
            if response.status_code == 200:
                images = response.json()["images"]

                # Create 2x2 grid
                col1, col2 = st.columns(2)
                cols = [col1, col2, col1, col2]

                # Assign images to the cells in the grid
                for i, image in enumerate(images):
                    img_data = base64.b64decode(image)
                    img = Image.open(io.BytesIO(img_data))
                    cols[i].image(img, width=300)
            else:
                st.error("Failed to generate album cover. Please try again.")


if __name__ == "__main__":
    main()
