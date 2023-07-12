import streamlit as st
import pandas as pd
import requests
import base64
import io
from PIL import Image
from pytz import timezone
from datetime import datetime

def main():
    st.title("앨범 커버 생성")

    # 1. Input album information
    st.header("앨범 정보 입력하기")
    
    song_names = st.text_input("노래 제목을 입력해주세요", placeholder="노래 제목")
    artist_name = st.text_input("당신의 예명(가수 이름)을 입력해주세요", placeholder="Your Artist Name")
    genre_kr = st.selectbox(
        "장르를 선택해주세요",
        [
            "발라드",
            "댄스",
            "랩/힙합",
            "알앤비",
            "팝",
            "소울",
            "인디",
            "락",
            "메탈",
            "트로트",
            "포크",
            "블루스",
            "재즈",
        ],
    )
    album_name = st.text_input("앨범명을 입력해주세요", placeholder="앨범명")
    release = st.date_input("발매일을 선택해주세요")
    lyric = st.text_area("가사를 입력해주세요", placeholder="노래 가사")

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
        "발라드": "Ballad",
        "댄스": "Dance",
        "랩/힙합": "Hip-Hop",
        "알앤비": "R&B",
        "팝": "Pop",
        "소울": "Soul",
        "인디": "Indie",
        "락": "Rock",
        "메탈": "Metal",
        "트로트": "Trot",
        "포크": "Folk",
        "블루스": "Blues",
        "재즈": "Jazz",
    }
    info["genre"] = genre_t[genre_kr]

    info_df = pd.DataFrame(
        list(info.values()), index=list(info.keys()), columns=["Input"]
    )
    info_table = st.dataframe(info_df, use_container_width=True)
    
    # Set 'img' variable to session state
    if 'img' not in st.session_state:
        st.session_state.img = ['', '', '', '']

    # 3. Inference
    st.header("앨범 커버 생성하기")
    gen_button = st.button("앨범 커버 생성하기", use_container_width=True)

    # Create 2x2 grid
    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2]
    
    if gen_button:
        with st.spinner("생성중..."):
            # Call the FastAPI server to generate the album cover
            response = requests.post("http://localhost:8000/generate_cover", json=info)
            if response.status_code == 200:
                images = response.json()["images"]

                # Assign images to the cells in the grid
                for i, image in enumerate(images):
                    img_data = base64.b64decode(image)
                    img = Image.open(io.BytesIO(img_data))
                    st.session_state.img[i] = img
                    cols[i].image(img, width=300)
            else:
                st.error("앨범 커버 생성에 실패했습니다. 다시 시도해주세요!")
    else:
        if st.session_state.img == ['', '', '', '']:
            for i in range(len(cols)):
                cols[i].empty()
        else:
            for i in range(len(cols)):
                cols[i].image(st.session_state.img[i], width=300)

    with st.expander("**REVIEW US!**"):
        review_rating = st.select_slider('생성된 이미지에 얼만큼 만족하시나요?', options=range(1, 6), value=5)
        st.markdown("<h5 style='text-align: center;'>"+"❤️"*review_rating+"🖤"*(5-review_rating)+"</h5>", unsafe_allow_html=True)
        review_text = st.text_input("review_text", placeholder="간단한 사용 후기를 남겨주세요!", label_visibility="hidden")
        review_btn = st.button("리뷰보내기")
        if review_btn:
            review = {
                "rating": review_rating,
                "comment": review_text
            }
            response = requests.post("http://localhost:8000/review", json=review)
            if response.status_code == 200:
                st.write("소중한 의견 감사합니다!")
            else:
                st.error("리뷰 보내기에 실패했습니다. 다시 시도해주세요!")
                
if __name__ == "__main__":
    main()
