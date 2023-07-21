//카카오 로그인
function kakaoLogin() {
    if (!Kakao.Auth.getAccessToken()) {
        Kakao.Auth.login({
            success: function (response) {
                Kakao.API.request({
                    url: '/v2/user/me',
                    success: function (response) {
                        alert('사용자 닉네임: ' + response.kakao_account.profile.nickname + '\n'
                            + '사용자 성별: ' + response.kakao_account.gender + '\n'
                            + '사용자 연령대: ' + response.kakao_account.age_range + '\n'
                            + '사용자 이메일: ' + response.kakao_account.email);
                        // document.getElementById("kakao-login").style.display = "none";
                        // document.getElementById("logout").style.display = "block";
                    },
                    fail: function (error) {
                        alert(
                            'login success, but failed to request user information: ' +
                            JSON.stringify(error)
                        )
                    },
                })
            },
            fail: function (error) {
                console.log(error)
            },
        })
    }
    else {
        alert("이미 로그인 상태입니다.")
    }

}
//카카오 로그아웃
function kakaoLogout() {
    if (Kakao.Auth.getAccessToken()) {
        Kakao.API.request({
            url: '/v1/user/unlink',
            success: function (response) {
                alert('로그아웃되었습니다.')
                // document.getElementById("kakao-login").style.display = "block";
                // document.getElementById("logout").style.display = "none";
            },
            fail: function (error) {
                alert('fail: ' + JSON.stringify(error))
            },
        })
        Kakao.Auth.setAccessToken(undefined)
    }
    else {
        alert('로그인 상태가 아닙니다.')
    }
}

function imageDownload(num) {
    new_a = document.createElement("a");
    new_a.href = document.getElementById("image" + num).src;
    new_a.download = "album_image_" + num + ".jpeg";
    document.body.appendChild(new_a);
    new_a.click();
    document.body.removeChild(new_a);
}

document.addEventListener("DOMContentLoaded", () => {
    // 모달을 열 때 클릭된 버튼 숫자정보(1~4) 가져오기
    $('#download_modal').on('show.bs.modal', function (event) {
        button = $(event.relatedTarget);
        buttonType = button.data('button-type');
    });

    document.querySelector("#review_send_btn").addEventListener("click", async (e) => {
        // e.preventDefault()
        user_starpoint = 0
        user_review = document.getElementById("review_comment").value
        for (let i = 1; i <= 10; i++) {
            cur_review = "starpoint_" + i
            if (document.getElementById(cur_review).checked == true) {
                user_starpoint = i / 2;
                break;
            }
        }
        if (user_starpoint == 0 || user_review == "") {
            alert('리뷰를 작성해주세요.');
            document.getElementById("review_comment").focus();
        }
        else {
            const reviewData = {
                rating: user_starpoint,
                comment: user_review
            };

            try {
                const response = await fetch('http://localhost:8000/review', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(reviewData),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                console.log(data);  // Log the response data to check if it's correct

                $('#download_modal').modal('hide');
                // TODO : 리뷰 저장하는 코드 추가
                alert("소중한 리뷰 감사드립니다!" + "\n" + "별점 : " + user_starpoint + "점" + "\n" + "한줄평 : " + user_review);
                // 리뷰 초기화
                document.getElementById("review_comment").value = ""
                document.getElementById(cur_review).checked = false

                // 리뷰작성시 이미지 다운로드
                imageDownload(buttonType);
            } catch (error) {
                console.error('Error:', error);
            }
        }
    })

    const badge_checked = "badge bg-primary-subtle border border-primary-subtle text-primary-emphasis rounded-pill mb-1"
    const badge_not_checked = "badge bg-secondary-subtle border border-secondary-subtle text-secondary-emphasis rounded-pill mb-1"
    document.querySelector("#img_create_btn").addEventListener("click", async (e) => {
        // TODO: 이미지 생성이 진행중인 모습 보여주기
        e.preventDefault()
        // 버튼 동작 체크
        console.log("Button Clicked!");
        select_model = ""
        select_song = document.getElementById("song_name").value
        select_artist = document.getElementById("artist_name").value
        select_album = document.getElementById("album_name").value
        select_genre = []
        select_lyrics = document.getElementById("lyrics").value

        if (document.getElementById("listGroupRadios1").checked == true) {
            select_model = document.getElementById("listGroupRadios1").value
        }
        else {
            select_model = document.getElementById("listGroupRadios2").value
        }
        genre = document.getElementsByClassName("badge")
        for (let i = 0; i < genre.length; i++) {
            cur_genre = genre[i].id;
            if (document.getElementById(cur_genre).className == badge_checked) {
                select_genre.push(document.getElementById(cur_genre).textContent)
            }
        }

        selects = "Model : " + select_model + "\n" + "Song : " + select_song + "\n" + "Artist : " + select_artist + "\n" + "Album : " + select_album + "\n" + "Genre : " + select_genre + "\n" + "Lyrics : " + select_lyrics + "\n"
        // alert(selects)

        // Fetch API call
        const albumInput = {
            song_names: select_song,
            artist_name: select_artist,
            genre: select_genre.join(", "),
            album_name: select_album,
            lyric: select_lyrics,
        };

        try {
            const response = await fetch('http://localhost:8000/generate_cover', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(albumInput),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            console.log(data.images);  // Log the images data to check if it's correct
            for (let i = 1; i <= 4; i++) {
                let imgElement = document.getElementById(`image${i}`);
                // Log img element
                // console.log(imgElement);
                // console.log(data.images[i-1]);
                imgElement.src = 'data:image/jpeg;base64,' + data.images[i - 1];
            }
        } catch (error) {
            console.error('Error:', error);
        }

    })
    document.querySelectorAll(".badge").forEach(obj => {
        obj.addEventListener("click", () => {
            console.log(document.getElementById(obj.id).style.background)
            if (document.getElementById(obj.id).className == badge_not_checked)
                document.getElementById(obj.id).className = badge_checked
            else {
                document.getElementById(obj.id).className = badge_not_checked
            }
        })
    })
})
