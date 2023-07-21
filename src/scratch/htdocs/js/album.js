function checklogin() {
    if (!Kakao.Auth.getAccessToken()) {
        alert('check-로그아웃');
        document.getElementById("kakao-login").style.display = "block";
        document.getElementById("logout").style.display = "none";
    }
    else {
        alert('check-로그인');
        document.getElementById("kakao-login").style.display = "none";
        document.getElementById("logout").style.display = "block";
    }
}

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
                alert('fail: ' + JSON.stringify(err))
            },
        })
        Kakao.Auth.setAccessToken(undefined)
    }
    else {
        alert('로그인 상태가 아닙니다.')
    }
}


document.addEventListener("DOMContentLoaded", () => {
    // 모달을 열 때 클릭된 버튼 숫자정보(1~4) 가져오기
    $('#download_modal').on('show.bs.modal', function (event) {
        button = $(event.relatedTarget);
        buttonType = button.data('button-type');
    });

    document.querySelector("#review_send_btn").addEventListener("click", () => {
        user_starpoint = 0
        user_review = document.getElementById("review_comment").value
        for (var i = 1; i <= 10; i++) {
            cur_review = "starpoint_" + i;
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
            $('#download_modal').modal('hide');
            //TODO : 리뷰 저장하는 코드 추가
            alert("소중한 리뷰 감사드립니다!" + "\n" + "별점 : " + user_starpoint + "점" + "\n" + "한줄평 : " + user_review);
            // 리뷰 초기화
            document.getElementById("review_comment").value = ""
            document.getElementById(cur_review).checked = false
            // TODO: 리뷰작성시 바로 다운로드되게 하기
            console.log(document.getElementById("image" + buttonType).src) // 가져온 링크주소
        }
    })


    const badge_checked = "badge bg-primary-subtle border border-primary-subtle text-primary-emphasis rounded-pill mb-1"
    const badge_not_checked = "badge bg-secondary-subtle border border-secondary-subtle text-secondary-emphasis rounded-pill mb-1"
    document.querySelector("#img_create_btn").addEventListener("click", () => {
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
        for (var i = 1; i <= 13; i++) {
            cur_genre = "genre" + i
            if (document.getElementById(cur_genre).className == badge_checked) {
                select_genre.push(document.getElementById(cur_genre).textContent)
            }
        }
        selects = "Model : " + select_model + "\n" + "Song : " + select_song + "\n" + "Artist : " + select_artist + "\n" + "Album : " + select_album + "\n" + "Genre : " + select_genre + "\n" + "Lyrics : " + select_lyrics + "\n"
        alert(selects)
    })

    document.querySelector("#genre1").addEventListener("click", () => {
        if (document.getElementById("genre1").className == badge_not_checked) {
            document.getElementById("genre1").className = badge_checked
        }
        else {
            document.getElementById("genre1").className = badge_not_checked
        }
    })
    document.querySelector("#genre2").addEventListener("click", () => {
        if (document.getElementById("genre2").className == badge_not_checked) {
            document.getElementById("genre2").className = badge_checked
        }
        else {
            document.getElementById("genre2").className = badge_not_checked
        }
    })
    document.querySelector("#genre3").addEventListener("click", () => {
        if (document.getElementById("genre3").className == badge_not_checked) {
            document.getElementById("genre3").className = badge_checked
        }
        else {
            document.getElementById("genre3").className = badge_not_checked
        }
    })
    document.querySelector("#genre4").addEventListener("click", () => {
        if (document.getElementById("genre4").className == badge_not_checked) {
            document.getElementById("genre4").className = badge_checked
        }
        else {
            document.getElementById("genre4").className = badge_not_checked
        }
    })
    document.querySelector("#genre5").addEventListener("click", () => {
        if (document.getElementById("genre5").className == badge_not_checked) {
            document.getElementById("genre5").className = badge_checked
        }
        else {
            document.getElementById("genre5").className = badge_not_checked
        }
    })
    document.querySelector("#genre6").addEventListener("click", () => {
        if (document.getElementById("genre6").className == badge_not_checked) {
            document.getElementById("genre6").className = badge_checked
        }
        else {
            document.getElementById("genre6").className = badge_not_checked
        }
    })
    document.querySelector("#genre7").addEventListener("click", () => {
        if (document.getElementById("genre7").className == badge_not_checked) {
            document.getElementById("genre7").className = badge_checked
        }
        else {
            document.getElementById("genre7").className = badge_not_checked
        }
    })
    document.querySelector("#genre8").addEventListener("click", () => {
        if (document.getElementById("genre8").className == badge_not_checked) {
            document.getElementById("genre8").className = badge_checked
        }
        else {
            document.getElementById("genre8").className = badge_not_checked
        }
    })
    document.querySelector("#genre9").addEventListener("click", () => {
        if (document.getElementById("genre9").className == badge_not_checked) {
            document.getElementById("genre9").className = badge_checked
        }
        else {
            document.getElementById("genre9").className = badge_not_checked
        }
    })
    document.querySelector("#genre10").addEventListener("click", () => {
        if (document.getElementById("genre10").className == badge_not_checked) {
            document.getElementById("genre10").className = badge_checked
        }
        else {
            document.getElementById("genre10").className = badge_not_checked
        }
    })
    document.querySelector("#genre11").addEventListener("click", () => {
        if (document.getElementById("genre11").className == badge_not_checked) {
            document.getElementById("genre11").className = badge_checked
        }
        else {
            document.getElementById("genre11").className = badge_not_checked
        }
    })
    document.querySelector("#genre12").addEventListener("click", () => {
        if (document.getElementById("genre12").className == badge_not_checked) {
            document.getElementById("genre12").className = badge_checked
        }
        else {
            document.getElementById("genre12").className = badge_not_checked
        }
    })
    document.querySelector("#genre13").addEventListener("click", () => {
        if (document.getElementById("genre13").className == badge_not_checked) {
            document.getElementById("genre13").className = badge_checked
        }
        else {
            document.getElementById("genre13").className = badge_not_checked
        }
    })

})
