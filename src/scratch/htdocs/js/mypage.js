// 이미지를 동적으로 추가하는 함수
function addImagesToContainer(userImages) {
    userImages.forEach((image) => {
        const imgElement = document.createElement('img');
        imgElement.src = image['url'];
        imgElement.alt = '이미지';

        imgElement.addEventListener('click', () => {
            showImageModal(image);
        });

        imageContainer.querySelector('.image-grid').appendChild(imgElement);
    });
}

function showImageModal(image) {
    const modalBody = document.querySelector('.modal-body');
    // 이미지 클릭 시 모달의 내용을 새로운 정보로 업데이트
    document.getElementById('request_time').innerText = image.request_time;
    document.getElementById('song_names').innerText = image.song_names;
    document.getElementById('artist_name').innerText = image.artist_name;
    document.getElementById('genre').innerText = image.genre;
    document.getElementById('album_name').innerText = image.album_name;
    document.getElementById('lyric').innerText = image.lyric;

    // 모달 띄우기
    const imageModal = new bootstrap.Modal(document.getElementById('imageModal'));
    imageModal.show();
  }

document.addEventListener("DOMContentLoaded", () => {
    const imageContainer = document.getElementById('imageContainer');
    // 여러 입력 테스트용
    const albumInput = {
        request_time: '생성날짜',
        song_names: '노래제목',
        artist_name: '아티스트명',
        genre: '장르',
        album_name: '앨범명',
        lyric: '가사',
        url: 'images/example1.jpg'
    };
    // TODO: 현재 사용자의 기록들 보여주기
    user_email = sessionStorage.getItem('user_email')  // 식별자 - 현재 사용자의 이메일
    const userImages = new Array(16).fill(albumInput)  // [albumInput, albumInput, ...] 와 같음
    addImagesToContainer(userImages)

})
