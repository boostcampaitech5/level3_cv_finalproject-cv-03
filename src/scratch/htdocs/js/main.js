
// JavaScript function to handle image load
function onImageLoad(image) {
    // The image has been fully loaded, and its dimensions are now available
    console.log('Image dimensions:', image.width, image.height);
}

// Function to fetch and display the latest and best-rated album images

async function fetchAlbumImages() {
    try {
        const response = await fetch('http://49.50.167.24:30008/api/get_album_images', { method: 'GET',
        mode: "cors",
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json'
            // 'Content-Type': 'application/x-www-form-urlencoded',
          },});

        const data = await response.json();

        if (response.ok) {
        // Assuming your container element has the ID "album-images-container"
            const container = document.getElementById('album-images-container');

        // Clear the existing content of the container
            container.innerHTML = '';

        // Loop through the received data.album_images and create image elements
            for (let i = 0; i < data.album_images.length; i++) {
                const albumImage = data.album_images[i];

                // Create the image element
                const img = document.createElement('img');

                img.className = 'default-img'; // Set the class name

                img.src = albumImage.url;
                img.alt = albumImage.song_names; // Set alt attribute to song_names or any relevant text
                img.width = '300';
                img.height = '300';

                const card = document.createElement('div');
                card.className = 'col';
                card.appendChild(img);

                const cardBody = document.createElement('div');
                cardBody.className = 'card-body';

                const cardText = document.createElement('p');
                cardText.className = 'card-text';
                cardText.innerText = `Song: ${albumImage.song_names}\nArtist: ${albumImage.artist_name}\nAlbum: ${albumImage.album_name}\nGenre: ${albumImage.genre}`;

                cardBody.appendChild(cardText);
                card.appendChild(cardBody);

                container.appendChild(card);
            };
        } else {
            console.error('Error fetching album images:', data.message);
        }
    } catch (error) {
        console.error('Error fetching album images:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetchAlbumImages();
  });
