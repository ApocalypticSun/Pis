import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class ImageSegmenter:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Imaginea nu a putut fi încărcată. Verificați calea.")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

    def region_based_segmentation(self, threshold=128):
        """Segmentare bazată pe regiuni folosind praguri"""
        _, binary = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def watershed_segmentation(self):
        """Segmentare folosind algoritmul Watershed"""
        # Aplicăm un filtru Gaussian pentru a reduce zgomotul
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # Detectăm zonele sigur de fundal/prima plan folosind thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Eliminăm zgomotul
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Zona sigură de fundal
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Calculăm distanța transformată
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        # Zona sigură de prim plan
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        # Zona necunoscută
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Etichetăm marcatorii
        _, markers = cv2.connectedComponents(sure_fg)

        # Adăugăm 1 la toate etichetele pentru a ne asigura că fundalul este 1, nu 0
        markers = markers + 1

        # Marcăm zona necunoscută cu 0
        markers[unknown == 255] = 0

        # Aplicăm Watershed
        markers = watershed(-dist_transform, markers, mask=opening)
        self.image[markers == -1] = [255, 0, 0]  # Marcăm granițele cu roșu

        return markers

    def clustering_based_segmentation(self, n_clusters=3):
        """Segmentare bazată pe clustering (K-means)"""
        # Redimensionăm imaginea pentru K-means
        pixel_values = self.image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Normalizăm valorile pixelilor
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(pixel_values)

        # Aplicăm K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(normalized_values)

        # Obținem etichetele și centroizii
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Transformăm înapoi în intervalul original
        centers = scaler.inverse_transform(centers)
        centers = np.uint8(centers)

        # Reconstruim imaginea segmentată
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(self.image.shape)

        return segmented_image

    def display_results(self, original, region_based, watershed, clustering):
        """Afișează rezultatele segmentării"""
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(original)
        plt.title('Imaginea Originală')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(region_based, cmap='gray')
        plt.title('Segmentare Bazată pe Regiuni')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(watershed)
        plt.title('Segmentare Watershed')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(clustering)
        plt.title('Segmentare Bazată pe Clustering')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# Exemplu de utilizare
if __name__ == "__main__":
    try:
        # Încărcăm imaginea (înlocuiți cu calea corectă)
        segmenter = ImageSegmenter('imagine_test.jpg')

        # Aplicăm metodele de segmentare
        region_based = segmenter.region_based_segmentation()
        watershed = segmenter.watershed_segmentation()
        clustering = segmenter.clustering_based_segmentation()

        # Afișăm rezultatele
        segmenter.display_results(segmenter.image, region_based, watershed, clustering)

    except Exception as e:
        print(f"Eroare: {e}")