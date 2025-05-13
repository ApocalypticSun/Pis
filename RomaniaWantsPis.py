import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class ImageSegmenter:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Imaginea nu a putut fi încărcată. Verificați calea.")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

    def region_based_segmentation(self, arg_type, threshold=128):
        # valori pentru argumentul "type":
        #    "binary" / "adaptive_gaussian" / "adaptive_mean" / "otsu"
        #    - reflectă diversele tipuri de segmentare bazată pe regiune
        #
        # valori pentru argumentul "threshold":
        #    0-100 => vor fi detectate zone mai întunecate
        #    100-200 => vor fi detectate zone mai luminoase

        if arg_type == "binary":
            _, binary = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        elif arg_type == "adaptive_gaussian":
            _, binary = cv2.threshold(self.gray, threshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        elif arg_type == "adaptive_mean":
            _, binary = cv2.threshold(self.gray, threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        elif arg_type == "otsu":
            _, binary = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)

        return binary

    def watershed_segmentation(self):
        # Aplicăm un filtru Gaussian pentru a reduce zgomotul
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # Detectăm zona sigură de fundal / prim plan folosind thresholding
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
        markers += 1

        # Marcăm zona necunoscută cu 0
        markers[unknown == 255] = 0

        # Aplicăm Watershed
        markers = watershed(-dist_transform, markers, mask=opening)
        self.image[markers == -1] = [255, 0, 0]  # Marcăm granițele cu roșu

        return markers

    def clustering_based_segmentation(self, n_clusters=3):
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

    def display_results(self, original, arg_region_based, arg_watershed, arg_clustering):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(original)
        plt.title('Imaginea Originală')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(arg_region_based, cmap='gray')
        plt.title('Segmentare Bazată pe Regiuni')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(arg_watershed)
        plt.title('Segmentare Watershed')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(arg_clustering)
        plt.title('Segmentare Bazată pe Clustering')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def display_results_region_based(self, arg_binary, arg_adaptive_gaussian, arg_adaptive_mean, arg_otsu):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(arg_binary, cmap='gray')
        plt.title('Segmentare Bazată pe Regiuni - Binar')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(arg_adaptive_gaussian, cmap='gray')
        plt.title('Segmentare Bazată pe Regiuni - Adaptiv Gaussian')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(arg_adaptive_mean, cmap='gray')
        plt.title('Segmentare Bazată pe Regiuni - Adaptiv Median')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(arg_otsu, cmap='gray')
        plt.title('Segmentare Bazată pe Regiuni - Otsu')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# Exemplu de utilizare
if __name__ == "__main__":
    try:
        # Încărcăm imaginea (înlocuiți cu calea corectă)
        segmenter = ImageSegmenter('imagine_test.jpg')

        # Aplicăm metodele de segmentare
        region_based_binary = segmenter.region_based_segmentation("binary")
        region_based_adaptive_gaussian = segmenter.region_based_segmentation("adaptive_gaussian")
        region_based_adaptive_mean = segmenter.region_based_segmentation("adaptive_mean")
        region_based_otsu = segmenter.region_based_segmentation("otsu")
        watershed = segmenter.watershed_segmentation()
        clustering = segmenter.clustering_based_segmentation()

        # Afișăm rezultatele
        segmenter.display_results(segmenter.image, region_based_binary, watershed, clustering)
        segmenter.display_results_region_based(region_based_binary, region_based_adaptive_gaussian, region_based_adaptive_mean, region_based_otsu)

    except Exception as e:
        print(f"Eroare: {e}")
