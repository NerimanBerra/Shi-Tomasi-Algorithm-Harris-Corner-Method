import cv2
import numpy as np
image =cv2.imread(r"C:\Users\berra\Desktop\RKSoft\7\sudoku.png")
def harris(image_path):
    # Harris köşe tespiti parametreleri
    blockSize = 2
    apertureSize = 3
    k = 0.04

    # Görüntüyü gri tonlamaya çevirin
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Harris köşe tespiti uygulayın
    dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)

    # Sonuçları normalleştirin
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Köşeleri işaretleyin
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > 120:
                cv2.circle(image, (j, i), 2, (0, 255, 0), 2)

    return image

# Fonksiyonu çağırın ve sonucu görüntüleyin
result_image = harris(image)
cv2.imwrite(r"C:\Users\berra\Desktop\RKSoft\7\HarrisCorners.png", result_image)


import numpy as np
import cv2

image = cv2.imread(r"C:\Users\berra\Desktop\RKSoft\7\sudoku.png")

def process(image):
    # Görüntüyü gri tonlamaya çevirin
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # İyi özellikleri tespit edin (köşeler)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=35, qualityLevel=0.05, minDistance=10)
    
    # Tespit edilen köşeler üzerinde döngü
    for pt in corners:
        print(pt)
        
        # Rastgele renkler oluşturun
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        
        # Köşe noktalarının koordinatlarını alın
        x = int(pt[0][0])
        y = int(pt[0][1])
        
        # Köşe noktalarına daireler çizin
        cv2.circle(image, (x, y), 5, (int(b), int(g), int(r)), 2)
    
    return image

# Fonksiyonu çağırın ve sonucu görüntüleyin
result_image = process(image)
