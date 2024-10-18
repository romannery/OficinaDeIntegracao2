import cv2
import numpy as np
import random


def processa_imagem(cam_img:str):
    def paintPerimeter(base: np.array, target: np.array, larg: int, alt: int, y: int, x: int, color: np.array):
        target[y: y+alt, x: x + larg] = color
        #target is the target image for painting
        #base is a grayscale image used to verify which cells have been painted
        
        if target[y - alt: y, x: x + larg].size > 0: # verfying and painting upper cell
            if cv2.countNonZero(base[y - alt: y, x: x + larg]) == 0:
                base[y - alt: y, x: x + larg] = 1
                paintPerimeter(base, target, larg, alt, y - alt, x, color)
        
        if target[y + alt: y + 2 * alt, x: x + larg].size > 0: # verfying and painting lower cell
            if cv2.countNonZero(base[y + alt: y + 2 * alt, x:  x + larg]) == 0:
                base[y + alt: y + 2 * alt, x:  x + larg] = 1
                paintPerimeter(base, target, larg, alt, y + alt, x, color)
        
        if target[y: y + alt, x + larg: x + 2 * larg].size > 0: # verfying and painting right cell
            if cv2.countNonZero(base[y: y + alt, x + larg: x + 2 * larg]) == 0:
                base[y: y + alt, x + larg: x + 2 * larg] = 1
                paintPerimeter(base, target, larg, alt, y, x + larg, color)
        
        if target[y: y + alt, x - larg: x].size > 0:
            if cv2.countNonZero(base[y: y + alt, x - larg: x]) == 0: # verfying and painting left cell
                base[y: y + alt, x - larg: x] = 1
                paintPerimeter(base, target, larg, alt, y, x - larg, color)


    # Load the input image
    image = cv2.imread(cam_img)
    imagem_original=image.copy()

    imagem_original=cv2.cvtColor(imagem_original,cv2.COLOR_BGR2BGRA)

    # Convert the image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Blur to improve accuracy
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 100, 200)
    #Test image of edge detection output
    #cv2.imshow('test', edges)

    # Define the grid cell size and color
    cell_size = (image.shape[1] // 250, image.shape[0] // 250)  # Calculate cell size
    grid_color = (50, 205, 50)  # Red grid lines (in BGR format)

    # Iterate over grid cells
    for x in range(0, image.shape[1], cell_size[0]):
        for y in range(0, image.shape[0], cell_size[1]):
            # Get the portion of the image corresponding to the cell
            cell = edges[y:y+cell_size[1], x:x+cell_size[0]]

            # Check if there are any edges in the cell
            if cv2.countNonZero(cell) > 0:
                # Fill the cell with a different color
                image[y:y+cell_size[1], x:x+cell_size[0]] = (0, 0, 255)  # Green color

    # detection values for the color black
    lower = np.array([0, 0, 255])
    upper = np.array([0, 0, 255])

    mask = cv2.inRange(image, lower, upper)

    masked = cv2.bitwise_and(image,image, mask=mask)

    result = image - masked
    final = image - result # inverting the mask so as to isolate the limits
    final_gs = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)


    def find_central_pixel(image):
        # Find the coordinates of the central pixel
        height, width = image.shape[:2]
        central_x = width // 2
        central_y = height // 2
        return central_x, central_y

    def paint_until_edge(image, central_x, central_y):
        # Create a mask for flood fill
        mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)

        # Define the fill color (you can change this to your desired color)
        fill_color = (0, 255, 0)  # Green color in BGR format

        # Perform flood fill operation
        cv2.floodFill(image, mask, (central_x, central_y), fill_color, loDiff=(10, 10, 10), upDiff=(10, 10, 10))

        return image


    # Find the central pixel
    central_x, central_y = find_central_pixel(final)

    # Paint until an edge is encountered
    result_image = paint_until_edge(final, central_x, central_y)

    kernel = np.ones((11,11))
    lower = np.array([0, 100, 0])
    upper = np.array([0, 255, 0])
    mask  = cv2.inRange(result_image, lower, upper)
    masked = cv2.bitwise_and(result_image,result_image, mask=mask)
    sem_vermelho = masked
    sem_vermelho[mask>0] = (255,255,255)

    kernel = np.ones((95,95))
    dilatado = cv2.dilate(sem_vermelho, kernel, iterations=1)
    erodido = cv2.erode(dilatado, kernel, iterations=1)


    y_nonzero, x_nonzero, _ = np.nonzero(erodido)
    y_min=np.min(y_nonzero)
    y_max=np.max(y_nonzero)
    x_min=np.min(x_nonzero)
    x_max=np.max(x_nonzero)
    corte = erodido[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    return erodido,corte,y_min,y_max,x_min,x_max,imagem_original

img1,corte1,y_min1,y_max1,x_min1,x_max1,imagem1_original = processa_imagem('terreno6.webp')
img2,corte2,y_min2,y_max2,x_min2,x_max2,imagem2_original = processa_imagem('terreno5.jpeg')

def iguala_dimensao(imagem):
    if imagem.shape[0]>imagem.shape[1]:
        divisao= int((imagem.shape[0]-imagem.shape[1])/2)
        borderType = cv2.BORDER_CONSTANT
        value = [(0), (0), (0)]

        imagem = cv2.copyMakeBorder(imagem, 0, 0, divisao, divisao, borderType, None, value)
   

    if imagem.shape[1]>imagem.shape[0]:
        divisao= int((imagem.shape[1]-imagem.shape[0])/2)
        borderType = cv2.BORDER_CONSTANT
        value = [(0), (0), (0)]

        imagem = cv2.copyMakeBorder(imagem, divisao, divisao, 0, 0, borderType, None, value)
   

    return imagem

corte2=iguala_dimensao(corte2)
corte2 = cv2.resize(corte2, (corte1.shape[1], corte1.shape[0]))

lower = np.array([254, 254, 254])
upper = np.array([255, 255, 255])
mask  = cv2.inRange(corte1, lower, upper)
corte1 = cv2.bitwise_and(corte1,corte1, mask=mask)
corte1[mask>0] = (0,255,0)


lower = np.array([254, 254, 254])
upper = np.array([255, 255, 255])
mask  = cv2.inRange(corte2, lower, upper)
corte2 = cv2.bitwise_and(corte2,corte2, mask=mask)
corte2[mask>0] = (0, 0, 255)

img_sum = cv2.add(corte1,corte2)

tmp = cv2.cvtColor(img_sum, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,25,cv2.THRESH_BINARY)
b, g, r = cv2.split(img_sum)
rgba = [b,g,r, alpha]
img_sum = cv2.merge(rgba,4)

mult=1
for y in range (0,img_sum.shape[0]):
    for x in range (0,img_sum.shape[1]):
        b,g,r,a=img_sum[y][x]
        if b!=0 or g!=0 or r!=0:
            imagem1_original[y_min1+y][x_min1+x][0]=imagem1_original[y_min1+y][x_min1+x][0]*int(mult*img_sum[y][x][0])
            imagem1_original[y_min1+y][x_min1+x][1]=imagem1_original[y_min1+y][x_min1+x][1]*int(mult*img_sum[y][x][1])
            imagem1_original[y_min1+y][x_min1+x][2]=imagem1_original[y_min1+y][x_min1+x][2]*int(mult*img_sum[y][x][2])


        
        
        



#imagem1_original[y_min1:y_max1,x_min1:x_max1]|=filtro1
cv2.imshow('teste',imagem1_original)
cv2.waitKey(0)
cv2.destroyAllWindows()