import argparse
import sys
import cv2 as cv
import os
import numpy as np

def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('-m', '--mode', help='Choose mode (image/imgproc)', dest='mode', default='image')
  parser.add_argument('-f', '--func', choices=['grayscale', 'resize', 'sepia', 'vignette', 'pixelation'],dest='func')
  parser.add_argument('-i', '--image',type=str,help='Path to input image', dest='image_path')
  parser.add_argument('-o', '--output', type=str, help='Output image name', dest='output_image',default='image_out.jpg',)

  args = parser.parse_args()
  return args

def highgui_samples(image_path):
  img = cv.imread(image_path)

  if img is None:
    print("Could not open or find the image")
    return

  cv.imshow('Init image', img)
  while True:
    key = cv.waitKey(0)
    if key == 27:  # Код клавиши Esc
      print("Escape pressed, closing window.")
      break

  cv.destroyAllWindows()

def generate_unique_filename(filename):
  base, ext = os.path.splitext(filename)
  count = 1

  while os.path.exists(filename):
    filename = f"{base}_{count}{ext}"
    count += 1
    
  return filename

# Функция перевода изображения в оттенки серого
def grayscale(image_path, output_image):
  img = cv.imread(image_path)

  if img is None:
    print("Could not open or find the image")
    return

  height, width, n = img.shape
  for i in range(height):
    for j in range(width):
      img[i,j] = 0.3*img[i,j,2] + 0.59*img[i,j,1] + 0.11*img[i,j,0]

  cv.imshow('Gray image', img)
  output_image = generate_unique_filename(output_image)
  cv.imwrite(output_image, img)
  
  while True:
    key = cv.waitKey(0)
    if key == 27:  # Код клавиши Esc
      print("Escape pressed, closing window.")
      break
  
  cv.destroyAllWindows()

# Функция изменения разрешения изображения
def resize(image_path, output_image):
  img = cv.imread(image_path)

  if img is None:
    print("Could not open or find the image")
    return

  height, width, n = img.shape
  scaleX, scaleY = 2, 2

  resize_img = np.zeros((height//scaleX, width//scaleY, n),np.uint8)

  B = img[:,:,0]
  G = img[:,:,1]
  R = img[:,:,2]

  resizeB = B[1::scaleX,1::scaleY]
  resizeG = G[1::scaleX,1::scaleY]
  resizeR = R[1::scaleX,1::scaleY]

  resize_img[:,:,0] = resizeB
  resize_img[:,:,1] = resizeG
  resize_img[:,:,2] = resizeR

  cv.imshow('Resize image', resize_img)
  output_image = generate_unique_filename(output_image)
  cv.imwrite(output_image, resize_img)

  print(f'Original shape: {img.shape}')
  print(f'Resized shape: {resize_img.shape}')

  while True:
    key = cv.waitKey(0)
    if key == 27:  # Код клавиши Esc
      print("Escape pressed, closing window.")
      break

  cv.destroyAllWindows()

# Функция применения фотоэффекта сепии к изображению
def sepia(image_path, output_image):
  img = cv.imread(image_path)

  if img is None:
    print("Could not open or find the image")
    return

  height, width = img.shape[:2]
  for i in range(height):
    for j in range(width):
      newR = int(0.393*img[i,j,2] + 0.769*img[i,j,1] + 0.189*img[i,j,0])
      newG = int(0.349*img[i,j,2] + 0.686*img[i,j,1] + 0.168*img[i,j,0])
      newB = int(0.272*img[i,j,2] + 0.534*img[i,j,1] + 0.131*img[i,j,0])

      if newR > 255:
        newR = 255
      if newG > 255:
        newG = 255
      if newB > 255:
        newB = 255

      img[i,j,2] = newR
      img[i,j,1] = newG
      img[i,j,0] = newB

  cv.imshow('Sepia image', img)
  output_image = generate_unique_filename(output_image)
  cv.imwrite(output_image, img)

  while True:
    key = cv.waitKey(0)
    if key == 27:  # Код клавиши Esc
      print("Escape pressed, closing window.")
      break

  cv.destroyAllWindows()

def create_vignette_mask(height, width, sigma):
  centerX = width // 2
  centerY = height // 2
    
  mask = np.zeros((height, width), dtype=np.float32)

  for i in range(height):
    for j in range(width):
      dx = (j - centerX) ** 2
      dy = (i - centerY) ** 2
      distance = np.sqrt(dx + dy)

      mask[i, j] = np.exp(-(distance ** 2) / ( sigma ** 2)) # гауссово значение

  mask = mask / np.max(mask)

  return mask

# Функция применения фотоэффекта виньетки к изображению
def vignette(image_path, output_image):
  img = cv.imread(image_path)

  if img is None:
    print("Could not open or find the image")
    return

  height, width, n = img.shape
  mask = create_vignette_mask(height, width, 800)

  for i in range(n):
    img[:,:,i] = img[:,:,i] * mask

  cv.imshow('Image with a vignette', img)
  output_image = generate_unique_filename(output_image)
  cv.imwrite(output_image, img)

  while True:
    key = cv.waitKey(0)
    if key == 27:  # Код клавиши Esc
      print("Escape pressed, closing window.")
      break

  cv.destroyAllWindows()

# Функция пикселизации заданной прямоугольной области изображения
def pixelation(image_path, output_image):
  img = cv.imread(image_path)

  if img is None:
    print("Could not open or find the image")
    return

  # Определяем координаты прямоугольной области (пикселизируем центральную область)
  height, width = img.shape[:2]
  top_left = (width // 4, height // 4)  # Верхний левый угол
  bottom_right = (3 * width // 4, 3 * height // 4)  # Нижний правый угол

  x1, y1 = top_left
  x2, y2 = bottom_right
  block_size = 30

  output = img.copy()

  for i in range(y1, y2, block_size):
    for j in range(x1, x2, block_size):
      block_y2 = min(i + block_size, y2)
      block_x2 = min(j + block_size, x2)

      block = output[i:block_y2, j:block_x2]
      avg_color = np.mean(block, axis=(0, 1), dtype=int)
      output[i:block_y2, j:block_x2] = avg_color

  cv.imshow('Image with a pixelation aria', output)
  output_image = generate_unique_filename(output_image)
  cv.imwrite(output_image, output)

  while True:
    key = cv.waitKey(0)
    if key == 27:  # Код клавиши Esc
      print("Escape pressed, closing window.")
      break

  cv.destroyAllWindows()

def main():
  args = parse_arguments()
    
  if args.mode == 'image':
    highgui_samples(args.image_path)
  elif args.mode == 'imgproc':
    if args.func == 'grayscale':
      grayscale(args.image_path, args.output_image)
    elif args.func =='resize':
      resize(args.image_path, args.output_image)
    elif args.func =='sepia':
      sepia(args.image_path, args.output_image)
    elif args.func == 'vignette':
      vignette(args.image_path, args.output_image)
    elif args.func == 'pixelation':
      pixelation(args.image_path, args.output_image)
  else:
    raise 'Unsupported \'mode\' value'


if __name__ == '__main__':
  sys.exit(main() or 0)