from PIL import Image, ImageEnhance

file = "C://Users/VarMoon/PycharmProjects/NeuralNetworks3x3/n_letter.png"

def bitmap_of_image(image_root):
    img = Image.open(file)
    img.convert("L")
    img_bin = []
    width, height = img.size
    for i in range(width):
        img_bin.append([])
        for j in range(height):
            color = img.getpixel((j, i))
            counter = 0
            for num in color:
                if num < 140:
                    counter += 1
            if counter == 3 or counter == 2:
                img_bin[i].append(1)
            else:
                img_bin[i].append(0)
    return img_bin


image_bin = bitmap_of_image(file)
img = Image.new('RGB', (28, 28), "black")
pixels = img.load()
for i in range(0, 28):
    for j in range(0, 28):
        if image_bin[i][j] == 0:
            pixels[j, i] = (255, 255, 255)
img.show()

print("yeey")
