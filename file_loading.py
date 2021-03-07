from PIL import Image, ImageEnhance

file = "C://Users/VarMoon/PycharmProjects/NeuralNetworks3x3/letters_5x5/test_4.png"


def bitmap_of_image(image_root):
    img = Image.open(image_root)
    img.convert("1")
    img_bin = []
    width, height = img.size
    red = 0
    green = 0
    blue = 0
    for i in range(width):
        for j in range(height):
            color = img.getpixel((j, i))
            count = 0
            for num in color:
                if count == 0:
                    red += num
                elif count == 1:
                    green += num
                elif count == 2:
                    blue += num
                count += 1
    red = red / (32 * 32)
    green = green / (32 * 32)
    blue = blue / (32 * 32)

    for i in range(width):
        img_bin.append([])
        for j in range(height):
            color = img.getpixel((j, i))
            counter = 0
            temp = 0
            average = 0
            for num in color:
                if temp == 0:
                    if red + 14 > num > red - 14:
                        counter += 1
                elif temp == 1:
                    if green + 14 > num > green - 14:
                        counter += 1
                else:
                    if blue + 14 > num > blue - 14:
                        counter += 1
                temp += 1
            if counter == 3 or counter == 2:
                img_bin[i].append(0)
            else:
                img_bin[i].append(1)
    return img_bin


bitmaps = bitmap_of_image(file)
print("yeey")
