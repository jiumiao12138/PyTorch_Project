from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("image/cover.jpg")

trans_tensor = transforms.ToTensor()
trans_img = trans_tensor(img)
writer.add_image("ToTenser", trans_img)

trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(trans_img)
writer.add_image("norm", img_norm)

print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
writer.add_image("resize", img_resize)
print(img_resize.size)

trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_tensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize_2", img_resize_2)

trans_random = transforms.RandomCrop((500, 1000))
trans_compose = transforms.Compose([trans_random, trans_tensor])
for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image("Random", img_crop, i)

writer.close()