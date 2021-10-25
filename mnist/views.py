from io import BytesIO

import torch
from django.shortcuts import render, HttpResponse
from PIL import Image
from torchvision import transforms as T

from .custom_mnist import Net
from base64 import b64encode


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net()
model.load_state_dict(torch.load('weights/model.pt', map_location=torch.device(device)))
model.eval()


transform = T.Compose([
    T.Resize(28),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

def index(request):
    return render(request, "index.html")


def predict(request):
    if request.method=='POST':
        im_bytes = request.FILES.get('image').read()
        im = Image.open(BytesIO(im_bytes))

        im = transform(im).unsqueeze(0)
        rand_num = request.POST.get('number')
        rand_num = torch.tensor(int(rand_num), dtype=torch.float).reshape(1,1)
        pred_lab, pred_sum, pred_diff = model(im, rand_num)

        # return HttpResponse(
        #     "Random number: {}, \n Pred Label: {} Pred Sum: {}, diff: {}".format(
        #         rand_num.item(), pred_lab.argmax().item(), 
        #         int(round(pred_sum.item(), 0)), int(round(pred_diff.item(), 0))
        #     )
        # )
        context = {
            'image': b64encode(im_bytes).decode("utf-8"),
            'label': pred_lab.argmax().item(),
            'sum' : int(round(pred_sum.item(), 0)),
            'diff': int(round(pred_diff.item(), 0))
        }
        return render(request, 'output.html', context)
        
    return render(request, "try.html")


def about(request):
    return render(request, "about.html")


