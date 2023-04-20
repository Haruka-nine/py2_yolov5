import torch
import gradio as  gr

model = torch.hub.load("../","custom",path="./weight/best.pt", source="local")

title = "基于Gradio的YOLOv5演示项目"

desc = "基于Gradio的YOLOv5项目，对apex假人进行识别"

base_conf,base_iou = 0.25,0.45

def det_image(img,conf,iou):
    model.conf = conf
    model.iou = iou
    return model(img).render()[0]

gr.Interface(inputs=[gr.Webcam(),gr.Slider(minimum=0,maximum=1,value=base_conf),gr.Slider(minimum=0,maximum=1,value=base_iou)],
             outputs=["image"],
             fn=det_image,
             title=title,
             description=desc,
             live=True, #动态更新，不需要每次都submit
             examples=[["./data/train/images/_5__640x640__jpg.rf.0f8063a8cdea2bba63d8c2d9126d2277.jpg",base_conf,base_iou]]
             ).launch(share=True)   #填入这个share=True，就会产生一个公网网址