import numpy as np
import torch
from utils import set_seed
from generator import Generator 
from discriminator import Discriminator 
from flask import Flask, request, render_template, redirect
from wtforms import Form, FloatField, IntegerField, SubmitField, validators
import io 
import base64 
from PIL import Image 


import os

# 重みのエポック数を入力
weight_epochs = 541
G = Generator()
D = Discriminator()
G.load_state_dict(torch.load(f'./netG_epoch_{str(weight_epochs)}.pth', map_location=torch.device('cpu')))
D.load_state_dict(torch.load(f'./netD_epoch_{str(weight_epochs)}.pth', map_location=torch.device('cpu')))

def generate():
    noise = torch.randn(1, 100, 1, 1)
    img_tensor = G(noise)
    pred = D(img_tensor).sigmoid().item()
    return img_tensor, pred 

def imtensor2imarray(img_tensor:torch.tensor):
    img_array = img_tensor.squeeze(0).detach().numpy().transpose(1, 2, 0)
    return img_array

def array2base64(imarray:np.ndarray):
    scaled_array = (imarray + 1.0) * 127.5
    uint8_img = np.round(scaled_array).astype(np.uint8)
    img = Image.fromarray(uint8_img)
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64
    

class ConfigForm(Form):
    Seed = IntegerField('シード',
                        [validators.InputRequired(message='0以上の整数を入力してください。'),
                         validators.NumberRange(min=0, message='0以上の整数を入力してください。')])     
    Threshold = FloatField('判別機の閾値(0 〜 1)',
                           [validators.InputRequired(message='0以上1未満の値を入力してください。'),
                            validators.NumberRange(min=0, max=1, message='0以上1未満の値を入力してください。')])
    Repeat = IntegerField('希望枚数',
                          [validators.InputRequired(message='1以上の整数を入力してください。'),
                           validators.NumberRange(min=1, message='1以上の整数を入力してください。')])
    MaxGenerate = IntegerField('最大生成回数',
                          [validators.InputRequired(message='1以上の整数を入力してください。'),
                           validators.NumberRange(min=1, message='1以上の整数を入力してください。')])
    submit = SubmitField('生成する')
    

# Flaskのインスタンスを作成
app = Flask(__name__)

# URLにアクセスがあった場合の挙動の設定
@app.route('/', methods=['GET', 'POST'])
def Generates():
    cfg_form = ConfigForm(request.form)
    # GET メソッドのとき
    if request.method == 'GET':
        print('----GET----')
        return render_template('index.html', forms=cfg_form)
    # POST メソッドのとき
    elif request.method == 'POST':
        print('----POST----')
        # ConfigFormの入力がおかしいとき
        if cfg_form.validate() == False:
            return render_template('index.html', forms=cfg_form)
        # ConfigFormの入力が正しいとき
        else:
            seed = int(request.form['Seed'])
            threshold = float(request.form['Threshold'])
            repeat = int(request.form['Repeat'])
            max_generate = int(request.form['MaxGenerate'])
            print('1 ---- get forms----')
            
            set_seed(seed)
            img_preds_ = []
            for _ in range(max_generate):
                img_tensor, predict_ = generate()     
                print('2 ----Generate----')
                img_array = imtensor2imarray(img_tensor)
                print('3 ----tensor -> array----')
                img_b64 = array2base64(img_array)
                print('4 ----array -> b64----')
                img_b64_data = f'data:image/png;base64,{img_b64}'
                if predict_ > threshold:
                    img_preds_.append((img_b64_data, predict_))
                    print('---add image---')

                # judge_ = '本物です！' if predict_ > 0.5 else '偽物です・・・'
            return render_template('result.html', img_preds=img_preds_)


if __name__ == '__main__':
    print('app run!')
    app.run(debug=True)
    
            
            
        